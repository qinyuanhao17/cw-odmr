import sys
import time
import nidaqmx
import pythoncom
import pyvisa
import cw_odmr_ui
import pandas as pd
import numpy as np
import pyqtgraph as pg

from threading import Thread
from nidaqmx.constants import *
from nidaqmx.stream_readers import CounterReader
from pulsestreamer import PulseStreamer, Sequence, OutputState, findPulseStreamers
from PyQt5 import QtGui
from PyQt5.QtGui import QIcon, QPixmap, QCursor, QColor
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtWidgets import QWidget, QApplication, QGraphicsDropShadowEffect, QFileDialog, QDesktopWidget, QVBoxLayout

class ConfigureChannels():
    def __init__(self):
        super().__init__()
        self._pulser_channels = {
            'ch_aom': 0, # output channel 0: AOM control
            'ch_switch': 1, # output channel 1: MW switch control
            'ch_tagger': 2, # output channel 2 
            'ch_sync': 3, # output channel 3
            'ch_daq': 4, # NI gate channel
            'ch_mw_source': 5 # N5181A frequency change channel
        }
        self._timetagger_channels = {
            'click_channel': 1,
            'start_channel':2,
            'next_channel':-2,
            'sync_channel':tt.CHANNEL_UNUSED,
        }   
        self._ni_6363_channels = {
            'apd_channel':'/Dev2/PFI0',
            'clock_channel':'/Dev2/PFI1',
            'odmr_ctr_channel':'/Dev2/ctr0'
        } 
    @property
    def pulser_channels(self):
        return self._pulser_channels
    @property
    def timetagger_channels(self):
        return self._timetagger_channels
    @property
    def ni_6363_channels(self):
        return self._ni_6363_channels
class Hardware():
    def __init__(self):
        super().__init__()

    def pulser_generate(self):
        devices = findPulseStreamers()
        # DHCP is activated in factory settings
        if devices !=[]:
            ip = devices[0][0]
        else:
            # if discovery failed try to connect by the default hostname
            # IP address of the pulse streamer (default hostname is 'pulsestreamer')
            print("No Pulse Streamer found")

        #connect to the pulse streamer
        pulser = PulseStreamer(ip)

        # Print serial number and FPGA-ID

        return pulser
    def daq_task_generate(self, apd_channel, odmr_ctr_channel, **kwargs):
        task = nidaqmx.Task()
        channel = task.ci_channels.add_ci_count_edges_chan(
            counter=odmr_ctr_channel,
            edge=Edge.RISING,
            count_direction=CountDirection.COUNT_UP
        )
        channel.ci_count_edges_term = apd_channel
        channel.ci_count_edges_active_edge = Edge.RISING
        
        return task, channel
    
class MyWindow(cw_odmr_ui.Ui_Form, QWidget):

    rf_info_msg = pyqtSignal(str)
    pulse_streamer_info_msg = pyqtSignal(str)
    data_processing_info_msg = pyqtSignal(str)


    def __init__(self):

        super().__init__()

        # init UI
        self.setupUi(self)
        self.ui_width = int(QDesktopWidget().availableGeometry().size().width()*0.75)
        self.ui_height = int(QDesktopWidget().availableGeometry().size().height()*1)
        self.resize(self.ui_width, self.ui_height)
        center_pointer = QDesktopWidget().availableGeometry().center()
        x = center_pointer.x()
        y = center_pointer.y()
        old_x, old_y, width, height = self.frameGeometry().getRect()
        self.move(int(x - width / 2), int(y - height / 2))

        # set flag off and widget translucent
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # set window blur
        self.render_shadow()
        
        # init window button signal
        self.window_btn_signal()
        
        '''
        RF init
        '''
        # Init RF combobox ui
        self.rf_cbx_test()
        
        # Init RF setup info ui
        self.rf_info_ui()

        # Init RF signal
        self.my_rf_signal()


        '''
        Configure channels
        '''
        channel_config = ConfigureChannels()
        pulser_channels = channel_config.pulser_channels
        daq_channels = channel_config.ni_6363_channels
        self._channels = {**pulser_channels, **daq_channels}

        '''
        ODMR init
        '''
        self.hardware = Hardware()
        self.pulser_singal_init()
        self.pulser_info_ui()
        self.pulse_streamer_info_msg.connect(self.pulser_slot)
        self.pulser_daq_on_activate()
        
        '''
        Data processing init
        '''
        self.plot_ui_init()
        self.data_processing_signal()
        self.data_processing_info_ui()

    def data_processing_signal(self):
        self.restore_view_btn.clicked.connect(self.restore_view)
        # Message signal
        self.data_processing_info_msg.connect(self.data_processing_slot)
        # Scroll area updating signal
        self.data_processing_scroll.verticalScrollBar().rangeChanged.connect(
            lambda: self.data_processing_scroll.verticalScrollBar().setValue(
                self.data_processing_scroll.verticalScrollBar().maximum()
            )
        )
        # plot signal
        self.plot_data_btn.clicked.connect(self.plot_result)
        self.repeat_count_num.valueChanged.connect(self.plot_result)
        self.save_plot_data_btn.clicked.connect(self.save_plot_data)
        # clear all signal
        self.clear_repeat_count_btn.clicked.connect(self.clear_repeat_count)
    def clear_repeat_count(self):
        self.repeat_count_num.setValue(0)
    def save_plot_data(self):
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, 'Choose Data File Path', r"d:", 'CSV Files (*.csv);;All Files (*)', options=options)
        startFreq = int(self.start_freq_spbx.value())
        stopFreq = int(self.stop_freq_spbx.value())
        stepFreq = float(self.step_freq_spbx.value())
        intensity_data = self.intensity_data
        
        frequency_data = np.arange(startFreq,stopFreq+stepFreq,stepFreq)
        df = pd.DataFrame({'Frequency': frequency_data, 'Intensity': intensity_data})
        df.to_csv(file_path, index=False, header=True)
    def plot_result(self):
        self.cw_odmr_plot.clear()
        startFreq = int(self.start_freq_spbx.value())
        stopFreq = int(self.stop_freq_spbx.value())
        stepFreq = float(self.step_freq_spbx.value())
        num_points = int((stopFreq - startFreq)/stepFreq) + 1
        freq_data = list(np.arange(startFreq,stopFreq+stepFreq,stepFreq))
        curve = self.cw_odmr_plot.plot(pen=pg.mkPen(color=(255,85,48), width=2))

        i_count = int(self.repeat_count_num.value())
        cw_odmr_data = np.array(self.cw_odmr_data[0:i_count*num_points])
        cw_odmr_data = cw_odmr_data.reshape(-1,num_points)
        self.intensity_data = np.sum(cw_odmr_data, axis=0)
        # self.intensity_data = list(self.intensity_data)
        
        curve.setData(freq_data, list(self.intensity_data))   
        # self.intensity_data = np.array(self.intensity_data)
    def data_processing_info_ui(self):

        self.data_processing_msg.setWordWrap(True)  # 自动换行
        self.data_processing_msg.setAlignment(Qt.AlignTop)  # 靠上

        # # 用于存放消息
        self.data_processing_msg_history = []

    def data_processing_slot(self, msg):

        # print(msg)
        self.data_processing_msg_history.append(msg)
        self.data_processing_msg.setText("<br>".join(self.data_processing_msg_history))
        self.data_processing_msg.resize(700, self.data_processing_msg.frameSize().height() + 20)
        self.data_processing_msg.repaint()  # 更新内容，如果不更新可能没有显示新内容

    def plot_ui_init(self):

        # Add pyqtGraph plot widget        
        self.cw_odmr_plot = pg.PlotWidget(enableAutoRange=True)
        graph_widget_layout = QVBoxLayout()
        graph_widget_layout.addWidget(self.cw_odmr_plot)
        self.graph_frame.setLayout(graph_widget_layout)
        self.cw_odmr_plot.setLabel("left","Intensity (Counts)")
        self.cw_odmr_plot.setLabel("bottom","RF Frequency (MHz)")
        self.cw_odmr_plot.setTitle('CW-ODMR', color='k')
        self.cw_odmr_plot.setBackground(background=None)
        self.cw_odmr_plot.getAxis('left').setPen('k')
        self.cw_odmr_plot.getAxis('left').setTextPen('k')
        self.cw_odmr_plot.getAxis('bottom').setPen('k')
        self.cw_odmr_plot.getAxis('bottom').setTextPen('k')
        self.cw_odmr_plot.getAxis('top').setPen('k')
        self.cw_odmr_plot.getAxis('right').setPen('k')
        self.cw_odmr_plot.showAxes(True)
        self.cw_odmr_plot.showGrid(x=True, y=True)

    def pulser_singal_init(self):
        # ASG scroll area scrollbar signal
        self.pulser_scroll.verticalScrollBar().rangeChanged.connect(
            lambda: self.pulser_scroll.verticalScrollBar().setValue(
                self.pulser_scroll.verticalScrollBar().maximum()
            )
        )
        self.set_pulser_count_btn.clicked.connect(self.set_pulse_and_count)
        self.odmr_start_btn.clicked.connect(self.odmr_start)
        self.odmr_stop_btn.clicked.connect(self.odmr_stop)
    def pulser_daq_on_activate(self):
        '''
        Pusler Init
        '''
        self.pulser = self.hardware.pulser_generate()
        '''
        DAQ Init
        '''
        self.task, self.odmr_ctr_channel = self.hardware.daq_task_generate(**self._channels)
        self.pulse_streamer_info_msg.emit('DAQ Counter channel: '+self.odmr_ctr_channel.channel_names[0])
        self.pulse_streamer_info_msg.emit('DAQ APD channel: '+self.odmr_ctr_channel.ci_count_edges_term)

    def pulser_daq_on_deactivate(self):
        self.pulser.reset()
        self.task.stop()
        self.task.close()
    def odmr_stop(self):
        self._stopConstant = True
     
    def odmr_start(self):
        self.pulser.reset()
        self.task.start()
        self._odmr_data_container = np.array([])
        self._stopConstant = False
        time.sleep(0.5)
        final = OutputState([self._channels['ch_aom']],0,0)
        self.pulser.stream(self.seq, -1, final)
        # Start daq in thread
        thread = Thread(
            target=self.count_data_thread_func
        )
        thread.start()
    def count_data_thread_func(self):
        n_sample = int(self.sample_spbx.value())
        number_of_samples = n_sample*4
        data_array = np.zeros(number_of_samples,dtype=np.uint32)
        while True:
            self.reader.read_many_sample_uint32(
                data=data_array,
                number_of_samples_per_channel=number_of_samples,
                timeout=10
            )
            if self._odmr_data_container.size == 0:
                self._odmr_data_container = data_array
            else:
                self._odmr_data_container = np.vstack(self._odmr_data_container, data_array).
            if self._stopConstant == True:
                break

    def start_stop_step(self):
        start = int(self.start_freq_spbx.value())
        stop = int(self.stop_freq_spbx.value())
        step = int(self.step_freq_spbx.value())
        num_points = int((stop - start)/step) + 1
        return start, stop, step, num_points
    def set_pulse_and_count(self, ch_aom, ch_switch, ch_daq, ch_mw_source):
        print(ch_aom, ch_switch, ch_daq, ch_mw_source)
        start, stop, step, num_points = self.start_stop_step()
        mw_on = int(1E6)*int(self.mw_time_spbx.value()) # in ms
        mw_off = mw_on #1ms
        daq_high = 1000 # 1us
        daq_wait = 1000 # 1us
        n_sample = int(self.sample_spbx.value())
        #define digital levels
        HIGH=1
        LOW=0
        seq_aom=[(mw_on+mw_off,HIGH)]*n_sample
        seq_switch=[(mw_on,HIGH),(mw_off,LOW)]*n_sample
        seq_daq=[(daq_wait,LOW),(daq_high,HIGH),(mw_on-2*daq_high-daq_wait,LOW),(daq_high,HIGH),(daq_wait,LOW),(daq_high,HIGH),(mw_off-2*daq_high-daq_wait,LOW),(daq_high,HIGH)]*n_sample
        seq_mw_source = [((mw_on+mw_off)*n_sample-daq_wait,LOW),(daq_wait,HIGH)]
        #create the sequence
        self.seq = Sequence()
        
        #set digital channels
        self.seq.setDigital(ch_aom, seq_aom)
        self.seq.setDigital(ch_switch, seq_switch)
        self.seq.setDigital(ch_daq, seq_daq)
        self.seq.setDigital(ch_mw_source, seq_mw_source)

        self.seq.plot()
        self.task.timing.cfg_samp_clk_timing(
            rate=2E6,
            source='/Dev2/PFI1',
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=4*n_sample
        )
        self.pulse_streamer_info_msg.emit('Counter input channel: '+self.odmr_ctr_channel.ci_count_edges_term)
        
        self.reader = CounterReader(self.task.in_stream)
        
    def pulser_info_ui(self):

        self.pulser_msg.setWordWrap(True)  # 自动换行
        self.pulser_msg.setAlignment(Qt.AlignTop)  # 靠上
        self.pulser_msg_history = []

    def pulser_slot(self, msg):

        # print(msg)
        self.pulser_msg_history.append(msg)
        self.pulser_msg.setText("<br>".join(self.asg_msg_history))
        self.pulser_msg.resize(700, self.asg_msg.frameSize().height() + 20)
        self.pulser_msg.repaint()  # 更新内容，如果不更新可能没有显示新内容



    '''Set window ui'''
    def window_btn_signal(self):
        # window button sigmal
        self.close_btn.clicked.connect(self.close)
        self.max_btn.clicked.connect(self.maxornorm)
        self.min_btn.clicked.connect(self.showMinimized)
        
    #create window blur
    def render_shadow(self):
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setOffset(0, 0)  # 偏移
        self.shadow.setBlurRadius(30)  # 阴影半径
        self.shadow.setColor(QColor(128, 128, 255))  # 阴影颜色
        self.mainwidget.setGraphicsEffect(self.shadow)  # 将设置套用到widget窗口中

    def maxornorm(self):
        if self.isMaximized():
            self.showNormal()
            self.norm_icon = QIcon()
            self.norm_icon.addPixmap(QPixmap(":/my_icons/images/icons/max.svg"), QIcon.Normal, QIcon.Off)
            self.max_btn.setIcon(self.norm_icon)
        else:
            self.showMaximized()
            self.max_icon = QIcon()
            self.max_icon.addPixmap(QPixmap(":/my_icons/images/icons/norm.svg"), QIcon.Normal, QIcon.Off)
            self.max_btn.setIcon(self.max_icon)

    def mousePressEvent(self, event):

        if event.button() == Qt.LeftButton:
            self.m_flag = True
            self.m_Position = QPoint
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))  # 更改鼠标图标
        
    def mouseMoveEvent(self, QMouseEvent):
        m_position = QPoint
        m_position = QMouseEvent.globalPos() - self.pos()
        width = QDesktopWidget().availableGeometry().size().width()
        height = QDesktopWidget().availableGeometry().size().height()
        if m_position.x() < width*0.7 and m_position.y() < height*0.06:
            self.m_flag = True
            if Qt.LeftButton and self.m_flag:                
                pos_x = int(self.m_Position.x())
                pos_y = int(self.m_Position.y())
                if pos_x < width*0.7 and pos_y < height*0.06:           
                    self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
                    QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))

    '''
    RF CONTROL
    '''
    def rf_info_ui(self):

        self.rf_msg.setWordWrap(True)  # 自动换行
        self.rf_msg.setAlignment(Qt.AlignTop)  # 靠上
        self.rf_msg_history = []

    def rf_slot(self, msg):

        # print(msg)
        self.rf_msg_history.append(msg)
        self.rf_msg.setText("<br>".join(self.rf_msg_history))
        self.rf_msg.resize(700, self.rf_msg.frameSize().height() + 20)
        self.rf_msg.repaint()  # 更新内容，如果不更新可能没有显示新内容

    def my_rf_signal(self):

        #open button signal
        self.rf_connect_btn.clicked.connect(self.boot_rf)

        #message signal
        self.rf_info_msg.connect(self.rf_slot)

        # RF scroll area scrollbar signal
        self.rf_scroll.verticalScrollBar().rangeChanged.connect(
            lambda: self.rf_scroll.verticalScrollBar().setValue(
                self.rf_scroll.verticalScrollBar().maximum()
            )
        )

        # combobox restore signal
        self.rf_visa_rst_btn.clicked.connect(self.rf_cbx_test)

        # RF On button signal
        self.rf_ply_stp_btn.clicked.connect(self.rf_ply_stp)
        # List Mode
        self.init_list_btn.clicked.connect(self.init_list)
        self.list_ply_stp_btn.clicked.connect(self.list_ply_stp)

    def rf_cbx_test(self):
        
        self.rf_cbx.clear()
        self.rm = pyvisa.ResourceManager()
        self.ls = self.rm.list_resources()
        self.rf_cbx.addItems(self.ls)

    def boot_rf(self):
        
        # Boot RF generator
        self.rf_port = self.rf_cbx.currentText()
        # print(self.rf_port)
        self._gpib_connection = self.rm.open_resource(self.rf_port)
        self._gpib_connection.write_termination = '\n'
        instrument_info = self._gpib_connection.query('*IDN?')
        
        # # 恢复出厂设置
        # self.fac = self.my_instrument.write(':SYST:PRES:TYPE FAC')
        
        # self.preset = self.my_instrument.write(':SYST:PRES')
        self._gpib_connection.write(':OUTPut:STATe OFF') # switch off the output
        self._gpib_connection.write('*RST')

        self.rf_info_msg.emit(repr(instrument_info))
        
        '''
        This part defines some initial settings of RF generator suited to CW-ODMR measurement
        '''
        # time.sleep(5)
    def list_ply_stp(self):
        output_status = self._gpib_connection.query(':OUTPut:STATe?')
        
        if output_status == '0\n':
            self.list_ply_stp_btn.setText('List Off')
            self.off_icon = QIcon()
            self.off_icon.addPixmap(QPixmap(":/my_icons/images/icons/stop.svg"), QIcon.Normal, QIcon.Off)
            self.list_ply_stp_btn.setIcon(self.off_icon)
            rtn = self._gpib_connection.write(':OUTPut:STATe ON')
            if rtn != 0:
                self.rf_info_msg.emit('List ON succeeded: {}'.format(rtn))
            else:
                self.rf_info_msg.emit('List ON failed')
                sys.emit()
        elif output_status == '1\n':
            self.list_ply_stp_btn.setText('List On')
            self.on_icon = QIcon()
            self.on_icon.addPixmap(QPixmap(":/my_icons/images/icons/play.svg"), QIcon.Normal, QIcon.Off)
            self.list_ply_stp_btn.setIcon(self.on_icon)
            rtn = self._gpib_connection.write(':OUTPut:STATe OFF')
            if rtn != 0:
                self.rf_info_msg.emit('List OFF succeeded: {}'.format(rtn))
            else:
                self.rf_info_msg.emit('List OFF failed')
                sys.emit()
    def init_list(self):
        start_freq = int(int(self.start_freq_spbx.value())*1e6)
        stop_freq = int(int(self.stop_freq_spbx.value())*1e6)
        step_freq = int(int(self.step_freq_spbx.value())*1e6)
        freq = range(start_freq,stop_freq+step_freq,step_freq)
        power = float(self.list_power_spbx.value())
        # setting sweep type to list
        self._gpib_connection.write(':LIST:TYPE:LIST:INIT:PRES')
        self._gpib_connection.write(':LIST:TYPE LIST')
        self._gpib_connection.write(':FREQ:MODE LIST')
        self._gpib_connection.write(':POW:MODE LIST')
        # set external trigger source for START a scan
        self._gpib_connection.write(':TRIGger:SOURce IMM')
        self._gpib_connection.write(':TRIGger:SLOP POS')
        
        # set a external trigger source to move the points in the list
        self._gpib_connection.write(':LIST:TRIGger:SOURce EXT')
        self._gpib_connection.write(':LIST:DIRection UP')

        if freq is not None:
            flist = '{0:f}'.format(freq[0])
            for ii in freq[1:-1]:
                flist += ', {0:f}'.format(ii)
            flist += ', {0:f}'.format(freq[-1]) # last frequency
            rtn = self._gpib_connection.write(':LIST:FREQ ' + flist)
            self.rf_info_msg.emit('List Freq set succeeded: {}'.format(rtn))
        if power is not None:
            plist = (len(freq)-1) * (str(power)+', ') + str(power)
            rtn = self._gpib_connection.write(':LIST:POW ' + plist)
            self.rf_info_msg.emit('List Power set succeeded: {}'.format(rtn))
        # because the device require to fill in a dwelling values in the list, following is to set an arbitrary time
        # this arbitrary time is not used in the sweeping because external signal is used to move the points in list
        point_dwell_time = int(self.dwell_time_spbx.value())/1000
        dlist = (len(freq)-1)*(str(point_dwell_time)+', ') + str(point_dwell_time)
        self._gpib_connection.write(':LIST:DWEL ' + dlist)
        rtn = self._gpib_connection.write(':LIST:DWEL:TYPE LIST')
        self.rf_info_msg.emit('List Dwelltime set succeeded: {}'.format(rtn))
        # self._gpib_connection.write(':LIST:RETR ON') #Sweep will return to first of the list after one sweep
        # arm the device by INIT:
        self._gpib_connection.write(':INITiate:CONTinuous ON')
        

    def rf_ply_stp(self):
        output_status = self._gpib_connection.query(':OUTPut:STATe?')
        
        if output_status == '0\n':
            frequency = float(self.cw_freq_spbx.value())*1e6
            power = float(self.cw_power_spbx.value())
            self.rf_ply_stp_btn.setText('RF OFF')
            self.off_icon = QIcon()
            self.off_icon.addPixmap(QPixmap(":/my_icons/images/icons/stop.svg"), QIcon.Normal, QIcon.Off)
            self.rf_ply_stp_btn.setIcon(self.off_icon)
            self._gpib_connection.write(':FREQ:MODE CW')
            self._gpib_connection.write(':FREQ:CW {0:f} Hz'.format(frequency))
            self._gpib_connection.write(':POWer:AMPLitude {0:f}'.format(power))
            rtn = self._gpib_connection.write(':OUTPut:STATe ON')
            if rtn != 0:
                self.rf_info_msg.emit('RF ON succeeded: {}'.format(rtn))
            else:
                self.rf_info_msg.emit('RF ON failed')
                sys.emit()
        elif output_status == '1\n':
            self.rf_ply_stp_btn.setText('RF ON  ')
            self.on_icon = QIcon()
            self.on_icon.addPixmap(QPixmap(":/my_icons/images/icons/play.svg"), QIcon.Normal, QIcon.Off)
            self.rf_ply_stp_btn.setIcon(self.on_icon)
            rtn = self._gpib_connection.write(':OUTPut:STATe OFF')
            if rtn != 0:
                self.rf_info_msg.emit('RF OFF succeeded: {}'.format(rtn))
            else:
                self.rf_info_msg.emit('RF OFF failed')
                sys.emit()
    def closeEvent(self,event):
        self._gpib_connection.write(':OUTPut:STATe OFF')
        self._gpib_connection.close()  
        self.rm.close()
        self.pulser_daq_on_deactivate()
        return
if __name__ == '__main__':

    app = QApplication(sys.argv)
    w = MyWindow()
    w.show()
    app.exec()
