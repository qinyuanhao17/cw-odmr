# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'cw_odmr.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1484, 882)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        Form.setStyleSheet("QWidget#Form{\n"
"    border-radius:10px;\n"
"}\n"
"\n"
"QWidget#mainwidget{\n"
"    background-color: rgba(239, 244, 249, 0);\n"
"    border-radius:10px;\n"
"}\n"
"QFrame#mainframe{\n"
"    background-color: rgba(239, 244, 249, 255);\n"
"    border-radius:10px;\n"
"}\n"
"QGroupBox{\n"
"    border-width: 2px;\n"
"    \n"
"    font: 25 9pt \"Microsoft YaHei UI Light\";\n"
"}\n"
"QPushButton{\n"
"    background-color: rgb(255, 255, 255);\n"
"    border-radius: 2px;\n"
"    border: 1px groove gray;\n"
"    font: 63 9pt \"Nunito Sans SemiBold\";\n"
"}\n"
"QTabWidget{\n"
"    font: 63 9pt \"Nunito Sans SemiBold\";\n"
"}\n"
"QLabel{\n"
"    font: 9pt \"Microsoft YaHei UI\";\n"
"}\n"
"QComboBox{\n"
"    \n"
"    font: 9pt \"Menlo for Powerline\";\n"
"}\n"
"")
        self.verticalLayout_19 = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout_19.setObjectName("verticalLayout_19")
        self.mainwidget = QtWidgets.QWidget(Form)
        self.mainwidget.setStyleSheet("")
        self.mainwidget.setObjectName("mainwidget")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.mainwidget)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.mainframe = QtWidgets.QFrame(self.mainwidget)
        self.mainframe.setStyleSheet("")
        self.mainframe.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.mainframe.setFrameShadow(QtWidgets.QFrame.Raised)
        self.mainframe.setObjectName("mainframe")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.mainframe)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(self.mainframe)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setMinimumSize(QtCore.QSize(0, 40))
        self.frame.setBaseSize(QtCore.QSize(0, 0))
        self.frame.setStyleSheet("")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_2.setContentsMargins(5, 1, 1, 1)
        self.horizontalLayout_2.setSpacing(1)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap(":/my_icons/images/icons/window_title_icon.png"))
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.label = QtWidgets.QLabel(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setStyleSheet("")
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.label_6 = QtWidgets.QLabel(self.frame)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_2.addWidget(self.label_6)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.min_btn = QtWidgets.QPushButton(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.min_btn.sizePolicy().hasHeightForWidth())
        self.min_btn.setSizePolicy(sizePolicy)
        self.min_btn.setMaximumSize(QtCore.QSize(60, 30))
        self.min_btn.setStyleSheet("QPushButton {    \n"
"    border: none;\n"
"    border-radius: 2px;\n"
"    background-color: rgba(239, 244, 249, 0);\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgba(160, 160, 160,100);\n"
"}\n"
"QPushButton:pressed {    \n"
"    background-color: transparent;\n"
"}\n"
"")
        self.min_btn.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/my_icons/images/icons/min.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.min_btn.setIcon(icon)
        self.min_btn.setObjectName("min_btn")
        self.horizontalLayout_2.addWidget(self.min_btn)
        self.max_btn = QtWidgets.QPushButton(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.max_btn.sizePolicy().hasHeightForWidth())
        self.max_btn.setSizePolicy(sizePolicy)
        self.max_btn.setMaximumSize(QtCore.QSize(60, 30))
        self.max_btn.setStyleSheet("QPushButton {    \n"
"    border: none;\n"
"    border-radius: 2px;\n"
"    background-color: rgba(239, 244, 249, 0);\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgba(160, 160, 160,100);\n"
"}\n"
"QPushButton:pressed {    \n"
"    background-color: transparent;\n"
"}")
        self.max_btn.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/my_icons/images/icons/max.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.max_btn.setIcon(icon1)
        self.max_btn.setIconSize(QtCore.QSize(20, 20))
        self.max_btn.setObjectName("max_btn")
        self.horizontalLayout_2.addWidget(self.max_btn)
        self.close_btn = QtWidgets.QPushButton(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.close_btn.sizePolicy().hasHeightForWidth())
        self.close_btn.setSizePolicy(sizePolicy)
        self.close_btn.setMaximumSize(QtCore.QSize(60, 30))
        self.close_btn.setStyleSheet("QPushButton {    \n"
"    border: none;\n"
"    border-radius: 2px;\n"
"    background-color: rgba(239, 244, 249, 0);\n"
"}\n"
"QPushButton:hover {\n"
"    background-color: rgba(196, 43, 28,200);\n"
"}\n"
"QPushButton:pressed {    \n"
"    background-color: transparent;\n"
"}")
        self.close_btn.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/my_icons/images/icons/close_btn.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.close_btn.setIcon(icon2)
        self.close_btn.setObjectName("close_btn")
        self.horizontalLayout_2.addWidget(self.close_btn)
        self.horizontalLayout_2.setStretch(3, 80)
        self.horizontalLayout_2.setStretch(4, 3)
        self.horizontalLayout_2.setStretch(5, 3)
        self.horizontalLayout_2.setStretch(6, 3)
        self.verticalLayout.addWidget(self.frame)
        self.frame_2 = QtWidgets.QFrame(self.mainframe)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(100)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setStyleSheet("QPushButton:hover {\n"
"    background-color: rgba(160, 160, 160,100);\n"
"}\n"
"QPushButton:pressed {    \n"
"    background-color: transparent;\n"
"}")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.groupBox_3 = QtWidgets.QGroupBox(self.frame_2)
        self.groupBox_3.setStyleSheet("")
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_4 = QtWidgets.QLabel(self.groupBox_3)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout.addWidget(self.label_4)
        self.rf_cbx = QtWidgets.QComboBox(self.groupBox_3)
        self.rf_cbx.setObjectName("rf_cbx")
        self.horizontalLayout.addWidget(self.rf_cbx)
        self.rf_visa_rst_btn = QtWidgets.QPushButton(self.groupBox_3)
        self.rf_visa_rst_btn.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/my_icons/images/icons/restore.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.rf_visa_rst_btn.setIcon(icon3)
        self.rf_visa_rst_btn.setIconSize(QtCore.QSize(20, 25))
        self.rf_visa_rst_btn.setObjectName("rf_visa_rst_btn")
        self.horizontalLayout.addWidget(self.rf_visa_rst_btn)
        self.rf_connect_btn = QtWidgets.QPushButton(self.groupBox_3)
        self.rf_connect_btn.setStyleSheet("QPushButton{\n"
"    font: 10pt \"Yu Gothic UI\";    \n"
"\n"
"}")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/my_icons/images/icons/connect.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.rf_connect_btn.setIcon(icon4)
        self.rf_connect_btn.setIconSize(QtCore.QSize(20, 25))
        self.rf_connect_btn.setObjectName("rf_connect_btn")
        self.horizontalLayout.addWidget(self.rf_connect_btn)
        self.horizontalLayout.setStretch(0, 3)
        self.horizontalLayout.setStretch(1, 4)
        self.horizontalLayout.setStretch(2, 1)
        self.horizontalLayout.setStretch(3, 2)
        self.verticalLayout_6.addLayout(self.horizontalLayout)
        self.groupBox_13 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_13.setObjectName("groupBox_13")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.groupBox_13)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_17 = QtWidgets.QLabel(self.groupBox_13)
        self.label_17.setObjectName("label_17")
        self.horizontalLayout_6.addWidget(self.label_17)
        self.cw_power_spbx = QtWidgets.QDoubleSpinBox(self.groupBox_13)
        self.cw_power_spbx.setMinimum(-50.0)
        self.cw_power_spbx.setMaximum(10.0)
        self.cw_power_spbx.setSingleStep(0.1)
        self.cw_power_spbx.setProperty("value", -30.0)
        self.cw_power_spbx.setObjectName("cw_power_spbx")
        self.horizontalLayout_6.addWidget(self.cw_power_spbx)
        self.label_18 = QtWidgets.QLabel(self.groupBox_13)
        self.label_18.setObjectName("label_18")
        self.horizontalLayout_6.addWidget(self.label_18)
        self.cw_freq_spbx = QtWidgets.QDoubleSpinBox(self.groupBox_13)
        self.cw_freq_spbx.setMinimum(100.0)
        self.cw_freq_spbx.setMaximum(6000.0)
        self.cw_freq_spbx.setSingleStep(0.1)
        self.cw_freq_spbx.setProperty("value", 2860.0)
        self.cw_freq_spbx.setObjectName("cw_freq_spbx")
        self.horizontalLayout_6.addWidget(self.cw_freq_spbx)
        self.rf_ply_stp_btn = QtWidgets.QPushButton(self.groupBox_13)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/my_icons/images/icons/play.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.rf_ply_stp_btn.setIcon(icon5)
        self.rf_ply_stp_btn.setIconSize(QtCore.QSize(20, 25))
        self.rf_ply_stp_btn.setObjectName("rf_ply_stp_btn")
        self.horizontalLayout_6.addWidget(self.rf_ply_stp_btn)
        self.verticalLayout_6.addWidget(self.groupBox_13)
        self.groupBox_14 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_14.setObjectName("groupBox_14")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_14)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.label_3 = QtWidgets.QLabel(self.groupBox_14)
        self.label_3.setObjectName("label_3")
        self.gridLayout_5.addWidget(self.label_3, 1, 3, 1, 1)
        self.init_list_btn = QtWidgets.QPushButton(self.groupBox_14)
        self.init_list_btn.setStyleSheet("QPushButton{\n"
"    font: 10pt \"Yu Gothic UI\";    \n"
"\n"
"}")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/my_icons/images/icons/download.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.init_list_btn.setIcon(icon6)
        self.init_list_btn.setIconSize(QtCore.QSize(20, 25))
        self.init_list_btn.setObjectName("init_list_btn")
        self.gridLayout_5.addWidget(self.init_list_btn, 2, 3, 1, 1)
        self.stop_freq_spbx = QtWidgets.QSpinBox(self.groupBox_14)
        self.stop_freq_spbx.setMaximum(6000)
        self.stop_freq_spbx.setProperty("value", 3000)
        self.stop_freq_spbx.setObjectName("stop_freq_spbx")
        self.gridLayout_5.addWidget(self.stop_freq_spbx, 0, 5, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.groupBox_14)
        self.label_14.setObjectName("label_14")
        self.gridLayout_5.addWidget(self.label_14, 0, 3, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.groupBox_14)
        self.label_15.setObjectName("label_15")
        self.gridLayout_5.addWidget(self.label_15, 0, 1, 1, 1)
        self.step_freq_spbx = QtWidgets.QDoubleSpinBox(self.groupBox_14)
        self.step_freq_spbx.setMaximum(10.0)
        self.step_freq_spbx.setSingleStep(0.1)
        self.step_freq_spbx.setProperty("value", 1.0)
        self.step_freq_spbx.setObjectName("step_freq_spbx")
        self.gridLayout_5.addWidget(self.step_freq_spbx, 1, 2, 1, 1)
        self.start_freq_spbx = QtWidgets.QSpinBox(self.groupBox_14)
        self.start_freq_spbx.setMaximum(3000)
        self.start_freq_spbx.setProperty("value", 2700)
        self.start_freq_spbx.setObjectName("start_freq_spbx")
        self.gridLayout_5.addWidget(self.start_freq_spbx, 0, 2, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.groupBox_14)
        self.label_16.setObjectName("label_16")
        self.gridLayout_5.addWidget(self.label_16, 1, 1, 1, 1)
        self.dwell_time_spbx = QtWidgets.QSpinBox(self.groupBox_14)
        self.dwell_time_spbx.setMinimum(1)
        self.dwell_time_spbx.setMaximum(2000)
        self.dwell_time_spbx.setProperty("value", 1000)
        self.dwell_time_spbx.setObjectName("dwell_time_spbx")
        self.gridLayout_5.addWidget(self.dwell_time_spbx, 1, 5, 1, 1)
        self.list_ply_stp_btn = QtWidgets.QPushButton(self.groupBox_14)
        self.list_ply_stp_btn.setIcon(icon5)
        self.list_ply_stp_btn.setIconSize(QtCore.QSize(20, 25))
        self.list_ply_stp_btn.setObjectName("list_ply_stp_btn")
        self.gridLayout_5.addWidget(self.list_ply_stp_btn, 2, 5, 1, 1)
        self.list_power_spbx = QtWidgets.QDoubleSpinBox(self.groupBox_14)
        self.list_power_spbx.setMinimum(-50.0)
        self.list_power_spbx.setMaximum(10.0)
        self.list_power_spbx.setSingleStep(0.1)
        self.list_power_spbx.setProperty("value", 5.0)
        self.list_power_spbx.setObjectName("list_power_spbx")
        self.gridLayout_5.addWidget(self.list_power_spbx, 2, 2, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.groupBox_14)
        self.label_19.setObjectName("label_19")
        self.gridLayout_5.addWidget(self.label_19, 2, 1, 1, 1)
        self.verticalLayout_6.addWidget(self.groupBox_14)
        self.groupBox_6 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_6.setObjectName("groupBox_6")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_6)
        self.gridLayout_2.setContentsMargins(5, 5, 5, 5)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.rf_scroll = QtWidgets.QScrollArea(self.groupBox_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.rf_scroll.sizePolicy().hasHeightForWidth())
        self.rf_scroll.setSizePolicy(sizePolicy)
        self.rf_scroll.setWidgetResizable(True)
        self.rf_scroll.setObjectName("rf_scroll")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 540, 69))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.rf_msg = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.rf_msg.setText("")
        self.rf_msg.setObjectName("rf_msg")
        self.verticalLayout_7.addWidget(self.rf_msg)
        self.rf_scroll.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout_2.addWidget(self.rf_scroll, 0, 0, 1, 1)
        self.verticalLayout_6.addWidget(self.groupBox_6)
        self.verticalLayout_4.addWidget(self.groupBox_3)
        self.groupBox = QtWidgets.QGroupBox(self.frame_2)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.groupBox_8 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_8.setObjectName("groupBox_8")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox_8)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_7 = QtWidgets.QLabel(self.groupBox_8)
        self.label_7.setObjectName("label_7")
        self.gridLayout_6.addWidget(self.label_7, 0, 1, 1, 1)
        self.sample_spbx = QtWidgets.QSpinBox(self.groupBox_8)
        self.sample_spbx.setMaximum(5000)
        self.sample_spbx.setProperty("value", 1000)
        self.sample_spbx.setObjectName("sample_spbx")
        self.gridLayout_6.addWidget(self.sample_spbx, 1, 2, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.groupBox_8)
        self.label_10.setObjectName("label_10")
        self.gridLayout_6.addWidget(self.label_10, 1, 1, 1, 1)
        self.mw_time_spbx = QtWidgets.QSpinBox(self.groupBox_8)
        self.mw_time_spbx.setMaximum(100)
        self.mw_time_spbx.setProperty("value", 1)
        self.mw_time_spbx.setObjectName("mw_time_spbx")
        self.gridLayout_6.addWidget(self.mw_time_spbx, 0, 2, 1, 1)
        self.repeat_spbx = QtWidgets.QSpinBox(self.groupBox_8)
        self.repeat_spbx.setMaximum(1000)
        self.repeat_spbx.setProperty("value", 10)
        self.repeat_spbx.setObjectName("repeat_spbx")
        self.gridLayout_6.addWidget(self.repeat_spbx, 3, 2, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.groupBox_8)
        self.label_9.setObjectName("label_9")
        self.gridLayout_6.addWidget(self.label_9, 3, 1, 1, 1)
        self.set_pulser_count_btn = QtWidgets.QPushButton(self.groupBox_8)
        self.set_pulser_count_btn.setStyleSheet("QPushButton{\n"
"    font: 10pt \"Yu Gothic UI\";    \n"
"\n"
"}")
        self.set_pulser_count_btn.setIcon(icon6)
        self.set_pulser_count_btn.setIconSize(QtCore.QSize(20, 25))
        self.set_pulser_count_btn.setObjectName("set_pulser_count_btn")
        self.gridLayout_6.addWidget(self.set_pulser_count_btn, 4, 2, 1, 1)
        self.horizontalLayout_8.addWidget(self.groupBox_8)
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_4.setObjectName("groupBox_4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.groupBox_4)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.odmr_start_btn = QtWidgets.QPushButton(self.groupBox_4)
        self.odmr_start_btn.setStyleSheet("QPushButton{\n"
"    font: 10pt \"Yu Gothic UI\";    \n"
"\n"
"}")
        self.odmr_start_btn.setIcon(icon5)
        self.odmr_start_btn.setIconSize(QtCore.QSize(20, 25))
        self.odmr_start_btn.setObjectName("odmr_start_btn")
        self.horizontalLayout_3.addWidget(self.odmr_start_btn)
        self.odmr_stop_btn = QtWidgets.QPushButton(self.groupBox_4)
        self.odmr_stop_btn.setStyleSheet("QPushButton{\n"
"    font: 10pt \"Yu Gothic UI\";    \n"
"\n"
"}")
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/my_icons/images/icons/stop.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.odmr_stop_btn.setIcon(icon7)
        self.odmr_stop_btn.setIconSize(QtCore.QSize(20, 25))
        self.odmr_stop_btn.setObjectName("odmr_stop_btn")
        self.horizontalLayout_3.addWidget(self.odmr_stop_btn)
        self.horizontalLayout_8.addWidget(self.groupBox_4)
        self.verticalLayout_9.addLayout(self.horizontalLayout_8)
        self.groupBox_7 = QtWidgets.QGroupBox(self.groupBox)
        self.groupBox_7.setObjectName("groupBox_7")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_7)
        self.gridLayout_3.setContentsMargins(5, 5, 5, 5)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.pulser_scroll = QtWidgets.QScrollArea(self.groupBox_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pulser_scroll.sizePolicy().hasHeightForWidth())
        self.pulser_scroll.setSizePolicy(sizePolicy)
        self.pulser_scroll.setWidgetResizable(True)
        self.pulser_scroll.setObjectName("pulser_scroll")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 540, 146))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents_2)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.pulser_msg = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.pulser_msg.setText("")
        self.pulser_msg.setObjectName("pulser_msg")
        self.verticalLayout_8.addWidget(self.pulser_msg)
        self.pulser_scroll.setWidget(self.scrollAreaWidgetContents_2)
        self.gridLayout_3.addWidget(self.pulser_scroll, 0, 0, 1, 1)
        self.verticalLayout_9.addWidget(self.groupBox_7)
        self.verticalLayout_4.addWidget(self.groupBox)
        self.horizontalLayout_5.addLayout(self.verticalLayout_4)
        self.groupBox_9 = QtWidgets.QGroupBox(self.frame_2)
        self.groupBox_9.setObjectName("groupBox_9")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox_9)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox_9)
        self.groupBox_5.setObjectName("groupBox_5")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_5)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.line_2 = QtWidgets.QFrame(self.groupBox_5)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout_2.addWidget(self.line_2)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setSpacing(5)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.repeat_num_cbx = QtWidgets.QComboBox(self.groupBox_5)
        self.repeat_num_cbx.setObjectName("repeat_num_cbx")
        self.horizontalLayout_10.addWidget(self.repeat_num_cbx)
        self.save_plot_data_btn = QtWidgets.QPushButton(self.groupBox_5)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/my_icons/images/icons/save.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.save_plot_data_btn.setIcon(icon8)
        self.save_plot_data_btn.setIconSize(QtCore.QSize(20, 25))
        self.save_plot_data_btn.setObjectName("save_plot_data_btn")
        self.horizontalLayout_10.addWidget(self.save_plot_data_btn)
        self.save_plot_btn = QtWidgets.QPushButton(self.groupBox_5)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap(":/my_icons/images/icons/picture.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.save_plot_btn.setIcon(icon9)
        self.save_plot_btn.setIconSize(QtCore.QSize(20, 25))
        self.save_plot_btn.setObjectName("save_plot_btn")
        self.horizontalLayout_10.addWidget(self.save_plot_btn)
        self.plot_btn = QtWidgets.QPushButton(self.groupBox_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plot_btn.sizePolicy().hasHeightForWidth())
        self.plot_btn.setSizePolicy(sizePolicy)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap(":/my_icons/images/icons/plot.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.plot_btn.setIcon(icon10)
        self.plot_btn.setIconSize(QtCore.QSize(20, 25))
        self.plot_btn.setObjectName("plot_btn")
        self.horizontalLayout_10.addWidget(self.plot_btn)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem1)
        self.groupBox_10 = QtWidgets.QGroupBox(self.groupBox_5)
        self.groupBox_10.setObjectName("groupBox_10")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.groupBox_10)
        self.horizontalLayout_4.setContentsMargins(-1, 0, -1, 5)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_13 = QtWidgets.QLabel(self.groupBox_10)
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_4.addWidget(self.label_13)
        self.repeat_count_num = QtWidgets.QSpinBox(self.groupBox_10)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.repeat_count_num.sizePolicy().hasHeightForWidth())
        self.repeat_count_num.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.repeat_count_num.setFont(font)
        self.repeat_count_num.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.repeat_count_num.setReadOnly(True)
        self.repeat_count_num.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.repeat_count_num.setMaximum(1000)
        self.repeat_count_num.setProperty("value", 0)
        self.repeat_count_num.setObjectName("repeat_count_num")
        self.horizontalLayout_4.addWidget(self.repeat_count_num)
        self.horizontalLayout_10.addWidget(self.groupBox_10)
        self.verticalLayout_2.addLayout(self.horizontalLayout_10)
        self.line = QtWidgets.QFrame(self.groupBox_5)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_2.addWidget(self.line)
        self.graph_frame = QtWidgets.QFrame(self.groupBox_5)
        self.graph_frame.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.graph_frame.setFrameShape(QtWidgets.QFrame.Panel)
        self.graph_frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.graph_frame.setLineWidth(3)
        self.graph_frame.setMidLineWidth(3)
        self.graph_frame.setObjectName("graph_frame")
        self.verticalLayout_2.addWidget(self.graph_frame)
        self.verticalLayout_2.setStretch(1, 1)
        self.verticalLayout_2.setStretch(3, 10)
        self.verticalLayout_3.addWidget(self.groupBox_5)
        self.groupBox_11 = QtWidgets.QGroupBox(self.groupBox_9)
        self.groupBox_11.setObjectName("groupBox_11")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.groupBox_11)
        self.gridLayout_7.setContentsMargins(5, 5, 5, 5)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.data_processing_scroll = QtWidgets.QScrollArea(self.groupBox_11)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.data_processing_scroll.sizePolicy().hasHeightForWidth())
        self.data_processing_scroll.setSizePolicy(sizePolicy)
        self.data_processing_scroll.setWidgetResizable(True)
        self.data_processing_scroll.setObjectName("data_processing_scroll")
        self.scrollAreaWidgetContents_3 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_3.setGeometry(QtCore.QRect(0, 0, 830, 96))
        self.scrollAreaWidgetContents_3.setObjectName("scrollAreaWidgetContents_3")
        self.verticalLayout_24 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents_3)
        self.verticalLayout_24.setObjectName("verticalLayout_24")
        self.data_processing_msg = QtWidgets.QLabel(self.scrollAreaWidgetContents_3)
        self.data_processing_msg.setText("")
        self.data_processing_msg.setObjectName("data_processing_msg")
        self.verticalLayout_24.addWidget(self.data_processing_msg)
        self.data_processing_scroll.setWidget(self.scrollAreaWidgetContents_3)
        self.gridLayout_7.addWidget(self.data_processing_scroll, 0, 0, 1, 1)
        self.verticalLayout_3.addWidget(self.groupBox_11)
        self.verticalLayout_3.setStretch(0, 5)
        self.verticalLayout_3.setStretch(1, 1)
        self.horizontalLayout_5.addWidget(self.groupBox_9)
        self.horizontalLayout_5.setStretch(0, 2)
        self.horizontalLayout_5.setStretch(1, 3)
        self.verticalLayout.addWidget(self.frame_2)
        self.verticalLayout_5.addWidget(self.mainframe)
        self.verticalLayout_19.addWidget(self.mainwidget)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "ASG-CW-ODMR Automation"))
        self.label_6.setText(_translate("Form", "----Version 1.0"))
        self.groupBox_3.setTitle(_translate("Form", "RF Control Panel"))
        self.label_4.setText(_translate("Form", "RF Generator:"))
        self.rf_connect_btn.setText(_translate("Form", "Connect"))
        self.groupBox_13.setTitle(_translate("Form", "CW-Mode"))
        self.label_17.setText(_translate("Form", "Power(dBm)"))
        self.label_18.setText(_translate("Form", "Frequency(MHz)"))
        self.rf_ply_stp_btn.setText(_translate("Form", "RF On"))
        self.groupBox_14.setTitle(_translate("Form", "List Mode"))
        self.label_3.setText(_translate("Form", "DwellTime(ms):"))
        self.init_list_btn.setText(_translate("Form", "Init List"))
        self.label_14.setText(_translate("Form", "StopFreq(MHz):"))
        self.label_15.setText(_translate("Form", "StartFreq(MHz):"))
        self.label_16.setText(_translate("Form", "StepFreq(MHz):"))
        self.list_ply_stp_btn.setText(_translate("Form", "List On"))
        self.label_19.setText(_translate("Form", "Power(dBm)"))
        self.groupBox_6.setTitle(_translate("Form", "DSG836 Setup Info"))
        self.groupBox.setTitle(_translate("Form", "ODMR Control Panel"))
        self.groupBox_8.setTitle(_translate("Form", "Pulse Sequence"))
        self.label_7.setText(_translate("Form", "MWTime(ms):"))
        self.label_10.setText(_translate("Form", "SamplePerFreq:"))
        self.label_9.setText(_translate("Form", "Repeat:"))
        self.set_pulser_count_btn.setText(_translate("Form", "Set"))
        self.groupBox_4.setTitle(_translate("Form", "Play"))
        self.odmr_start_btn.setText(_translate("Form", "Start"))
        self.odmr_stop_btn.setText(_translate("Form", "Stop"))
        self.groupBox_7.setTitle(_translate("Form", "ODMR Setup Info"))
        self.groupBox_9.setTitle(_translate("Form", "Data Processing"))
        self.groupBox_5.setTitle(_translate("Form", "DAQ Plot"))
        self.save_plot_data_btn.setText(_translate("Form", "Save Plot Data"))
        self.save_plot_btn.setText(_translate("Form", "Save Plot"))
        self.plot_btn.setText(_translate("Form", "PushButton"))
        self.groupBox_10.setTitle(_translate("Form", "RepeatCount"))
        self.label_13.setText(_translate("Form", "Repeat Count:"))
        self.groupBox_11.setTitle(_translate("Form", "Data Processing Info"))
import resources_rc
