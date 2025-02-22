from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # 设置主窗体
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1080)
        # 主窗口不可调整大小
        MainWindow.setFixedSize(MainWindow.width(), MainWindow.height())
        # 设置中心窗体
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # groupBox_AftShow: 放置处理后图像的groupBox容器
        self.groupBox_AftShow = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_AftShow.setGeometry(QtCore.QRect(970, 10, 940, 720))
        self.groupBox_AftShow.setObjectName("groupBox_AftShow")
        # label_AftPicShow: 放置处理后图像的label，label里面放置图片
        self.label_AftPicShow = QtWidgets.QLabel(self.groupBox_AftShow)
        self.label_AftPicShow.setGeometry(QtCore.QRect(10, 20, 920, 690))
        self.label_AftPicShow.setText("")
        self.label_AftPicShow.setObjectName("label_AftPicShow")
        # groupBox_PreShow: 放置处理前图像的groupBox容器
        self.groupBox_PreShow = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_PreShow.setGeometry(QtCore.QRect(10, 10, 940, 720))
        self.groupBox_PreShow.setObjectName("groupBox_PreShow")
        # label_PrePicShow: 放置处理前图像的label，label里面放置图片
        self.label_PrePicShow = QtWidgets.QLabel(self.groupBox_PreShow)
        self.label_PrePicShow.setGeometry(QtCore.QRect(10, 20, 920, 690))
        self.label_PrePicShow.setText("")
        self.label_PrePicShow.setScaledContents(True)
        self.label_PrePicShow.setObjectName("label_PrePicShow")
        # groupBox: 放置色彩处理方式的groupBox容器
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        # 调整groupBox的大小以容纳更多控件
        self.groupBox.setGeometry(QtCore.QRect(10, 740, 500, 330))
        self.groupBox.setObjectName("groupBox")
        # label_datatype_2: 选择色彩处理方式
        self.label_datatype_2 = QtWidgets.QLabel(self.groupBox)
        self.label_datatype_2.setGeometry(QtCore.QRect(10, 20, 100, 30))
        self.label_datatype_2.setObjectName("label_datatype_2")
        # comboBox_ColorDeal: 选择色彩处理方式的下拉框
        self.comboBox_ColorDeal = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_ColorDeal.setGeometry(QtCore.QRect(120, 20, 150, 30))
        self.comboBox_ColorDeal.setObjectName("comboBox_ColorDeal")
        self.comboBox_ColorDeal.addItem("选择色彩处理方式")
        self.comboBox_ColorDeal.addItem("原始图片")
        self.comboBox_ColorDeal.addItem("反色处理")
        self.comboBox_ColorDeal.addItem("灰值化")
        self.comboBox_ColorDeal.addItem("Lab颜色模型")
        self.comboBox_ColorDeal.addItem("YCrCb颜色模型")
        self.comboBox_ColorDeal.addItem("HSI颜色模型")
        # label_process_method: 选择处理方法
        self.label_process_method = QtWidgets.QLabel(self.groupBox)
        self.label_process_method.setGeometry(QtCore.QRect(10, 60, 100, 30))
        self.label_process_method.setObjectName("label_process_method")
        # comboBox_ProcessMethod: 选择处理方法的下拉框
        self.comboBox_ProcessMethod = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_ProcessMethod.setGeometry(QtCore.QRect(120, 60, 150, 30))
        self.comboBox_ProcessMethod.setObjectName("comboBox_ProcessMethod")
        self.comboBox_ProcessMethod.addItem("OpenCV")
        self.comboBox_ProcessMethod.addItem("手写函数")
        # label_slider: 滚动条标签
        self.label_slider = QtWidgets.QLabel(self.groupBox)
        self.label_slider.setGeometry(QtCore.QRect(10, 100, 100, 30))
        self.label_slider.setObjectName("label_slider")
        # slider_SamplingPrecision: 滚动条
        self.slider_SamplingPrecision = QtWidgets.QSlider(self.groupBox)
        self.slider_SamplingPrecision.setGeometry(QtCore.QRect(120, 100, 150, 30))
        self.slider_SamplingPrecision.setOrientation(QtCore.Qt.Horizontal)
        self.slider_SamplingPrecision.setObjectName("slider_SamplingPrecision")
        self.slider_SamplingPrecision.setRange(1, 100)  # 设置范围
        self.slider_SamplingPrecision.setValue(1)  # 设置初始值为最小值
        # label_SamplingValue: 显示采样间隔数的标签
        self.label_SamplingValue = QtWidgets.QLabel(self.groupBox)
        self.label_SamplingValue.setGeometry(QtCore.QRect(280, 100, 50, 30))
        self.label_SamplingValue.setObjectName("label_SamplingValue")

        # label_quantization: 量化等级标签
        self.label_quantization = QtWidgets.QLabel(self.groupBox)
        self.label_quantization.setGeometry(QtCore.QRect(10, 140, 100, 30))
        self.label_quantization.setObjectName("label_quantization")
        # slider_QuantizationLevel: 量化等级滚动条
        self.slider_QuantizationLevel = QtWidgets.QSlider(self.groupBox)
        self.slider_QuantizationLevel.setGeometry(QtCore.QRect(120, 140, 150, 30))
        self.slider_QuantizationLevel.setOrientation(QtCore.Qt.Horizontal)
        self.slider_QuantizationLevel.setObjectName("slider_QuantizationLevel")
        self.slider_QuantizationLevel.setRange(1, 256)  # 设置范围
        self.slider_QuantizationLevel.setValue(256)  # 设置初始值为最大值
        # label_QuantizationValue: 显示量化等级数的标签
        self.label_QuantizationValue = QtWidgets.QLabel(self.groupBox)
        self.label_QuantizationValue.setGeometry(QtCore.QRect(280, 140, 50, 30))
        self.label_QuantizationValue.setObjectName("label_QuantizationValue")

        # label_non_uniform_sampling: 非均匀采样标签
        self.label_non_uniform_sampling = QtWidgets.QLabel(self.groupBox)
        self.label_non_uniform_sampling.setGeometry(QtCore.QRect(10, 180, 100, 30))
        self.label_non_uniform_sampling.setObjectName("label_non_uniform_sampling")

        # comboBox_NonUniformSampling: 选择非均匀采样方式的下拉框
        self.comboBox_NonUniformSampling = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_NonUniformSampling.setGeometry(QtCore.QRect(120, 180, 150, 30))
        self.comboBox_NonUniformSampling.setObjectName("comboBox_NonUniformSampling")
        self.comboBox_NonUniformSampling.addItem("关闭非均匀采样")
        self.comboBox_NonUniformSampling.addItem("基于边缘的非均匀采样")

        # label_non_uniform_quantization: 非均匀量化标签
        self.label_non_uniform_quantization = QtWidgets.QLabel(self.groupBox)
        self.label_non_uniform_quantization.setGeometry(QtCore.QRect(10, 220, 100, 30))
        self.label_non_uniform_quantization.setObjectName("label_non_uniform_quantization")

        # comboBox_NonUniformQuantization: 选择非均匀量化方式的下拉框
        self.comboBox_NonUniformQuantization = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_NonUniformQuantization.setGeometry(QtCore.QRect(120, 220, 150, 30))
        self.comboBox_NonUniformQuantization.setObjectName("comboBox_NonUniformQuantization")
        self.comboBox_NonUniformQuantization.addItem("关闭非均匀量化")
        self.comboBox_NonUniformQuantization.addItem("非均匀量化1")

        # label_datatype: 选择数据类型
        self.label_datatype = QtWidgets.QLabel(self.centralwidget)
        self.label_datatype.setGeometry(QtCore.QRect(600, 800, 150, 30))
        self.label_datatype.setObjectName("label_datatype")
        # pushButton_VideoDisplay: 检测按钮 开始播放视频
        self.pushButton_VideoDisplay = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_VideoDisplay.setGeometry(QtCore.QRect(900, 800, 100, 30))
        self.pushButton_VideoDisplay.setObjectName("pushButton_VideoDisplay")
        # comboBox_SelectData: 选择数据类型的
        self.comboBox_SelectData = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_SelectData.setGeometry(QtCore.QRect(760, 800, 120, 30))
        self.comboBox_SelectData.setObjectName("comboBox_SelectData")
        self.comboBox_SelectData.addItem("下拉选择")
        self.comboBox_SelectData.addItem("读入图像")
        self.comboBox_SelectData.addItem("读入本地视频")
        
        # pushButton_SaveImage: 另存为按钮
        self.pushButton_SaveImage = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_SaveImage.setGeometry(QtCore.QRect(1050, 800, 100, 30))
        self.pushButton_SaveImage.setObjectName("pushButton_SaveImage")
        self.pushButton_SaveImage.setText("另存为")
        self.pushButton_SaveImage.setVisible(False)  # 初始时隐藏

        # label_DebugInfo: 显示调试信息的标签
        self.label_DebugInfo = QtWidgets.QLabel(self.centralwidget)
        self.label_DebugInfo.setGeometry(QtCore.QRect(1000, 960, 1880, 30))  # 调整位置和大小
        self.label_DebugInfo.setObjectName("label_DebugInfo")

        # 设置主窗体的中心窗体
        MainWindow.setCentralWidget(self.centralwidget)
        # 设置菜单栏
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1920, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        # 设置状态栏
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        # 设置窗体标题
        self.retranslateUi(MainWindow)
        # 设置信号与槽函数
        # 选择数据类型与选择读入数据的槽函数
        self.comboBox_SelectData.currentIndexChanged['int'].connect(MainWindow.onbuttonclick_selectDataType) # type: ignore
        # 检测按钮与检测函数的槽函数
        self.pushButton_VideoDisplay.clicked.connect(MainWindow.onbuttonclick_videodisplay) # type: ignore
        # 选择色彩处理方式与选择色彩处理方式的槽函数
        self.comboBox_ColorDeal.currentIndexChanged['int'].connect(MainWindow.oncombox_selectColorType) # type: ignore
        # 选择处理方法是用手写函数还是OpenCV方法的槽函数
        self.comboBox_ProcessMethod.currentIndexChanged['int'].connect(MainWindow.deal_method) 
        
        # 滚动条与更新采样间隔的槽函数
        self.slider_SamplingPrecision.valueChanged['int'].connect(MainWindow.update_sampling_interval) # type: ignore
        # 滚动条与更新采样间隔标签的槽函数
        self.slider_SamplingPrecision.valueChanged['int'].connect(MainWindow.update_sampling_value_label) # type: ignore
        
        # 滚动条与更新量化等级的槽函数
        self.slider_QuantizationLevel.valueChanged['int'].connect(MainWindow.update_quantization_level)
        # 量化等级滚动条与更新量化等级标签的槽函数
        self.slider_QuantizationLevel.valueChanged['int'].connect(MainWindow.update_quantization_value_label) # type: ignore
        
        # 另存为按钮与保存图片的槽函数
        self.pushButton_SaveImage.clicked.connect(MainWindow.onbuttonclick_saveimage) # type: ignore

        # 非均匀采样方式与更新非均匀采样方式的槽函数
        self.comboBox_NonUniformSampling.currentIndexChanged['int'].connect(MainWindow.update_non_uniform_sampling)

        # 非均匀量化方式与更新非均匀量化方式的槽函数
        self.comboBox_NonUniformQuantization.currentIndexChanged['int'].connect(MainWindow.update_non_uniform_quantization)

        # 设置窗体标题
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        MainWindow.update_sampling_value_label(self.slider_SamplingPrecision.value())
        MainWindow.update_quantization_value_label(self.slider_QuantizationLevel.value())

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox_AftShow.setTitle(_translate("MainWindow", "测试实例"))
        self.groupBox_PreShow.setTitle(_translate("MainWindow", "输入实例"))
        self.groupBox.setTitle(_translate("MainWindow", "第一章"))
        self.label_datatype_2.setText(_translate("MainWindow", "色彩处理："))
        self.comboBox_ColorDeal.setItemText(0, _translate("MainWindow", "选择处理方式"))
        self.comboBox_ColorDeal.setItemText(1, _translate("MainWindow", "原始图片"))
        self.comboBox_ColorDeal.setItemText(2, _translate("MainWindow", "反色处理"))
        self.comboBox_ColorDeal.setItemText(3, _translate("MainWindow", "灰值化"))
        self.comboBox_ColorDeal.setItemText(4, _translate("MainWindow", "Lab颜色模型"))
        self.comboBox_ColorDeal.setItemText(5, _translate("MainWindow", "YCrCb颜色模型"))
        self.comboBox_ColorDeal.setItemText(6, _translate("MainWindow", "HSI颜色模型"))
        self.label_process_method.setText(_translate("MainWindow", "处理方法："))
        self.comboBox_ProcessMethod.setItemText(0, _translate("MainWindow", "OpenCV"))
        self.comboBox_ProcessMethod.setItemText(1, _translate("MainWindow", "手写函数"))
        self.label_slider.setText(_translate("MainWindow", "采样间隔："))
        self.label_SamplingValue.setText(_translate("MainWindow", "1"))  # 初始值为1
        self.label_quantization.setText(_translate("MainWindow", "量化等级："))
        self.label_QuantizationValue.setText(_translate("MainWindow", "256"))
        self.label_datatype.setText(_translate("MainWindow", "选择数据类型："))
        self.pushButton_VideoDisplay.setText(_translate("MainWindow", "检测"))
        self.comboBox_SelectData.setItemText(1, _translate("MainWindow", "读入图像"))
        self.comboBox_SelectData.setItemText(2, _translate("MainWindow", "读入本地视频"))

        self.label_DebugInfo.setText(_translate("MainWindow", "调试信息："))

        self.label_non_uniform_sampling.setText(_translate("MainWindow", "非均匀采样："))
        self.comboBox_NonUniformSampling.setItemText(0, _translate("MainWindow", "关闭非均匀采样"))
        self.comboBox_NonUniformSampling.setItemText(1, _translate("MainWindow", "基于边缘的非均匀采样"))

        self.label_non_uniform_quantization.setText(_translate("MainWindow", "非均匀量化："))
        self.comboBox_NonUniformQuantization.setItemText(0, _translate("MainWindow", "关闭非均匀量化"))
        self.comboBox_NonUniformQuantization.setItemText(1, _translate("MainWindow", "非均匀量化1"))

    def update_sampling_value_label(self, value):
        """更新采样间隔标签"""
        self.label_SamplingValue.setText(str(value))
    
    def update_quantization_value_label(self, value):
        """更新量化等级标签"""
        self.label_QuantizationValue.setText(str(value))