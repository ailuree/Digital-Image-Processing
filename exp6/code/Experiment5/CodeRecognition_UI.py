from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
import cv2 as cv
import os
from Experiment5.Experiment_6_e import load_captcha, denoise_captcha, normalize_captcha

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1200, 800)
        
        # 创建控制面板区域（左侧）
        self.groupBox_Control = QtWidgets.QGroupBox(Dialog)
        self.groupBox_Control.setGeometry(QtCore.QRect(10, 10, 280, 600))
        self.groupBox_Control.setObjectName("groupBox_Control")
        
        # 添加控制按钮
        self.pushButton_LoadImage = QtWidgets.QPushButton(self.groupBox_Control)
        self.pushButton_LoadImage.setGeometry(QtCore.QRect(20, 30, 240, 40))
        self.pushButton_LoadImage.setObjectName("pushButton_LoadImage")
        
        self.pushButton_Denoise = QtWidgets.QPushButton(self.groupBox_Control)
        self.pushButton_Denoise.setGeometry(QtCore.QRect(20, 80, 240, 40))
        self.pushButton_Denoise.setObjectName("pushButton_Denoise")
        
        self.pushButton_Locate = QtWidgets.QPushButton(self.groupBox_Control)
        self.pushButton_Locate.setGeometry(QtCore.QRect(20, 130, 240, 40))
        self.pushButton_Locate.setObjectName("pushButton_Locate")
        
        self.pushButton_Normalize = QtWidgets.QPushButton(self.groupBox_Control)
        self.pushButton_Normalize.setGeometry(QtCore.QRect(20, 180, 240, 40))
        self.pushButton_Normalize.setObjectName("pushButton_Normalize")
        
        self.pushButton_Recognize = QtWidgets.QPushButton(self.groupBox_Control)
        self.pushButton_Recognize.setGeometry(QtCore.QRect(20, 230, 240, 40))
        self.pushButton_Recognize.setObjectName("pushButton_Recognize")
        
        # 添加识别结果显示区域
        self.groupBox_Result = QtWidgets.QGroupBox(self.groupBox_Control)
        self.groupBox_Result.setGeometry(QtCore.QRect(20, 290, 240, 120))
        self.groupBox_Result.setObjectName("groupBox_Result")
        
        self.label_Result = QtWidgets.QLabel(self.groupBox_Result)
        self.label_Result.setGeometry(QtCore.QRect(20, 30, 200, 70))
        self.label_Result.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(24)
        self.label_Result.setFont(font)
        self.label_Result.setObjectName("label_Result")
        
        # 创建图像显示区域（右侧）
        self.groupBox_Display = QtWidgets.QGroupBox(Dialog)
        self.groupBox_Display.setGeometry(QtCore.QRect(300, 10, 880, 600))
        self.groupBox_Display.setObjectName("groupBox_Display")
        
        # 原始图像显示区
        self.label_OriginImage = QtWidgets.QLabel(self.groupBox_Display)
        self.label_OriginImage.setGeometry(QtCore.QRect(20, 30, 840, 270))
        self.label_OriginImage.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.label_OriginImage.setAlignment(QtCore.Qt.AlignCenter)
        self.label_OriginImage.setObjectName("label_OriginImage")
        
        # 处理结果显示区
        self.label_ProcessImage = QtWidgets.QLabel(self.groupBox_Display)
        self.label_ProcessImage.setGeometry(QtCore.QRect(20, 310, 840, 270))
        self.label_ProcessImage.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.label_ProcessImage.setAlignment(QtCore.Qt.AlignCenter)
        self.label_ProcessImage.setObjectName("label_ProcessImage")
        
        # 创建使用说明区域（底部）
        self.groupBox_Help = QtWidgets.QGroupBox(Dialog)
        self.groupBox_Help.setGeometry(QtCore.QRect(10, 620, 1170, 170))
        self.groupBox_Help.setObjectName("groupBox_Help")
        
        # 添加使用说明文本
        self.textBrowser_Help = QtWidgets.QTextBrowser(self.groupBox_Help)
        self.textBrowser_Help.setGeometry(QtCore.QRect(20, 30, 1000, 130))
        self.textBrowser_Help.setObjectName("textBrowser_Help")
        
        # 添加截屏保存按钮
        self.pushButton_Screenshot = QtWidgets.QPushButton(self.groupBox_Help)
        self.pushButton_Screenshot.setGeometry(QtCore.QRect(1030, 30, 120, 130))
        self.pushButton_Screenshot.setObjectName("pushButton_Screenshot")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "验证码识别"))
        self.groupBox_Control.setTitle(_translate("Dialog", "控制面板"))
        self.pushButton_LoadImage.setText(_translate("Dialog", "载入验证码图像"))
        self.pushButton_Denoise.setText(_translate("Dialog", "验证码图像去噪"))
        self.pushButton_Locate.setText(_translate("Dialog", "验证码数字定位"))
        self.pushButton_Normalize.setText(_translate("Dialog", "验证码归一化"))
        self.pushButton_Recognize.setText(_translate("Dialog", "验证码识别"))
        self.groupBox_Result.setTitle(_translate("Dialog", "识别结果"))
        self.label_Result.setText(_translate("Dialog", "待识别"))
        self.groupBox_Display.setTitle(_translate("Dialog", "图像显示"))
        self.label_OriginImage.setText(_translate("Dialog", "验证码原图"))
        self.label_ProcessImage.setText(_translate("Dialog", "处理结果"))
        self.groupBox_Help.setTitle(_translate("Dialog", "使用说明"))
        self.pushButton_Screenshot.setText(_translate("Dialog", "截屏保存"))
        
        # 设置使用说明文本
        help_text = """使用说明：
1. 点击"载入验证码图像"按钮选择要识别的验证码图片
2. 点击"验证码图像去噪"按钮对图像进行预处理
3. 点击"验证码数字定位"按钮定位图像中的数字位置
4. 点击"验证码归一化"按钮对定位到的数字进行归一化处理
5. 点击"验证码识别"按钮进行最终的数字识别
6. 识别结果将显示在左侧的识别结果区域
7. 点击"截屏保存"按钮可以保存当前的处理结果"""
        self.textBrowser_Help.setText(_translate("Dialog", help_text)) 

class CodeRecognitionDialog(QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super(CodeRecognitionDialog, self).__init__(parent)
        self.setupUi(self)
        self.img = None  # 存储原始图像
        self.processed_img = None  # 存储处理后的图像
        
        # 绑定按钮点击事件
        self.pushButton_LoadImage.clicked.connect(self.onButtonLoadImage)
        self.pushButton_Denoise.clicked.connect(self.onButtonDenoise)
        self.pushButton_Normalize.clicked.connect(self.onButtonNormalize)
        self.pushButton_Screenshot.clicked.connect(self.onButtonScreenshot)
        
        # 设置图像显示区域的缩放属性
        self.label_OriginImage.setScaledContents(True)
        self.label_ProcessImage.setScaledContents(True)
        
    def onButtonLoadImage(self):
        """载入验证码图像"""
        filename, _ = QFileDialog.getOpenFileName(self, "选择验证码图像", 
                                                os.getcwd(), 
                                                "Images (*.png *.jpg *.bmp);;All Files (*)")
        if filename:
            self.img = load_captcha(filename)
            if self.img is not None:
                self.showImage(self.label_OriginImage, self.img)
                self.processed_img = self.img.copy()  # 初始化处理图像
            else:
                QMessageBox.warning(self, "警告", "图像加载失败！")
                
    def onButtonDenoise(self):
        """验证码图像去噪"""
        if self.img is None:
            QMessageBox.warning(self, "警告", "请先载入图像！")
            return
            
        self.processed_img = denoise_captcha(self.img)
        if self.processed_img is not None:
            self.showImage(self.label_ProcessImage, self.processed_img)
        else:
            QMessageBox.warning(self, "警告", "图像去噪失败！")
            
    def onButtonNormalize(self):
        """验证码归一化"""
        if self.processed_img is None:
            QMessageBox.warning(self, "警告", "请先进行图像去噪！")
            return
            
        normalized_img, debug_img = normalize_captcha(self.processed_img)
        if normalized_img is not None and debug_img is not None:
            # 在处理结果区域显示归一化结果
            self.processed_img = normalized_img  # 保存归一化结果
            self.showImage(self.label_ProcessImage, normalized_img)
            
            # 在原图区域显示调试图像
            self.showImage(self.label_OriginImage, debug_img)
        else:
            QMessageBox.warning(self, "警告", "图像归一化失败！")
            
    def onButtonScreenshot(self):
        """截屏保存"""
        if self.processed_img is None:
            QMessageBox.warning(self, "警告", "没有可保存的处理结果！")
            return
            
        filename, _ = QFileDialog.getSaveFileName(self, "保存处理结果", 
                                                os.getcwd(), 
                                                "Images (*.png *.jpg *.bmp);;All Files (*)")
        if filename:
            cv.imwrite(filename, self.processed_img)
            QMessageBox.information(self, "提示", "保存成功！")
            
    def showImage(self, label, img):
        """在指定的label上显示图像"""
        if len(img.shape) == 3:
            height, width, channel = img.shape
            bytesPerLine = 3 * width
            qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        else:
            height, width = img.shape
            qImg = QImage(img.data, width, height, width, QImage.Format_Indexed8)
        
        label.setPixmap(QPixmap.fromImage(qImg)) 