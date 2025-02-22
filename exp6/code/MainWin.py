import os
import cv2 as cv
import numpy as np
import sys
from MainDetect_UI import *

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow
# from Experiment_1 import *
# from Experiment_2 import *
from Experiment5.Experiment_3 import *
from Experiment5.Experiment_4 import *
from Experiment5.Experiment_5 import *
from Experiment5.Experiment_6 import *
from Experiment5.CodeRecognition_UI import Ui_Dialog as UI_CodeRecognition
from Experiment5.CodeRecognition_UI import CodeRecognitionDialog

class Main(QMainWindow, Ui_MainWindow):
    """重写主窗体类"""
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)
        self.setupUi(self) # 初始化窗体显示
        self.timer = QTimer(self) # 初始化定时器
        # 设置在label中自适应显示图片
        self.label_PrePicShow.setScaledContents(True)
        self.label_PrePicShow.setStyleSheet("QLabel{background-color:rgb(0,0,0);}")  # 初始黑化图像显示区域
        self.label_AftPicShow.setScaledContents(True)
        self.label_AftPicShow.setStyleSheet("QLabel{background-color:rgb(0,0,0);}")
        self.img = None
        # 添加角点检测参数
        self.harris_k = 0.04
        self.susan_t = 27
        # 添加细胞计数参数
        self.circularity_threshold = 0.7

    def onbuttonclick_selectDataType(self, index):
        """选择输入数据类型(图像,视频)"""
        if index == 1:
            filename, _ = QFileDialog.getOpenFileName(self, "选择图像", os.getcwd(), "Images (*.bmp *.jpg *.png);;All (*)")
            self.img = cv.imread(filename, 0) # 参数-1，原通道，1，强制三通道(彩图)，0强制一通道(灰度图)  强制读取灰度图
            self.img_show(self.label_PrePicShow, self.img)
        elif index == 2:
            filename, _ = QFileDialog.getOpenFileName(self, "选择视频", os.getcwd(), "Videos (*.avi *.mp4);;All (*)")
            self.capture = cv.VideoCapture(filename)
            self.fps = self.capture.get(cv.CAP_PROP_FPS)  # 获得视频帧率
            self.timer.timeout.connect(self.slot_video_display)
            flag, self.img = self.capture.read()  # 显示视频第一帧
            if flag:
                self.img_show(self.label_PrePicShow, self.img)

    def onbuttonclick_videodisplay(self):
        """显示视频控制函数, 用于连接定时器超时触发槽函数"""
        if self.pushButton_VideoDisplay.text() == "检测":
            self.timer.start(1000 / self.fps)
            self.pushButton_VideoDisplay.setText("暂停")
        else:
            self.timer.stop()
            self.pushButton_VideoDisplay.setText("检测")

    def slot_video_display(self):
        """定时器超时触发槽函, 在label上显示每帧视频, 防止卡顿"""
        flag, self.img = self.capture.read()
        if flag:
            self.img_show(self.label_PrePicShow, self.img)
        else:
            self.capture.release()
            self.timer.stop()

    def oncombox_selectColorType(self, index):
        """选择图像色彩处理方式"""
        imgDeal = color_deal(self.img, index)
        self.img_show(self.label_AftPicShow, imgDeal)

    def onslide_imgSample(self):
        """滚动条选择采样间隔"""
        iv = self.slider_ImgSample.value()
        self.label_sample.setText(str(iv))
        imgSample = img_sample(self.img, iv)
        self.img_show(self.label_AftPicShow, imgSample)

    def onslide_imgQuanty(self):
        """滚动条选择量化范围"""
        if self.img.shape[-1] == 3:
            msg_box = QMessageBox(QMessageBox.Warning, '警告', '请选择灰度图像！')
            msg_box.exec_()
            return
        q_Size = self.slider_ImgQuanty.value()
        self.label_quanty.setText('0-' + str(q_Size))
        imgQuanty = img_quanty(self.img, q_Size)
        self.img_show(self.label_AftPicShow, imgQuanty)

    def onslide_imgZoom(self):
        """控制滚动条实现图像缩放"""
        zm = self.slider_ImgZoom.value()
        self.label_zoom.setText(str(zm+1) + 'tms') if zm > -1 else self.label_zoom.setText('1/' + str(-zm+1) + 'tms')
        zm = zm + 1 if zm > -1 else 1.0 / (-zm + 1)
        imgZoom = img_zoom(self.img, zm)
        self.img_show(self.label_AftPicShow, imgZoom)
    def onslide_imgTranslation(self):
        """控制滚动条实现图像左右平移"""
        trans = self.slider_ImgTranslation.value()
        self.label_tanslation.setText(str(trans) + 'pix')
        imgTrans = img_translation(self.img, trans)
        self.img_show(self.label_AftPicShow, imgTrans)

    def onslide_imgRotation(self):
        """控制滚动条进行图像旋转"""
        rot = self.slider_ImgRotate.value()
        self.label_rotate.setText(str(rot) + '°')
        imgRotate = img_rotation(self.img, rot)
        self.img_show(self.label_AftPicShow, imgRotate)

    def onbuttonclick_imgMirror(self):
        """点击按钮做图像镜面"""
        imgMirror = img_imgMirror(self.img)
        self.img_show(self.label_AftPicShow, imgMirror)

    def onbuttonclick_cardCrrection(self):
        """点击按钮做名片矫正"""
        imgCard = card_correction(self.img)
        self.img_show(self.label_AftPicShow, imgCard)

    def oncombox_selectGrayDeal(self, index):
        """选择灰度增强处理方式"""
        imgDeal = gray_deal(self.img, index)
        self.img_show(self.label_AftPicShow, imgDeal)

    def onbutttonclick_histEqual(self):
        """点击做图像增强"""
        imgHist = hist_equalization(self.img, self.checkBox_hist.isChecked())
        self.img_show(self.label_AftPicShow, imgHist)

    def oncombox_selectConvType(self, index):
        """选择图像平滑方式"""
        imgDeal = gray_smooth(self.img, index)
        self.img_show(self.label_AftPicShow, imgDeal)

    def oncombox_selectEdgeDetecType(self, index):
        """选择图像平滑方式"""
        imgDeal = edge_detect(self.img, index)
        self.img_show(self.label_AftPicShow, imgDeal)

    def oncombox_selectHoughDetecType(self, index):
        """选择图像平滑方式"""
        if self.img.shape[-1] == 3:
            pass
        if index == 1:
            img, hough_img = line_detect(self.img, 2)
            self.img_show(self.label_PrePicShow, img)
            self.img_show(self.label_AftPicShow, hough_img)
        elif index == 2:
            new_img = circle_detect(self.img)
            self.img_show(self.label_AftPicShow, new_img)

    def onbuttonclick_Otsu(self):
        """点击做图像增强"""
        imgOtsu = otsu(self.img, self.checkBox_hist_otsu.isChecked())
        self.img_show(self.label_AftPicShow, imgOtsu)

    def oncombox_selectMorphyBinary(self, index):
        """选择图像二值形态学运算方式"""
        imgDeal = morphy_binary(self.img, index)
        self.img_show(self.label_AftPicShow, imgDeal)

    def oncombox_selectMorphyGray(self, index):
        """选择图像灰值形态学运算方式"""
        imgDeal = morphy_gray(self.img, index)
        self.img_show(self.label_AftPicShow, imgDeal)

    def onbuttonclick_MorphyThin(self):
        """点击做图像快速形态学细化算法(骨架提取)"""
        deal_type = 0 if self.radioButton_Skeleton.isChecked() else 1
        imgThin = fast_thin(self.img, deal_type)
        self.img_show(self.label_AftPicShow, imgThin)

    def onslide_harris_k(self):
        """控制Harris检测的k值"""
        self.harris_k = self.slider_harris_k.value() * 0.01
        if self.img is not None:
            corner_img = harris_corner_detect(self.img, self.harris_k)
            self.img_show(self.label_AftPicShow, corner_img)
            
    def onslide_susan_t(self):
        """控制SUSAN检测的阈值"""
        self.susan_t = self.slider_susan_t.value()
        if self.img is not None:
            corner_img = susan_corner_detect(self.img, self.susan_t)
            self.img_show(self.label_AftPicShow, corner_img)
            
    def onbuttonclick_HarrisCorner(self):
        """点击进行Harris角点检测"""
        if self.img is None:
            msg_box = QMessageBox(QMessageBox.Warning, '警告', '请先选择图像！')
            msg_box.exec_()
            return
        
        corner_img = harris_corner_detect(self.img, self.harris_k)
        self.img_show(self.label_AftPicShow, corner_img)
        
    def onbuttonclick_SusanCorner(self):
        """点击进行SUSAN角点检测"""
        if self.img is None:
            msg_box = QMessageBox(QMessageBox.Warning, '警告', '请先选择图像！')
            msg_box.exec_()
            return
        
        corner_img = susan_corner_detect(self.img, self.susan_t)
        self.img_show(self.label_AftPicShow, corner_img)

    def onslide_circularity(self):
        """控制细胞计数的圆形度阈值"""
        self.circularity_threshold = self.slider_circularity.value() * 0.1
        if self.img is not None:
            marked_img, count, binary = count_cells(self.img, self.circularity_threshold)
            self.img_show(self.label_AftPicShow, marked_img)
            cv.imshow("Binary Image", binary)

    def onbuttonclick_CellCount(self):
        """点击进行细胞计数"""
        if self.img is None:
            msg_box = QMessageBox(QMessageBox.Warning, '警告', '请先选择图像！')
            msg_box.exec_()
            return
        
        # 进行细胞计数
        marked_img, count, binary = count_cells(self.img, self.circularity_threshold)
        
        # 显示二值化结果和标记结果
        # 创建一个新窗口显示二值化图像
        cv.imshow("Binary Image", binary)
        
        self.img_show(self.label_AftPicShow, marked_img)
        
        # 显示计数结果
        msg_box = QMessageBox(QMessageBox.Information, '计数结果', 
                             f'检测到{count}个细胞')
        msg_box.exec_()

    def img_show(self, label, img):
        """图片在对应label中显示"""
        if img.shape[-1] == 3:
            qimage = QImage(img.data.tobytes(), img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format_RGB888).rgbSwapped()
        else:
            qimage = QImage(img.data.tobytes(), img.shape[1], img.shape[0], img.shape[1], QImage.Format_Indexed8)
        label.setPixmap(QPixmap.fromImage(qimage))

    def onbuttonclick_CodeRecognition(self):
        """点击打开验证码识别界面"""
        dialog = CodeRecognitionDialog(self)
        dialog.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Main()
    win.show()
    print(app.exec_())
    sys.exit(app.exec_())
