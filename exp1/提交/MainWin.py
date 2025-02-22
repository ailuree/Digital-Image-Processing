import os
import cv2
import sys
import numpy as np
from MainDetect_UI import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow
from Experiment_1 import *
from abc import ABC, abstractmethod
from scipy.interpolate import griddata
from scipy.interpolate import interp2d

class ProcessingStrategy(ABC):
    @abstractmethod
    def apply(self, img):
        pass

# 均匀采样处理策略
class UniformSampling(ProcessingStrategy):
    def __init__(self, interval):
        self.interval = interval

    def apply(self, img):
        # 对图像均匀采样处理
        # 确保步长不超过图像尺寸
        interval = min(self.interval, img.shape[0], img.shape[1])
        # 采样处理
        sampled_img = img[::interval, ::interval]  # 具体原理就是按照interval间隔取值 取像素点
        # 放大采样后的图像，适应原始尺寸
        return cv2.resize(sampled_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
# 均匀量化处理策略
class UniformQuantization(ProcessingStrategy):
    def __init__(self, levels):
        self.levels = levels

    def apply(self, img):
        # 确保量化等级至少为2
        levels = max(2, self.levels)
        # 计算量化步长
        step = 256 // levels
        # 确保图像是uint8类型
        img = img.astype(np.uint8)
        # 如果是彩色图像，分别对每个通道进行量化
        if len(img.shape) == 3:
            quantized_img = np.zeros_like(img)
            for c in range(3):
                quantized_img[:, :, c] = (img[:, :, c] // step) * step
        else:
            # 对灰度图像进行量化
            quantized_img = (img // step) * step
        return quantized_img
# 动态采样的非均匀采样处理策略_opencv
class NonUniformSampling_opencv(ProcessingStrategy):
    def __init__(self, min_interval, max_interval):
        self.min_interval = min_interval
        self.max_interval = max_interval

    def apply(self, img):
        # 将图像转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

        # 计算图像梯度
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # 图像梯度归一化
        norm_gradient = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
        
        # 动态调整采样间隔
        adaptive_intervals = self.min_interval + (self.max_interval - self.min_interval) * (1 - norm_gradient)
        adaptive_intervals = np.clip(adaptive_intervals, self.min_interval, self.max_interval).astype(int)
        
        # 采样图像
        rows, cols = img.shape[:2]
        y, x = np.mgrid[0:rows:1, 0:cols:1]
        
        # 生成采样掩码
        x_mask = np.mod(x, adaptive_intervals) == 0
        y_mask = np.mod(y, adaptive_intervals) == 0
        mask = x_mask & y_mask
        
        # 采样图像值
        sampled_y, sampled_x = y[mask], x[mask]
        sampled_values = img[sampled_y, sampled_x]
        
        # 二维插值 - 使用 griddata 进行双线性插值
        grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
        sampled_img = griddata((sampled_x, sampled_y), sampled_values, (grid_x, grid_y), method='linear')
        
        # 处理插值结果中的NaN值
        sampled_img = np.nan_to_num(sampled_img, nan=np.nanmean(sampled_img))
        
        return sampled_img.astype(img.dtype)
        


# 动态采样的非均匀采样处理策略
class NonUniformSampling(ProcessingStrategy):
    """
    1. 在图像细节丰富的地方，减小采样间隔（增大采样频率），获得更多的图像信息。
    2. 在图像变化缓慢的地方，粗采样（采样间隔大）
    """
    def __init__(self, min_interval, max_interval):
        self.min_interval = min_interval
        self.max_interval = max_interval

    def apply(self, img):
        # 将图像转换为灰度图像
        if len(img.shape) == 3:
            gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img

        # 计算图像梯度
        gradient_x = np.zeros_like(gray)
        gradient_y = np.zeros_like(gray)
        gradient_x[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) / 2
        gradient_y[1:-1, :] = (gray[2:, :] - gray[:-2, :]) / 2
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # 图像梯度归一化
        norm_gradient = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min())
        
        # 动态调整采样间隔
        adaptive_intervals = self.min_interval + (self.max_interval - self.min_interval) * (1 - norm_gradient)
        adaptive_intervals = np.clip(adaptive_intervals, self.min_interval, self.max_interval).astype(int)
        
        # 采样图像
        rows, cols = img.shape[:2]
        y, x = np.mgrid[0:rows:1, 0:cols:1]
        
        # 生成采样掩码
        x_mask = np.mod(x, adaptive_intervals) == 0
        y_mask = np.mod(y, adaptive_intervals) == 0
        mask = x_mask & y_mask
        
        # 采样图像值
        sampled_y, sampled_x = y[mask], x[mask]
        sampled_values = img[sampled_y, sampled_x]
        
        # 二维插值 - 使用 griddata 进行双线性插值
        grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
        sampled_img = griddata((sampled_x, sampled_y), sampled_values, (grid_x, grid_y), method='linear')
        
        # 处理插值结果中的NaN值
        sampled_img = np.nan_to_num(sampled_img, nan=np.nanmean(sampled_img))
        
        return sampled_img.astype(img.dtype)

# 动态调整量化等级的非均匀量化处理策略_opencv
class NonUniformQuantization_opencv(ProcessingStrategy):
    """
    动态调整量化等级的非均匀量化处理策略 
    在图像灰度/颜色值变化比较剧烈的地方，量化级别可以减少
    在图像灰度/颜色值变化比较平缓的地方，量化级别可以增加

    1. 计算局部变化程度：使用图像梯度或拉普拉斯算子来计算图像的局部变化程度。
    2. 动态调整量化级别：根据局部变化程度动态调整量化级别。
    """
    def __init__(self, base_levels, max_levels):
        self.base_levels = base_levels
        self.max_levels = max_levels

    def apply(self, img):
        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        # 计算图像梯度
        gradient = cv2.Laplacian(gray, cv2.CV_64F)
        gradient_magnitude = np.abs(gradient)
        # 归一化梯度
        norm_gradient = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
        # 动态调整量化级别
        adaptive_levels = self.base_levels + (self.max_levels - self.base_levels) * (1 - norm_gradient)
        adaptive_levels = np.clip(adaptive_levels, self.base_levels, self.max_levels)
        # 量化图像
        quantized_img = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                level = adaptive_levels[i, j]
                step = 256 // level
                quantized_img[i, j] = (img[i, j] // step) * step
        return quantized_img

# 动态调整量化等级的非均匀量化处理策略
class NonUniformQuantization(ProcessingStrategy):
    """
    动态调整量化等级的非均匀量化处理策略 
    在图像灰度/颜色值变化比较剧烈的地方，量化级别可以减少
    在图像灰度/颜色值变化比较平缓的地方，量化级别可以增加

    1. 计算局部变化程度：使用图像梯度或拉普拉斯算子来计算图像的局部变化程度。
    2. 动态调整量化级别：根据局部变化程度动态调整量化级别。
    """
    def __init__(self, base_levels, max_levels):
        self.base_levels = base_levels
        self.max_levels = max_levels

    def apply(self, img):
        # 将图像转换为灰度图像
        if len(img.shape) == 3:
            gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img

        # 计算图像梯度
        gradient_x = np.zeros_like(gray)
        gradient_y = np.zeros_like(gray)
        gradient_x[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) / 2
        gradient_y[1:-1, :] = (gray[2:, :] - gray[:-2, :]) / 2
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # 图像梯度归一化
        norm_gradient = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min())
        
        # 动态调整量化级别
        adaptive_levels = self.base_levels + (self.max_levels - self.base_levels) * (1 - norm_gradient)
        adaptive_levels = np.clip(adaptive_levels, self.base_levels, self.max_levels)
        
        # 量化图像
        quantized_img = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                level = adaptive_levels[i, j]
                step = 256 // level
                quantized_img[i, j] = (img[i, j] // step) * step
        
        return quantized_img


class Main(QMainWindow, Ui_MainWindow):
    """重写主窗体类"""
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)
        self.setupUi(self) # 初始化窗体显示
        self.timer = QTimer(self) # 初始化定时器  定时器是为了显示视频
        # 设置在label中自适应显示图片
        self.label_PrePicShow.setScaledContents(True)    # 设置label自适应显示图片
        self.label_PrePicShow.setStyleSheet("QLabel{background-color:rgb(0,0,0);}")  # 初始黑化图像显示区域
        self.label_AftPicShow.setScaledContents(True)
        self.label_AftPicShow.setStyleSheet("QLabel{background-color:rgb(0,0,0);}")
        self.img = None
        self.imgDeal = None
        self.method = 0
        # 初始化采样间隔
        self.sampling_interval = 1
        # 初始化量化等级
        self.quantization_level = 256
        # 初始化处理策略列表
        self.processing_strategies = []

        # 初始化非均匀采样参数
        self.min_interval = 1
        self.max_interval = 4

        # 初始化非均匀量化参数
        self.base_levels = 4
        self.max_levels = 16

        # 连接下拉框选择非均匀变化信号到槽函数
        self.comboBox_NonUniformSampling.currentIndexChanged.connect(self.update_non_uniform_sampling)
        self.comboBox_NonUniformQuantization.currentIndexChanged.connect(self.update_non_uniform_quantization)

    def onbuttonclick_selectDataType(self, index):
        """选择输入数据类型(图像,视频)  显示图像或视频"""
        if index == 1:
            filename, _ = QFileDialog.getOpenFileName(self, "选择图像", os.getcwd(), "Images (*.jpg *.png *.bmp);;All (*)")
            self.img = cv2.imread(filename)                                     # 读取图像
            self.label_PrePicShow.setPixmap(QPixmap(filename))                  # 在label中显示图像
        elif index == 2:
            filename, _ = QFileDialog.getOpenFileName(self, "选择视频", os.getcwd(), "Videos (*.avi *.mp4);;All (*)")
            self.capture = cv2.VideoCapture(filename)                           # 读取视频

            self.fps = self.capture.get(cv2.CAP_PROP_FPS)                       # 获得视频帧率

            self.timer.timeout.connect(self.slot_video_display)                 # 定时器超时触发槽函数
            flag, self.img = self.capture.read()                                # 显示视频第一帧
            if flag:
                self.img_show(self.label_PrePicShow, self.img)                  # 在label中显示视频第一帧

    def onbuttonclick_videodisplay(self):
        """显示视频控制函数, 用于连接定时器超时触发槽函数"""
        if self.pushButton_VideoDisplay.text() == "检测":                       # 判断按钮状态   当按钮为检测时
            self.timer.start(1000 / self.fps)                                   # 启动定时器，1000/fps为每帧间隔时间
            self.pushButton_VideoDisplay.setText("暂停")                        # 按下检测按钮后，按钮显示为暂停
        else:
            self.timer.stop()                                                   # 暂停定时器(此时按钮应显示的是暂停)
            self.pushButton_VideoDisplay.setText("检测")                        # 按下暂停按钮后，按钮显示为检测

    def slot_video_display(self):
        """定时器超时触发槽函数, 在label上显示每帧视频, 防止卡顿"""
        flag, self.img = self.capture.read()                                    # 读取视频帧
        if flag:
            self.img_show(self.label_PrePicShow, self.img)                      # 在label中显示视频帧
        else:
            self.capture.release()
            self.timer.stop()

    def oncombox_selectColorType(self, index):
        """选择图像色彩处理方式"""
        self.imgDeal = color_deal(self.img, index, self, self.method)            # 调用色彩处理函数 color_deal,处理方式却决于index
        self.apply_processing()  # 应用处理策略
        self.pushButton_SaveImage.setVisible(True)                              # 显示另存为按钮

    def deal_method(self, index):
        """选择色彩处理是用手写函数还是OpenCV方法"""
        self.method = index

    def img_show(self, label, img):                                             # 在label中显示图片 
        """图片在对应label中显示"""
        # 将opencv图像转换为QImage
        # 彩色图
        if img.shape[-1] == 3:                                                  # 判断图片通道数
            qimage = QImage(img.data, img.shape[1], img.shape[0], img.shape[1]*3, QImage.Format_RGB888).rgbSwapped()
        # 灰度图
        else:
            qimage = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Indexed8)
        # 在对应label中显示图片
        label.setPixmap(QPixmap.fromImage(qimage))

    def onbuttonclick_saveimage(self):
        """将处理后的图片另存为"""
        filename, _ = QFileDialog.getSaveFileName(self, "保存图片", os.getcwd(), "Images (*.jpg *.png);;All (*)")
        cv2.imwrite(filename, self.imgDeal)                                         # 保存图片

    def update_debug_info(self, message):
        """更新调试信息"""
        self.label_DebugInfo.setText(f"调试信息：{message}")

    def update_sampling_interval(self, value):
        """更新采样间隔"""
        self.sampling_interval = max(1, value)  # 确保采样间隔至少为1
        self.update_processing_strategies()
        self.apply_processing()
        
    def update_quantization_level(self, value):
        """更新量化等级"""
        self.quantization_level = max(2, value)
        self.update_processing_strategies()
        self.apply_processing()

    def update_non_uniform_sampling(self, index):
        """更新非均匀采样处理策略"""
        if index == 1:
            self.processing_strategies = [NonUniformSampling(self.min_interval, self.max_interval)]   # 参数值分别是 最小采样间隔, 最大采样间隔
        else:
            self.update_processing_strategies()
        self.apply_processing()

    def update_non_uniform_quantization(self, index):
        """更新自适应量化处理策略"""
        if index == 1:
            self.processing_strategies = [NonUniformQuantization(self.base_levels, self.max_levels)]
        else:
            self.update_processing_strategies()
        self.apply_processing()

    def update_processing_strategies(self):
        """更新处理策略列表"""
        self.processing_strategies = [
            UniformSampling(self.sampling_interval),
            UniformQuantization(self.quantization_level)
        ]

    def apply_processing(self):
        """应用处理策略到处理后的图片"""
        if self.imgDeal is not None:
            try:
                processed_img = self.imgDeal
                for strategy in self.processing_strategies:
                    processed_img = strategy.apply(processed_img)
                self.img_show(self.label_AftPicShow, processed_img)
                self.imgDeal = processed_img
                self.update_debug_info("处理成功")
            except Exception as e:
                self.update_debug_info(f"处理失败: {e}")
                print(e)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Main()
    win.show()
    sys.exit(app.exec_())