import cv2 as cv
import numpy as np
import math as m
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import *
from Experiment4.Linear_UI import Ui_Dialog as UI_Linear
from Experiment4.Log_UI import Ui_Dialog as UI_Log
from Experiment4.Exp_UI import Ui_Dialog as UI_Exp
from Experiment4.Pow_UI import Ui_Dialog as UI_Pow
from Experiment4.Conv_UI import Ui_Dialog as UI_Conv
import time

def gray_deal(img, deal_Type):
    """根据用户的选择，对于图像做相应的灰度增强处理"""
    if img.shape[-1] == 3:
        pass
    if deal_Type == 1:
        img = linear_strench(img)
    elif deal_Type == 2:
        img = log_strench(img)
    elif deal_Type == 3:
        img = exp_strench(img)
    elif deal_Type == 4:
        img = pow_strench(img)
    return img
# ---------------------- 灰度变换 ---------------------- #
# 线性变换
def linear_strench(img):
    """*功能 : 根据传入的图像及给定的c,d两个灰值区间参数值，进行线性拉伸
    *注意，只对灰度图像拉伸，函数：g(x,y)=(d-c)/(b-a)*[f(x,y)-a]+c=k*[f(x,y)-a]+c"""
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    a, b = np.min(img), np.max(img)
    # 交互对话框
    q_dialog = QDialog()
    dialog = UI_Linear()
    dialog.setupUi(q_dialog) # 继承QDialog()， 使得dialog具有show()方法
    dialog.lineEdit_a.setText(str(a))  # 显示原图灰度范围
    dialog.lineEdit_b.setText(str(b))
    dialog.lineEdit_c.setText(str(a))  # 初始化变换后灰度范围
    dialog.lineEdit_d.setText(str(b))
    q_dialog.show()
    if q_dialog.exec() == QDialog.Accepted:  # 提取用户交互变换后灰度范围
        c = int(dialog.lineEdit_c.text())
        d = int(dialog.lineEdit_d.text())

        time1 = time.time() # 程序计时开始
        # 核心代码：线性拉伸
        k = (d - c) / (b - a)    
        new_img = k * (img - a) + c                          # img - a 利用广播机制，对每个像素进行操作
        new_img = np.clip(new_img, 0, 255).astype(np.uint8)  # 确保像素值在0-255之间 且像素值为整数

        time2 = time.time() # 程序计时结束
        print("灰度增强程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
        return new_img
# 对数变换
def log_strench(img):
    """*功能 : 根根据传入的图像及给定的a,b,c三个参数值，进行对数非线性拉伸
    *注意，只对灰度进行拉伸，函数：g(x,y)=a+lg[f(x,y)+1]/(c*lgb)
    a，b，c分别表示对数变换的平移、缩放和对数底数

    对数变换的作用是扩展图像的低灰度值区域，（因为对数函数在x比较小时，y增长较快，x比较大时，y增长较慢）而压缩高灰度值区域
    """
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    # ui交互
    q_dialog = QDialog()
    dialog = UI_Log()
    dialog.setupUi(q_dialog) # 继承QDialog()， 使得dialog具有show()方法
    dialog.lineEdit_a.setText(str(0.0))  # 初始化对数变换参数
    dialog.lineEdit_b.setText(str(2.0))
    dialog.lineEdit_c.setText(str(0.03))
    q_dialog.show()
    # 用户交互判断
    if q_dialog.exec() == QDialog.Accepted:  # 提取用户交互的参数
        a, b, c = float(dialog.lineEdit_a.text()), float(dialog.lineEdit_b.text()), float(dialog.lineEdit_c.text())
        if c == 0 or b <= 0 or b == 1: # 对参数进行预判断
            if QMessageBox(QMessageBox.Warning, '警告', '参数设置不合理！').exec():
                return img

        time1 = time.time() # 程序计时开始
        # 核心代码：对数变换
        new_img = a + np.log1p(img) / (c * np.log(b))        # np.log1p(x) = np.log(1+x)  还是利用广播机制
        new_img = np.clip(new_img, 0, 255).astype(np.uint8)  # 确保像素值在0-255之间 且像素值为整数

        time2 = time.time() # 程序计时结束
        print("对数变换程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img
# 指数变换
def exp_strench(img):
    """*功能 : 根根据传入的图像及给定的a,b,c三个参数值，进行指数非线性拉伸
    *注意，只对灰度进行拉伸，函数：g(x,y)=b^c[f(x,y)-a]-1

    指数函数的作用是扩展图像的高灰度值区域，压缩低灰度值区域 因为指数函数在x比较小时，y增长较慢，x比较大时，y增长较快
    """
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像

    q_dialog = QDialog()
    dialog = UI_Exp()
    dialog.setupUi(q_dialog) # 继承QDialog()， 使得dialog具有show()方法
    dialog.lineEdit_a.setText(str(150))  # 初始化对数变换参数
    dialog.lineEdit_b.setText(str(1.5))
    dialog.lineEdit_c.setText(str(0.6))
    q_dialog.show()
    # 用户交互判断
    if q_dialog.exec() == QDialog.Accepted:  # 提取用户交互的参数
        a, b, c = float(dialog.lineEdit_a.text()), float(dialog.lineEdit_b.text()), float(dialog.lineEdit_c.text())
        if b <= 0 or b == 1: # 对参数进行预判断
            if QMessageBox(QMessageBox.Warning, '警告', '参数设置不合理！').exec():
                return img

        time1 = time.time() # 程序计时开始

        # 核心代码：指数变换
        new_img = np.power(b, c * (img - a)) - 1             # 还是利用广播机制
        new_img = np.clip(new_img, 0, 255).astype(np.uint8)  # 确保像素值在0-255之间 且像素值为整数

        time2 = time.time() # 程序计时结束
        print("指数变换程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img
# 幂律变换
def pow_strench(img):
    """*功能 : 根根据传入的图像及给定的c,r两个参数值，进行幂律非线性拉伸
    *注意，只对灰度进行拉伸，函数：g(x,y)=c[f(x,y)]^r

    幂律变换可以实现两种不同的效果，当r>1时，扩展高灰度值区域，压缩低灰度值区域，类似指数变换。当r<1时，扩展低灰度值区域，压缩高灰度值区域，类似对数变换。
    """
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像

    q_dialog = QDialog()
    dialog = UI_Pow()
    dialog.setupUi(q_dialog) # 继承QDialog()， 使得dialog具有show()方法
    dialog.lineEdit_c.setText(str(1))  # 初始化对数变换参数
    dialog.lineEdit_r.setText(str(1.5))
    q_dialog.show()
    # 用户交互判断
    if q_dialog.exec() == QDialog.Accepted:  # 提取用户交互的参数
        c, r = float(dialog.lineEdit_c.text()), float(dialog.lineEdit_r.text())
        if r <= 0 or c <= 0: # 对参数进行预判断
            if QMessageBox(QMessageBox.Warning, '警告', '参数设置不合理！').exec():
                return img

        time1 = time.time() # 程序计时开始

        # 核心代码：幂律变换
        new_img = c * np.power(img, r)                      # 还是利用广播机制
        new_img = np.clip(new_img, 0, 255).astype(np.uint8) # 确保像素值在0-255之间 且像素值为整数

        time2 = time.time() # 程序计时结束
        print("幂律变换程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img

# 直方图均衡化
def hist_equalization(img, jug):
    """*功能 : 直方图均衡化算法, jug判断返回是图像/直方图 jug实际是ui中的判断是否显示直方图的变量"""
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    time1 = time.time()  # 程序计时开始
    hist = creat_histogram(img)                    # 计算直方图

    # 核心代码：直方图均衡化
    if img.ndim == 2:  # 灰度图像
        cdf = np.cumsum(hist)                                                  # cumsum()计算累加和
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())     # 将累积分布函数归一化到0-255
        new_img = cdf_normalized[img].astype(np.uint8)
    else:  # 彩色图像
        for i in range(3):
            cdf = np.cumsum(hist[i])
            cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
            new_img[:,:,i] = cdf_normalized[img[:,:,i]].astype(np.uint8)

    time2 = time.time()  # 程序计时结束
    print("图像均衡算法程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    if jug:  # 用opencv的方法把两幅图像的直方图绘制在一张图上
        imgs = [img, new_img]
        colors = ("b", "r")
        texts = ("original histogram", "histogram after equalization")
        for i in range(2):
            hist = cv.calcHist([imgs[i]], [0], None, [256], [0, 255])
            plt.plot(hist, color=colors[i], label=texts[i])
        plt.xlim([0, 256])
        plt.legend()
        plt.show()
    return new_img
# 计算传入图像的直方图  遍历法
def creat_histogram(img):
    """*功能 : 计算传入图像的直方图，若是彩色图像，计算各颜色分量直方图并返回"""
    rows, cols = img.shape[:2]  # 获取宽和高
    hist = []
    if img.ndim == 2: # 灰度图像统计直方图
        hist = [0] * 256  # 建立灰度图像直方图  hist是一个256个元素的列表 每个元素表示一个灰度值出现的次数
        # 图像遍历  把每个像素的灰度值作为直方图的索引，统计每个灰度值出现的次数
        for row in range(rows):
            for col in range(cols):
                hist[img[row][col]] += 1
    elif img.ndim == 3:  # 彩色图像统计直方图
        hist = [[0] * 256, [0] * 256, [0] * 256]  # 建立彩色图像直方图  hist是一个包含3个元素的列表 每个元素是一个256个元素的列表 表示每个通道的直方图 这个列表的索引表示该通道下一个灰度值的出现次数
        # 图像遍历  把每个像素每个通道的灰度值作为直方图的索引，统计每个灰度值出现的次数
        for row in range(rows):
            for col in range(cols):
                hist[0][img[row][col][0]] += 1
                hist[1][img[row][col][1]] += 1
                hist[2][img[row][col][2]] += 1
    return hist

# 计算传入图像的直方图  Numpy版本
# def creat_histogram(img):
#     """*功能 : 计算传入图像的直方图，若是彩色图像，计算各颜色分量直方图并返回"""
#     if img.ndim == 2:  # 灰度图像
#         hist = np.bincount(img.ravel(), minlength=256)     # img.ravel() 将图像像素值展平为一维数组 至少有256 统计每个像素值出现次数
#     elif img.ndim == 3:  # 彩色图像
#         hist = [np.bincount(img[..., i].ravel(), minlength=256) for i in range(3)]  # img[..., i] 表示取图像的第i个通道 每个通道的灰度值展平为一维数组 至少有256 统计每个像素值出现次数
#     return hist

# 计算传入图像的直方图  OpenCV版本
# def creat_histogram(img):
#     """*功能 : 计算传入图像的直方图，若是彩色图像，计算各颜色分量直方图并返回"""
#     if img.ndim == 2:  # 灰度图像
#         # [img] 输入图像  [0] 通道索引  None 掩码图像  [256] 直方图 bins  [0, 255] 像素值范围
#         hist = cv.calcHist([img], [0], None, [256], [0, 255])     # calcHist返回的是一个形状为(256, 1)的二维数组，256行，1列 每个元素都是一个列表（里面只有一个元素，表示第该灰度值的出现次数）表示每个灰度值出现的次数 
#         hist = hist.ravel()  # 将二维数组展平为一维数组 每个元素表示一个灰度值出现的次数 （一个列表）
#     elif img.ndim == 3:  # 彩色图像
#         hist = [cv.calcHist([img], [i], None, [256], [0, 255]).ravel() for i in range(3)]  #每个通道的直方图
#     return hist

def gray_smooth(img, deal_Type):
    """根据用户的选择，对于图像做相应的图像平滑处理"""
    if img.shape[-1] == 3:
        pass
    if deal_Type == 1:
        img = neighbor_average(img)
    elif deal_Type == 2:
        img = median_filter(img)
    return img

# 邻域平均平滑 numpy实现
def neighbor_average(img):
    """*功能 : 用户交互卷积模板，获取卷积系数进行邻域平滑，只对灰度图像处理"""
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    q_dialog = QDialog()
    dialog = UI_Conv()
    dialog.setupUi(q_dialog) # 继承QDialog()， 使得dialog具有show()方法
    q_dialog.show()
    if q_dialog.exec() == QDialog.Accepted:  # 提取用户交互的参数
        np_kernel = np.array([[float(dialog.lineEdit1.text()), float(dialog.lineEdit2.text()), float(dialog.lineEdit3.text())],
                             [float(dialog.lineEdit4.text()), float(dialog.lineEdit5.text()), float(dialog.lineEdit6.text())],
                             [float(dialog.lineEdit7.text()), float(dialog.lineEdit8.text()), float(dialog.lineEdit9.text())]])
        np_kernel = np_kernel/np_kernel.sum() # 正则化

        time1 = time.time() # 程序计时开始

        # 核心代码：邻域平均平滑 numpy实现
        padded_img = np.pad(img, ((1, 1), (1, 1)), mode='edge')                     # 边缘填充成边缘像素值
        windows = np.lib.stride_tricks.sliding_window_view(padded_img, (3, 3))      # numpy的滑动窗口函数 3×3窗口 边缘填充成边缘像素值
        new_img = np.sum(windows * np_kernel, axis=(2, 3))                          # 利用广播机制，对每个像素进行操作 每个像素乘以卷积核的对应元素求和
        
        new_img = np.clip(new_img, 0, 255).astype(np.uint8)                         # 确保像素值在0-255之间 且像素值为整数

        time2 = time.time() # 程序计时结束
        print("邻域平均平滑程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img

# 邻域平均平滑 opencv实现
# def neighbor_average(img):
#     """*功能 : 用户交互卷积模板，获取卷积系数进行邻域平滑，只对灰度图像处理"""
#     rows, cols = img.shape[:2]  # 获取宽和高
#     new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
#     q_dialog = QDialog()
#     dialog = UI_Conv()
#     dialog.setupUi(q_dialog) # 继承QDialog()， 使得dialog具有show()方法
#     q_dialog.show()
#     if q_dialog.exec() == QDialog.Accepted:  # 提取用户交互的参数
#         np_kernel = np.array([[float(dialog.lineEdit1.text()), float(dialog.lineEdit2.text()), float(dialog.lineEdit3.text())],
#                              [float(dialog.lineEdit4.text()), float(dialog.lineEdit5.text()), float(dialog.lineEdit6.text())],
#                              [float(dialog.lineEdit7.text()), float(dialog.lineEdit8.text()), float(dialog.lineEdit9.text())]])
#         np_kernel = np_kernel/np_kernel.sum() # 正则化

#         time1 = time.time() # 程序计时开始

#         # 核心代码：邻域平均平滑 opencv实现   filter2D是opencv的卷积函数
#         new_img = cv.filter2D(img, -1, np_kernel)  # img 输入图像  -1 输出图像深度  np_kernel 卷积核

#         time2 = time.time() # 程序计时结束
#         print("邻域平均平滑程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
#     return new_img

# 中值滤波 numpy实现
def median_filter(img):
    """*功能 : 中值滤波"""
    rows, cols = img.shape[:2]  # 获取宽和高
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    len = 3  # 定义中值滤波模板3×3

    time1 = time.time() # 程序计时开始

    # 核心代码：中值滤波 numpy实现
    padded_img = np.pad(img, ((1, 1), (1, 1)), mode='edge')                     # 边缘填充成边缘像素值
    windows = np.lib.stride_tricks.sliding_window_view(padded_img, (3, 3))      # numpy的滑动窗口函数 3×3窗口 边缘填充成边缘像素值
    new_img = np.median(windows, axis=(2, 3)).astype(np.uint8)                  # 每个像素先取出3×3窗口排序，取中值

    time2 = time.time() # 程序计时结束
    print("中值滤波程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img

# 中值滤波 opencv实现
# def median_filter(img):
#     """*功能 : 中值滤波"""
#     rows, cols = img.shape[:2]  # 获取宽和高
#     new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像

#     len = 3  # 定义中值滤波模板3×3

#     time1 = time.time() # 程序计时开始

#     # 核心代码：中值滤波 opencv实现
#     new_img = cv.medianBlur(img, len)    # img 输入图像  len 滤波器孔径的线性尺寸 必须是奇数

#     time2 = time.time() # 程序计时结束
#     print("中值滤波程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
#     return new_img
