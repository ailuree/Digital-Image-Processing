import cv2 as cv
import numpy as np
import math as m
from numba import jit # 转换为机器代码，加速运算
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import *
from Experiment4.Conv_UI import Ui_Dialog as UI_Conv
import time
from scipy.signal import convolve2d

# 边缘检测
def edge_detect(img, deal_Type):
    """根据用户的选择，对于图像做相应的灰度增强处理"""
    if img.shape[-1] == 3:
        pass
    if deal_Type == 1:
        img = prewitt(img)
    elif deal_Type == 2:
        img = sobel(img)
    elif deal_Type == 3:
        img = log(img)
    elif deal_Type == 4:
        img = canny(img)
    return img
# prewitt算子
def prewitt(img):
    """*功能 : 根据prewitt对应的卷积模板，对图像进行边缘检测
    *注意，这里只引入水平和竖直两个方向边缘检测卷积模板"""
    time1 = time.time()  # 程序计时开始
    # numpy实现 -- 使用FFT进行卷积操作
    # prewitt算子卷积核 水平和竖直方向
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    # 使用快速傅里叶变换（FFT）进行卷积
    img_pad = np.pad(img, ((1, 1), (1, 1)), mode='edge')   # 边缘填充，填充方式为边缘复制 (1,1),(1,1)表示在图像的上下左右各填充一行/列
    img_fft = np.fft.fft2(img_pad)                         # fft2对二维数组进行傅里叶变换  转换后实际上就是一个傅里叶的频谱图
    kernel_x_fft = np.fft.fft2(kernel_x, s=img_pad.shape)  # 对水平卷积核傅里叶变换 同时将卷积核的大小设置为与图像相同 会把卷积核的中心对齐到图像的中心 然后复制卷积核填充到图像的大小
    kernel_y_fft = np.fft.fft2(kernel_y, s=img_pad.shape)
    gx = np.real(np.fft.ifft2(img_fft * kernel_x_fft))[1:-1, 1:-1]  # 频域中直接将两个图像相乘，然后再进行逆变换，得到卷积结果 （卷积定理）
    gy = np.real(np.fft.ifft2(img_fft * kernel_y_fft))[1:-1, 1:-1]
    # 计算梯度幅值
    new_img = np.sqrt(gx**2 + gy**2)    # 把水平和竖直方向的梯度幅值相加再开方 最后梯度高的地方即边缘会有更高的值
    # 归一化到0-255
    new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    time2 = time.time()  # 程序计时结束
    print("prewitt算子边缘检测NumPy FFT优化程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # # numpy实现  -- 使用scipy的convolve2d进行卷积操作
    # # prewitt算子 水平和竖直方向
    # kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    # kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    # # 使用scipy的convolve2d进行卷积操作
    # gx = convolve2d(img, kernel_x, mode='same', boundary='symm')  # mode='same'输出的大小与输入的大小相同，boundary='symm'边界填充方式为对称填充
    # gy = convolve2d(img, kernel_y, mode='same', boundary='symm')
    # # 计算梯度幅值
    # new_img = np.sqrt(gx**2 + gy**2)
    # # 归一化到0-255
    # new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    # time2 = time.time()  # 程序计时结束
    # print("prewitt算子边缘检测NumPy scipy-convolve2d优化程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # # opencv实现
    # kernelx = cv.getDerivKernels(1, 0, 3)  
    # kernely = cv.getDerivKernels(0, 1, 3)
    # img_prewittx = cv.sepFilter2D(img, cv.CV_32F, kernelx[0], kernelx[1]) 
    # img_prewitty = cv.sepFilter2D(img, cv.CV_32F, kernely[0], kernely[1])
    # new_img = cv.magnitude(img_prewittx, img_prewitty)  # 计算梯度幅值
    # new_img = np.uint8(np.clip(new_img, 0, 255))
    
    # time2 = time.time()  # 程序计时结束
    # print("prewitt算子边缘检测CV程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img
# sobel算子
def sobel(img):
    """*功能 : 根据sobel对应的卷积模板，对图像进行边缘检测
       *注意，这里只引入水平和竖直两个方向边缘检测卷积模板"""
    time1 = time.time()  # 程序计时开始
    # numpy实现 -- 使用FFT进行卷积操作
    # sobel算子 水平和竖直方向
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    # 使用快速傅里叶变换（FFT）进行卷积
    img_pad = np.pad(img, ((1, 1), (1, 1)), mode='edge')
    img_fft = np.fft.fft2(img_pad)
    kernel_x_fft = np.fft.fft2(kernel_x, s=img_pad.shape)
    kernel_y_fft = np.fft.fft2(kernel_y, s=img_pad.shape)
    gx = np.real(np.fft.ifft2(img_fft * kernel_x_fft))[1:-1, 1:-1]
    gy = np.real(np.fft.ifft2(img_fft * kernel_y_fft))[1:-1, 1:-1]
    # 计算梯度幅值
    new_img = np.sqrt(gx**2 + gy**2)
    # 归一化到0-255
    new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    time2 = time.time()  # 程序计时结束
    print("sobel算子边缘检测NumPy优化程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # # numpy实现 -- 使用scipy的convolve2d进行卷积操作
    # # sobel算子 水平和竖直方向
    # kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    # kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    # gx = convolve2d(img, kernel_x, mode='same', boundary='symm')
    # gy = convolve2d(img, kernel_y, mode='same', boundary='symm')
    # # 计算梯度幅值
    # new_img = np.sqrt(gx**2 + gy**2)
    # # 归一化到0-255
    # new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    # time2 = time.time()  # 程序计时结束
    # print("sobel算子边缘检测NumPy scipy-convolve2d优化程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # # opencv实现
    # sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    # sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)
    # new_img = cv.magnitude(sobelx, sobely)
    # new_img = np.uint8(np.clip(new_img, 0, 255))
    # time2 = time.time()  # 程序计时结束
    # print("sobel算子边缘检测CV程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img
# log算子
def log(img):
    """
    *功能 : 根据LOG算子对应的卷积模板，对图像进行边缘检测
    发现：在new_img = np.clip(new_img, 0, 255).astype(np.uint8)分割前如果先new_img = np.abs(new_img)取绝对值就会使边缘很模糊，去掉绝对值就会很好
    """
    time1 = time.time()  # 程序计时开始
    # numpy实现 -- 使用FFT进行卷积操作
    prw_conv = np.array([[-2, -4, -4, -4, -2], [-4, 0, 8, 0, -4], [-4, 8, 24, 8, -4],
                         [-4, 0, 8, 0, -4], [-2, -4, -4, -4, -2]], dtype=float)
    # 高斯模糊
    img = cv.GaussianBlur(img, (3, 3), 0)
    # 使用快速傅里叶变换（FFT）进行卷积
    img_pad = np.pad(img, ((2, 2), (2, 2)), mode='edge') # 上下左右各填充两行/列 
    img_fft = np.fft.fft2(img_pad)
    kernel_fft = np.fft.fft2(prw_conv, s=img_pad.shape)
    new_img = np.real(np.fft.ifft2(img_fft * kernel_fft))[2:-2, 2:-2] # 取实部
    # 取绝对值并限制在0-255范围内
    new_img = np.clip(new_img, 0, 255).astype(np.uint8)  
    time2 = time.time()  # 程序计时结束
    print("log算子边缘检测NumPy FFT优化程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # # numpy实现 -- 使用scipy的convolve2d进行卷积操作
    # prw_conv = np.array([[-2, -4, -4, -4, -2], [-4, 0, 8, 0, -4], [-4, 8, 24, 8, -4],
    #                      [-4, 0, 8, 0, -4], [-2, -4, -4, -4, -2]], dtype=float)
    # # 高斯模糊
    # img = cv.GaussianBlur(img, (3, 3), 0)
    # # 使用scipy的convolve2d进行卷积操作
    # new_img = convolve2d(img, prw_conv, mode='same', boundary='symm')
    # # 取绝对值并限制在0-255范围内
    # new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    # time2 = time.time()  # 程序计时结束
    # print("log算子边缘检测NumPy scipy-convolve2d优化程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # opencv实现
    # 高斯模糊
    img_blur = cv.GaussianBlur(img, (3, 3), 0)
    # 直接对原图应用log算子
    new_img = cv.Laplacian(img_blur, cv.CV_64F, ksize=5)
    # 取绝对值并限制在0-255范围内
    new_img = np.uint8(np.clip(new_img, 0, 255))         
    time2 = time.time()  # 程序计时结束
    print("log算子边缘检测CV程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    return new_img
# canny算子
def canny(img):
    """*功能 : canny算子"""
    time1 = time.time()  # 程序计时开始
    # OpenCV实现
    time1 = time.time()  # 程序计时开始
    new_img = cv.Canny(img, 100, 200)                   # 100是低阈值，200是高阈值
    time2 = time.time()  # 程序计时结束
    print("canny算子边缘检测CV程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img

# 大津阈值分割
def otsu(img, jug):
    """*功能：大津阈值分割，求取直方图数组，根据类内方差最小，类间方差最大原理自动选择阈值，
    *注意：只处理灰度图像"""
    time1 = time.time()  # 程序计时开始
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    hist = cv.calcHist([img], [0], None, [256], [0, 255]) # 直接用cv函数计算直方图，hist: numpy格式
  
    # numpy实现大津阈值分割   （输入灰度图）
    hist = hist.flatten()                   # 确保hist是一维数组  0-255的各个灰度级别的像素个数
    total = hist.sum()                      # 像素总数    
    weight1 = np.cumsum(hist)               # 利用cumsum计算累计像素数和 这里返回的是256大小的数组 每一个元素是前n个元素的和
    weight2 = total - weight1               # 后景 累计和
    mean1 = np.cumsum(hist * np.arange(256)) / (weight1 + 1e-10)  # np.cumsum(hist * np.arange(256)) 为前景的灰度级别累计和 像素数*灰度级别 即像素值和  weight1是前景的像素数  得出前景累计均值
    mean2 = (np.cumsum(hist * np.arange(256))[-1] - np.cumsum(hist * np.arange(256))) / (weight2 + 1e-10)   # 后景累计均值
    variance = weight1 * weight2 * (mean1 - mean2) ** 2  # 类间方差  256大小 表示阈值为0-255时的各个类间方差
    max_t = np.argmax(variance)                          # 找出使类间方差最大的阈值
    new_img[img > max_t] = 255                           # 二值化
    time2 = time.time()  # 程序计时结束
    print("大津阈值numpy程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # # opencv大津阈值分割
    # max_t, new_img = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
    # time2 = time.time()  # 程序计时结束
    # print("大津阈值cv程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    if jug:
        plt.plot(hist, color="r", label="otsu value in histogram")
        plt.xlim([0, 256])
        plt.axvline(max_t, color='green') # 在直方图中绘制出阈值位置
        plt.legend() # 用于给图像加图例，各种符号和颜色所代表内容与指标的说明
        plt.show()
    return new_img
# 霍夫变换做直线和圆检测
def hough_detect(img, deal_Type):
    """根据用户的选择，对于图像做相应的图像平滑处理"""
    if img.shape[-1] == 3:
        pass
    if deal_Type == 1:
        img = line_detect(img, 100)
    elif deal_Type == 2:
        img = circle_detect(img)
    return img
# 霍夫变换 求取霍夫域
def hough_transform(img):
    """根据传入的图像，求取目标点对应的hough域，公式：ρ = x cos θ + y sin θ
    注：默认图像中255点为目标点"""
    rows, cols = img.shape[:2]  # 获取宽和高
    diagonal = int(np.sqrt(rows**2 + cols**2))  # 图像对角线长度 作为霍夫空间的最大ρ值
    hg_rows, hg_cols = 180, diagonal * 2        # 霍夫空间的行和列  行数为180,对应0到179度
    # 确保图像是二值化的
    img = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)[1]
    # 获取所有非零点的坐标
    y_idxs, x_idxs = np.nonzero(img)
    # 预计算所有角度的sin和cos值
    thetas = np.deg2rad(np.arange(hg_rows))   # 生成0-179度的角度数组,并转换为弧度
    cos_thetas = np.cos(thetas)               # 计算所有角度的cos值
    sin_thetas = np.sin(thetas)               # 计算所有角度的sin值
    # 计算ρ值并累加到hough_img
    rhos = np.round(x_idxs[:, None] * cos_thetas + y_idxs[:, None] * sin_thetas).astype(int) # 计算每个点在每个角度下的ρ值 得到很多ρ-θ线
    rhos += diagonal  # 将ρ值移动到正数范围
    # 创建和累加霍夫空间
    hough_img = np.zeros((hg_rows, hg_cols), dtype=np.int32)  # 全0,全黑
    np.add.at(hough_img, (np.arange(hg_rows)[None, :], rhos), 1) # 将每个点在每个角度下的ρ值累加到hough_img中, 每次加1

    return hough_img
# 霍夫变换做直线检测 
def line_detect(img, num):
    """*功能 : 通过hough变换检测直线，num:需检测直线的条数"""
    time1 = time.time()  # 程序计时开始
    rows, cols = img.shape[:2]  # 获取原图宽和高
    # NumPy实现的直线检测
    hough_img = hough_transform(img)  # 获取霍夫域图像
    # 归一化霍夫域图像到0-255  原ρ值比较大在0到400都有
    hough_img_normalized = cv.normalize(hough_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    
    # 找出投票数最高的点
    threshold = np.max(hough_img) // 2   # 设置阈值为最大值的一半  
    peaks = np.argwhere(hough_img > threshold)  # 找出所有投票数大于阈值的点
    
    # 将原图转为彩色图，以便绘制彩色直线
    img_color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    
    diagonal = int(np.sqrt(rows**2 + cols**2))   # 图像对角线长度
    # 公式：ρ = x cos θ + y sin θ  反解出过直线的点
    for theta, rho in peaks[:num]:  # 只绘制前num条线
        a = np.cos(np.deg2rad(theta))  # a = cosθ 是直线的方向向量
        b = np.sin(np.deg2rad(theta))  # b = sinθ 是直线的法向量
        x0 = a * (rho - diagonal)   
        y0 = b * (rho - diagonal)  # (x0, y0)是直线的中点
        x1 = int(x0 + 1000 * (-b)) 
        y1 = int(y0 + 1000 * (a))  
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))   # (x1, y1)和(x2, y2)是直线的两个点,扩展1000个像素确保直线足够长
        # 计算直线的两个端点
        cv.line(img_color, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
    time2 = time.time() # 程序计时结束
    print("hough直线检测手写程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # 返回在原图上标注了直线的图像和归一化后的霍夫域图像
    return img_color, hough_img_normalized

    # # opencv函数检测直线
    # lines = cv.HoughLines(255 - img, 1, np.pi / 180, 100)  # 这里对最后一个参数使用了经验型的值
    # hough_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR) # 创建新图像，用于标记直线
    # if lines is not None:
    #     for line in lines:
    #         p, a = line[0]  # 第一个元素是距离rho, 第二个元素是角度theta
    #         pt_start = (0, int(p/m.sin(a))) # 绘制直线起点
    #         pt_end = (cols, int((p-cols*m.cos(a))/m.sin(a))) # 绘制直线终点
    #         cv.line(hough_img, pt_start, pt_end, (0, 0, 255), 1)
    # time2 = time.time() # 程序计时结束
    # print("hough直线检测opencv程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    
    # return img, hough_img
# 霍夫变换做圆检测
def circle_detect(img):
    """*功能 : 利用opencv中的hough圆检测，检测出图像中的圆"""
    time1 = time.time()  # 程序计时开始
    
    # 预处理
    img_blur = cv.GaussianBlur(img, (5, 5), 0)  # 高斯模糊
    edges = cv.Canny(img_blur, 50, 150, apertureSize=3)  # 边缘检测
    
    # 创建彩色图像用于绘制结果
    new_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    
    # 使用HoughCircles检测圆
    circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, dp=1, minDist=100,
                              param1=100, param2=30, minRadius=10, maxRadius=100)  # 各个参数是 edge图像，检测方法，累加器分辨率，圆心最小距离，Canny边缘阈值，圆心阈值，最小半径，最大半径
    
    # 如果检测到圆，则绘制它们
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # 绘制外圆
            cv.circle(new_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # 绘制圆心
            cv.circle(new_img, (i[0], i[1]), 2, (0, 0, 255), 3)
    
    time2 = time.time()  # 程序计时结束
    print("hough圆检测opencv程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img