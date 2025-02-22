import cv2 as cv
import numpy as np
from skimage import morphology
import math as m
from numba import jit  # 转换为机器代码，加速运算
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import *
from Experiment5.Conv5_UI import Ui_Dialog as UI_Conv
import time

def harris_corner_detect(img, k=0.04):
    """
    手写实现Harris角点检测
    :param img: 输入图像
    :param k: Harris参数，默认0.04
    :return: 标记了角点的彩色图像
    """
    time1 = time.time()
    if len(img.shape) == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 转换为float32类型
    gray = np.float32(gray)
    
    # 计算x和y方向的梯度
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    Ix = cv.filter2D(gray, -1, kernel_x)
    Iy = cv.filter2D(gray, -1, kernel_y)
    
    # 计算梯度乘积
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy
    
    # 使用高斯滤波滑
    kernel_size = 3
    sigma = 2.0
    gaussian_kernel = cv.getGaussianKernel(kernel_size, sigma)
    gaussian_kernel = gaussian_kernel @ gaussian_kernel.T
    
    Ixx = cv.filter2D(Ixx, -1, gaussian_kernel)
    Ixy = cv.filter2D(Ixy, -1, gaussian_kernel)
    Iyy = cv.filter2D(Iyy, -1, gaussian_kernel)
    
    # 计算Harris响应
    det = Ixx * Iyy - Ixy * Ixy
    trace = Ixx + Iyy
    harris_response = det - k * trace * trace
    
    # 阈值处理
    threshold = harris_response.max() * 0.01
    
    # 非极大值抑制
    window_size = 3
    corner_mask = np.zeros_like(harris_response, dtype=bool)
    rows, cols = harris_response.shape
    
    for i in range(window_size, rows - window_size):
        for j in range(window_size, cols - window_size):
            if harris_response[i, j] > threshold:
                window = harris_response[i-window_size:i+window_size+1, 
                                      j-window_size:j+window_size+1]
                if harris_response[i, j] == window.max():
                    corner_mask[i, j] = True
    
    # 创建输出图像
    if len(img.shape) == 2:
        mark = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    else:
        mark = img.copy()
    
    # 标记角点
    y_coords, x_coords = np.where(corner_mask)
    for y, x in zip(y_coords, x_coords):
        # 画圆标记角点
        cv.circle(mark, (x, y), 1, (0, 0, 255), 1)  # 增加圆的半径和线宽
        # 画十字标记
        cv.drawMarker(mark, (x, y), (0, 0, 255), 
                     markerType=cv.MARKER_CROSS,
                     markerSize=2,  # 增加十字标记的大小
                     thickness=1)    # 增加线宽
    
    time2 = time.time()
    print(f"手写Harris角点检测(k={k:.3f})：{(time2-time1)*1000:.3f}毫秒")
    return mark

# def harris_corner_detect(img, k=0.04):
#     """
#     使用OpenCV实现的Harris角点检测
#     :param img: 输入图像
#     :param k: Harris参数，默认0.04
#     :return: 标记了角点的彩色图像
#     """
#     time1 = time.time()
#     if len(img.shape) == 3:
#         gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     else:
#         gray = img
#     gray = np.float32(gray)
    
#     dst = cv.cornerHarris(gray, blockSize=2, ksize=3, k=k)
#     dst = cv.dilate(dst, None)
    
#     if len(img.shape) == 2:
#         mark = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
#     else:
#         mark = img.copy()
    
#     # 标记角点 画圆和十字标记且清晰
#     mark[dst > 0.01 * dst.max()] = [0, 0, 255]

#     time2 = time.time()
#     print(f"OpenCV_Harris角点检测(k={k:.3f})：{(time2-time1)*1000:.3f}毫秒")
#     return mark

def susan_corner_detect(img, t=27):
    """
    使用OpenCV的FAST角点检测器实现
    :param img: 输入图像
    :param t: 亮度差阈值，默认27
    :return: 标记了角点的彩色图像
    """
    time1 = time.time()
    
    # 转换为灰度图
    if len(img.shape) == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 创建FAST检测器
    fast = cv.FastFeatureDetector_create(
        threshold=t,              # 亮度差阈值
        nonmaxSuppression=True,  # 使用非极大值抑制
        type=cv.FAST_FEATURE_DETECTOR_TYPE_9_16  # 使用16点圆形模板
    )
    
    # 检测关键点
    keypoints = fast.detect(gray, None)
    
    # 创建输出图像
    if len(img.shape) == 2:
        mark = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    else:
        mark = img.copy()
    
    # 标记角点
    for kp in keypoints:
        x, y = map(int, kp.pt)
        # 画圆标记角点
        cv.circle(mark, (x, y), 1, (0, 0, 255), 1)
        # 画十字标记
        cv.drawMarker(mark, (x, y), (0, 0, 255), 
                     markerType=cv.MARKER_CROSS,
                     markerSize=2,
                     thickness=1)
    
    time2 = time.time()
    print(f"FAST角点检测(t={t})：{(time2-time1)*1000:.3f}毫秒")
    print(f"检测到{len(keypoints)}个角点")
    return mark

# def susan_corner_detect(img, t=27):
#     """
#     手写实现SUSAN角点检测
#     :param img: 输入图像
#     :param t: 亮度差阈值，默认27
#     :return: 标记了角点的彩色图像
#     """
#     time1 = time.time()
    
#     # 转换为灰度图
#     if len(img.shape) == 3:
#         gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     else:
#         gray = img.copy()
    
#     # 创建圆形模板
#     radius = 6
#     template = np.zeros((2*radius+1, 2*radius+1), dtype=bool)
#     y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
#     mask = x*x + y*y <= radius*radius
#     template[mask] = True
    
#     # 计算模板面积
#     template_area = np.sum(template)
#     g = template_area * 0.3  # 进一步降低USAN面积阈值
    
#     # 填充边界以处理边缘像素
#     padded = cv.copyMakeBorder(gray, radius, radius, radius, radius, cv.BORDER_REFLECT)
    
#     # 存储USAN面积和角点响应
#     rows, cols = gray.shape
#     usan_area = np.zeros_like(gray, dtype=float)
#     corner_response = np.zeros_like(gray, dtype=float)
    
#     # 计算USAN面积和角点响应
#     for i in range(radius, rows+radius):
#         for j in range(radius, cols+radius):
#             nucleus = padded[i, j]
#             neighborhood = padded[i-radius:i+radius+1, j-radius:j+radius+1]
#             diff = np.abs(neighborhood - nucleus)
#             similar = (diff <= t)
#             area = np.sum(similar & template)
#             usan_area[i-radius, j-radius] = area
#             if area < g:
#                 corner_response[i-radius, j-radius] = (g - area) / g
    
#     # 非极大值抑制
#     window_size = 9  # 增大抑制窗口
#     min_response = 0.9  # 提高最小响应阈值
#     corner_mask = np.zeros_like(corner_response, dtype=bool)
    
#     for i in range(window_size, rows-window_size):
#         for j in range(window_size, cols-window_size):
#             if corner_response[i,j] > min_response:
#                 window = corner_response[i-window_size:i+window_size+1, 
#                                       j-window_size:j+window_size+1]
#                 if corner_response[i,j] == window.max():
#                     local_max = window.max()
#                     local_mean = window.mean()
#                     if local_max > 3.0 * local_mean:  # 提高局部对比度要求
#                         corner_mask[i,j] = True
    
#     # 创建输出图像
#     if len(img.shape) == 2:
#         mark = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
#     else:
#         mark = img.copy()
    
#     # 标记角点
#     y_coords, x_coords = np.where(corner_mask)
#     for y, x in zip(y_coords, x_coords):
#         cv.circle(mark, (x, y), 1, (0, 0, 255), 1)
#         cv.drawMarker(mark, (x, y), (0, 0, 255), 
#                      markerType=cv.MARKER_CROSS,
#                      markerSize=2,
#                      thickness=1)
    
#     time2 = time.time()
#     print(f"手写SUSAN角点检测(t={t})：{(time2-time1)*1000:.3f}毫秒")
#     print(f"检测到{len(y_coords)}个角点")
#     return mark

def count_cells(img, threshold=0.7):
    """
    细胞计数功能
    :param img: 输入图像
    :param threshold: 圆形度阈值，默认0.7
    :return: 标记了细胞的图像和细胞数量
    """
    time1 = time.time()
    
    # 转换为灰度图
    if len(img.shape) == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # 图像预处理
    # 高斯滤波去噪
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu二值化
    _, binary = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)  # 添加THRESH_BINARY_INV
    
    # 形态学操作
    kernel = np.ones((3,3), np.uint8)
    # 开运算去除小噪点
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)
    # 闭运算填充小孔
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=2)
    
    # 连通区域分析
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary, connectivity=8)
    
    # 创建输出图像
    if len(img.shape) == 2:
        mark = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    else:
        mark = img.copy()
    
    # 调试信息
    print(f"检测到{num_labels-1}个连通区域")
    
    # 统计符合条件的细胞
    cell_count = 0
    min_area = 50  # 降低最小面积阈值
    max_area = 5000  # 添加最大面积阈值
    
    # 遍历所有连通区域（跳过背景）
    for i in range(1, num_labels):
        area = stats[i, cv.CC_STAT_AREA]
        
        # 调试信息
        print(f"区域 {i}: 面积={area}")
        
        # 过滤面积不合适的区域
        if area < min_area or area > max_area:
            continue
            
        # 获取当前连通区域的掩码
        mask = (labels == i).astype(np.uint8)
        
        # 查找轮廓
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        
        # 计算圆形度
        perimeter = cv.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # 调试信息
        print(f"区域 {i}: 圆形度={circularity:.3f}")
        
        # 如果圆形度大于阈值，认为是细胞
        if circularity > threshold:
            cell_count += 1
            
            # 获取中心点
            x = int(centroids[i][0])
            y = int(centroids[i][1])
            
            # 在图像上标记细胞
            cv.drawContours(mark, [contour], -1, (0, 255, 0), 2)  # 绿色轮廓
            cv.circle(mark, (x, y), 3, (0, 0, 255), -1)  # 红色中心点
            cv.putText(mark, f'{cell_count}', (x-10, y-10), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # 蓝色编号
    
    time2 = time.time()
    print(f"细胞计数完成，共检测到{cell_count}个细胞")
    print(f"处理时间：{(time2-time1)*1000:.3f}毫秒")
    
    # 返回二值化图像用于调试
    binary_debug = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    return mark, cell_count, binary_debug
