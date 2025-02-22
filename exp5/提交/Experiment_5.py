import cv2 as cv
import numpy as np
from skimage import morphology
import math as m
from numba import jit  # 转换为机器代码，加速运算
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import *
from Experiment5.Conv5_UI import Ui_Dialog as UI_Conv
import time
from numpy.lib.stride_tricks import as_strided

def morphy_binary(img, deal_Type):
    """根据用户的选择，对于图像做相应的二值形态学处理"""
    if img.shape[-1] == 3:
        pass
    q_dialog = QDialog()
    dlg = UI_Conv()
    dlg.setupUi(q_dialog)  # 继承QDialog()， 使得dialog具有show()方法
    q_dialog.show()
    if q_dialog.exec() == QDialog.Accepted:  # 提取用户交互的参数
        np_kernel = np.array(
            [[int(dlg.lineEdit1.text()), int(dlg.lineEdit2.text()), int(dlg.lineEdit3.text()), int(dlg.lineEdit4.text()), int(dlg.lineEdit5.text())],
             [int(dlg.lineEdit6.text()), int(dlg.lineEdit7.text()), int(dlg.lineEdit8.text()), int(dlg.lineEdit9.text()), int(dlg.lineEdit10.text())],
             [int(dlg.lineEdit11.text()), int(dlg.lineEdit12.text()), int(dlg.lineEdit13.text()), int(dlg.lineEdit14.text()), int(dlg.lineEdit15.text())],
             [int(dlg.lineEdit16.text()), int(dlg.lineEdit17.text()), int(dlg.lineEdit18.text()), int(dlg.lineEdit19.text()), int(dlg.lineEdit20.text())],
             [int(dlg.lineEdit21.text()), int(dlg.lineEdit22.text()), int(dlg.lineEdit23.text()), int(dlg.lineEdit24.text()), int(dlg.lineEdit25.text())]
            ])

        if deal_Type == 1:
            img = erosion_binary(img, np_kernel)
        elif deal_Type == 2:
            img = dilation_binary(img, np_kernel)
        elif deal_Type == 3:
            img = open_binary(img, np_kernel)
        elif deal_Type == 4:
            img = close_binary(img, np_kernel)
    return img

def morphy_gray(img, deal_Type):
    """根据用户的选择，对于图像做相应的灰值形态学处理"""
    if img.shape[-1] == 3:
        pass
    q_dialog = QDialog()
    dlg = UI_Conv()
    dlg.setupUi(q_dialog)  # 继承QDialog()， 使得dialog具有show()方法
    q_dialog.show()
    if q_dialog.exec() == QDialog.Accepted:  # 提取用户交互的参数
        np_kernel = np.array(
            [[int(dlg.lineEdit1.text()), int(dlg.lineEdit2.text()), int(dlg.lineEdit3.text()), int(dlg.lineEdit4.text()), int(dlg.lineEdit5.text())],
             [int(dlg.lineEdit6.text()), int(dlg.lineEdit7.text()), int(dlg.lineEdit8.text()), int(dlg.lineEdit9.text()), int(dlg.lineEdit10.text())],
             [int(dlg.lineEdit11.text()), int(dlg.lineEdit12.text()), int(dlg.lineEdit13.text()), int(dlg.lineEdit14.text()), int(dlg.lineEdit15.text())],
             [int(dlg.lineEdit16.text()), int(dlg.lineEdit17.text()), int(dlg.lineEdit18.text()), int(dlg.lineEdit19.text()), int(dlg.lineEdit20.text())],
             [int(dlg.lineEdit21.text()), int(dlg.lineEdit22.text()), int(dlg.lineEdit23.text()), int(dlg.lineEdit24.text()), int(dlg.lineEdit25.text())]
            ])

        if deal_Type == 1:
            img = erosion_gray(img, np_kernel)
        elif deal_Type == 2:
            img = dilation_gray(img, np_kernel)
        elif deal_Type == 3:
            img = open_gray(img, np_kernel)
        elif deal_Type == 4:
            img = close_gray(img, np_kernel)
        elif deal_Type == 5:
            img = morphy_gray_edge(img, np_kernel)
    return img
# 二值形态学-腐蚀
def erosion_binary(img, np_kernel):
    """*功能 : 根据传入的图像进行二值腐蚀，默认255像素点为目标
    *注意，传入的图像必须为二值图像, kernel为结构形状, 函数：XΘS={x|S+x∈X}"""
    time1 = time.time()  # 程序计时开始
    # numpy实现 - 滑动窗口
    k_rows, k_cols = np_kernel.shape
    pad_height = k_rows // 2
    pad_width = k_cols // 2
    # 填充图像
    padded_img = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    # 使用stride_tricks来创建滑动窗口视图
    window_view = as_strided(padded_img,
                             shape=(img.shape[0], img.shape[1], k_rows, k_cols),
                             strides=padded_img.strides * 2)
    # 应用腐蚀操作
    mask = np_kernel == 1
    eroded = np.all(window_view[..., mask] == 255, axis=-1)
    new_img = eroded.astype(np.uint8) * 255

    time2 = time.time()  # 程序计时结束
    print("numpy二值腐蚀运算处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # # opencv图像腐蚀过程
    # new_img = cv.erode(img, np_kernel.astype(np.uint8))   # 直接用opencv的腐蚀函数
    # time2 = time.time()  # 程序计时结束
    # print("opencv二值腐蚀运算处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    
    return new_img
# 二值形态学-膨胀
def dilation_binary(img, np_kernel):
    """*功能 : 根据传入的图像进行二值膨胀，默认255像素点为目标
    *注意，传入的图像必须为二值图像, kernel为结构形状, 这里用的是腐蚀函数的对偶运算：
    X⊕S=∪{S+x|x∈X}=(X^c Θ S^v)^c"""
    time1 = time.time()  # 程序计时开始
    # NumPy实现 - 滑动窗口
    k_rows, k_cols = np_kernel.shape
    pad_height = k_rows // 2
    pad_width = k_cols // 2
    # 填充图像
    padded_img = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    # 使用stride_tricks来创建滑动窗口视图
    window_view = as_strided(padded_img,
                             shape=(img.shape[0], img.shape[1], k_rows, k_cols),
                             strides=padded_img.strides * 2)
    # 在应用膨胀操作之前，先反射结构元素  对于非对称的结构元素，使用结构元素的反射会使膨胀后的结果总的位置和形状保持一致
    np_kernel_reflected = np.flip(np_kernel)
    mask = np_kernel_reflected == 1
    dilated = np.any(window_view[..., mask] == 255, axis=-1)
    new_img = dilated.astype(np.uint8) * 255

    time2 = time.time()  # 程序计时结束
    print("numpy二值膨胀运算处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # # OpenCV图像膨胀过程
    # time1 = time.time()  # 程序计时开始
    # new_img = cv.dilate(img, np_kernel.astype(np.uint8))   # 直接用OpenCV的膨胀函数
    # time2 = time.time()  # 程序计时结束
    # print("opencv二值膨胀运算处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    
    return new_img
# 二值形态学-开运算
def open_binary(img, np_kernel):
    """*功能 : 根据传入的图像行二值开，默认255像素点为目标(ww)
    *注意，传入的图像必须为二值图像，二值开函数：X○S=(XΘS)⊕S"""
    time1 = time.time()  # 程序计时开始
    # 手写图像二值开运算过程
    new_img = erosion_binary(img, np_kernel)  # 先腐蚀
    new_img = dilation_binary(new_img, np_kernel)  # 再膨胀
    time2 = time.time()  # 程序计时结束
    print("手写二值开运算程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # # opencv图像二值开运算过程
    # new_img = cv.morphologyEx(img, cv.MORPH_OPEN, np_kernel.astype(np.uint8))
    # time2 = time.time()  # 程序计时结束
    # print("opencv二值开运算程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img
# 二值形态学-闭运算
def close_binary(img, np_kernel):
    """*功能 : 根据传入的图像进行二值闭，默认255像素点为目标(ww)
        *注意，传入的图像必须为二值图像，二值闭函数：X·S=(X⊕S)ΘS"""
    time1 = time.time()  # 程序计时开始
    # 手写图像二值闭运算过程
    new_img = dilation_binary(img, np_kernel)  # 先膨胀
    new_img = erosion_binary(new_img, np_kernel) # 再腐蚀
    time2 = time.time()  # 程序计时结束
    print("手写二闭运算程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # # opencv图像二值闭运算过程
    # new_img = cv.morphologyEx(img, cv.MORPH_CLOSE, np_kernel.astype(np.uint8))
    # time2 = time.time()  # 程序计时结束
    # print("opencv二值闭运算程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img
# 灰值形态学-快速细化（骨架提取）
def fast_thin(img, deal_Type):
    """*功能 : 根据传入的图像进行快速形态学细化，默认0像素点为背景
    deal_Type: 0 - skeletonize, 1 - medial_axis"""
    time1 = time.time()  # 程序计时开始
    
    # 确保输入图像是二值图像
    img = 255 - img # 先反转图像  因为目标是白色，对于黑色目标的图像需要先反转
    if img.dtype != bool:
        _, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        img = img > 0
    
    if deal_Type == 0:
        # 使用skeletonize方法
        skeleton = morphology.skeletonize(img)
        new_img = (skeleton * 255).astype(np.uint8)
        method = "skeletonize"
    elif deal_Type == 1:
        # 使用medial_axis方法
        skeleton, distance = morphology.medial_axis(img, return_distance=True)
        # 使用距离信息来增强骨架
        new_img = (skeleton * 255).astype(np.uint8)
        method = "medial_axis"

    time2 = time.time()  # 程序计时结束
    print(f"skimage {method} 细化算法处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    
    return new_img
# 灰值形态学-腐蚀
def erosion_gray(img, np_kernel):
    """*功能 : 根据传入的图像进行灰值腐蚀*注意，传入的图像必须为灰值图像,
       kernel为结构形状, 函数：(fΘb)(s,t)=min{f(s+x, t+y)}"""
    time1 = time.time()  # 程序计时开始
    # NumPy实现
    k_rows, k_cols = np_kernel.shape
    pad_height = k_rows // 2
    pad_width = k_cols // 2
    # 填充图像
    padded_img = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=255)
    # 使用stride_tricks来创建滑动窗口视图
    window_view = as_strided(padded_img,
                             shape=(img.shape[0], img.shape[1], k_rows, k_cols),
                             strides=padded_img.strides * 2)
    # 应用腐蚀操作
    mask = np_kernel == 1
    eroded = np.min(window_view[..., mask], axis=-1)
    new_img = eroded.astype(np.uint8)
    time2 = time.time()  # 程序计时结束
    print("NumPy灰值腐蚀运算处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # # OpenCV实现
    # time1 = time.time()  # 重新开始计时
    # new_img = cv.erode(img, np_kernel.astype(np.uint8))
    # time2 = time.time()  # 程序计时结束
    # print("OpenCV灰值腐蚀运算处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    
    return new_img
# 灰值形态学-膨胀
def dilation_gray(img, np_kernel):
    """*功能 : 根据传入的图像进行灰值膨胀*注意，传入的图像必须为灰值图像,
       kernel为结构形状, 函数：(f⊕b)(s,t)=max{f(s-x, t-y) + b(x,y)}"""
    time1 = time.time()  # 程序计时开始
    # NumPy实现
    k_rows, k_cols = np_kernel.shape
    pad_height = k_rows // 2
    pad_width = k_cols // 2
    # 填充图像
    padded_img = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    # 使用stride_tricks来创建滑动窗口视图
    window_view = as_strided(padded_img,
                             shape=(img.shape[0], img.shape[1], k_rows, k_cols),
                             strides=padded_img.strides * 2)
    # 应用膨胀操作，这里反转了结构元素
    np_kernel_reflected = np.flip(np_kernel)
    mask = np_kernel_reflected == 1
    dilated = np.max(window_view[..., mask], axis=-1)
    new_img = dilated.astype(np.uint8)
    time2 = time.time()  # 程序计时结束
    print("NumPy灰值膨胀运算处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # # OpenCV实现
    # time1 = time.time()  # 重新开始计时
    # new_img = cv.dilate(img, np_kernel.astype(np.uint8))
    # time2 = time.time()  # 程序计时结束
    # print("OpenCV灰值膨胀运算处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    return new_img
# 灰值形态学-开运算     # 思考：灰值形态学的开闭运算都能对图像产生什么效果
def open_gray(img, np_kernel):
    """*功能 : 根据传入的图像进行灰值开，灰值开函数：X○S=(XΘS)⊕S"""
    time1 = time.time()  # 程序计时开始
    # 手写图像灰值开运算过程
    new_img = erosion_gray(img, np_kernel) # 先腐蚀
    new_img = dilation_gray(new_img, np_kernel)  # 再膨胀
    time2 = time.time()  # 程序计时结束
    print("手写灰值开运算程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # # opencv图像灰值开运算过程
    # new_img = cv.morphologyEx(img, cv.MORPH_OPEN, np_kernel.astype(np.uint8))
    # time2 = time.time()  # 程序计时结束
    # print("opencv灰值开运算程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img
# 灰值形态学-闭运算
def close_gray(img, np_kernel):
    """*功能 : 根据传入的图像进行灰值闭，灰值闭运算函数：X·S=(X⊕S)ΘS"""
    time1 = time.time()  # 程序计时开始
    # 手写图像灰值闭运算过程
    new_img = dilation_gray(img, np_kernel)  # 先膨胀
    new_img = erosion_gray(new_img, np_kernel) # 再腐蚀
    time2 = time.time()  # 程序计时结束
    print("手写灰值闭运算程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # # opencv图像灰值闭运算过程
    # new_img = cv.morphologyEx(img, cv.MORPH_CLOSE, np_kernel.astype(np.uint8))
    # time2 = time.time()  # 程序计时结束
    # print("opencv灰值闭运算程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img
# 灰值形态学-梯度求取（边缘）  思考：为什么结构元素变大，边缘会更亮更清晰，但是整体图像会变得更模糊
def morphy_gray_edge(img, np_kernel):
    """*功能 : 根据传入的图像进行灰值边缘求取，函数为：g(X)=(X⊕S)-(XΘS)"""
    time1 = time.time()  # 程序计时开始
    # 手写图像灰值形态学边缘求取过程
    ero_img = erosion_gray(img, np_kernel)  # 腐蚀
    dil_img = dilation_gray(img, np_kernel)  # 膨胀
    new_img = cv.absdiff(dil_img, ero_img)   # 取绝对差
    time2 = time.time()  # 程序计时结束
    print("手写灰值形态学边缘程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # # opencv图像灰值形态学边缘求取过程
    # ero_img = cv.erode(img, np_kernel.astype(np.uint8))  # 腐蚀
    # dil_img = cv.dilate(img, np_kernel.astype(np.uint8))  # 膨胀
    # new_img = cv.absdiff(dil_img, ero_img)  # 取绝对差
    # time2 = time.time()  # 程序计时结束  
    # print("opencv灰值形态学边缘程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img
