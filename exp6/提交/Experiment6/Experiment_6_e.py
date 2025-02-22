import cv2 as cv
import numpy as np
from skimage import morphology
import math
from numba import jit
from matplotlib import pyplot as plt
from PyQt5.QtWidgets import *
from Experiment5.Conv5_UI import Ui_Dialog as UI_Conv
import time
import traceback

def load_captcha(filename):
    """载入验证码图像"""
    try:
        img = cv.imread(filename)
        if img is None:
            raise Exception("无法读取图像文件")
        return img
    except Exception as e:
        print(f"载入图像失败: {str(e)}")
        return None

def denoise_captcha(img):
    """验证码图像去噪"""
    if img is None:
        return None
    
    try:
        # 转换到HSV颜色空间
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv)
        
        # 定位噪点
        bw1 = (h > 0.16 * 180) & (h < 0.30 * 180)
        bw2 = (s > 0.65 * 255) & (s < 0.80 * 255)
        bw = bw1 & bw2
        
        # 过滤噪点
        img_denoised = img.copy()
        img_denoised[bw] = [255, 255, 255]
        
        # 转换为灰度图
        gray = cv.cvtColor(img_denoised, cv.COLOR_BGR2GRAY)
        
        # 二值化
        _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        
        # 查找所有轮廓
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # 创建掩码
        mask = np.zeros_like(binary)
        
        # 过滤轮廓
        for contour in contours:
            # 计算轮廓面积
            area = cv.contourArea(contour)
            
            # 计算轮廓周长
            perimeter = cv.arcLength(contour, True)
            
            # 计算轮廓的边界框
            x, y, w, h = cv.boundingRect(contour)
            
            # 计算长宽比
            aspect_ratio = float(w)/h if h > 0 else 0
            
            # 设置过滤条件
            is_valid = (
                80 < area < 8000 and           # 面积范围
                perimeter > 20 and             # 最小周长
                0.15 < aspect_ratio < 3.0 and  # 长宽比范围
                10 < h < 150 and              # 高度范围
                5 < w < 150                    # 宽度范围
            )
            
            if is_valid:
                # 在掩码上绘制有效轮廓
                cv.drawContours(mask, [contour], -1, 255, -1)
        
        # 使用掩码获取去噪后的图像
        denoised = cv.bitwise_and(binary, mask)
        
        # 轻微的形态学操作清理边缘
        kernel = np.ones((2,2), np.uint8)
        denoised = cv.morphologyEx(denoised, cv.MORPH_OPEN, kernel, iterations=1)
        
        return denoised
        
    except Exception as e:
        print(f"图像去噪失败: {str(e)}")
        return None

def normalize_captcha(img, target_size=(32, 32)):
    """
    验证码归一化，使用Hu矩实现姿态和尺度归一化
    :param img: 输入图像(二值化后的)
    :param target_size: 目标大小，默认32x32
    :return: 归一化后的图像和中间结果
    """
    if img is None:
        return None, None
    
    try:
        # 确保图像是二值化的
        if len(img.shape) == 3:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        else:
            binary = img.copy()
        
        # 查找轮廓
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # 过滤太小的轮廓
        contours = [cnt for cnt in contours if cv.contourArea(cnt) > 100]
        
        # 按x坐标排序轮廓
        contours = sorted(contours, key=lambda c: cv.boundingRect(c)[0])
        
        # 创建结果图像
        char_size = 64  # 单个字符的大小
        padding = 10    # 字符间距
        result_height = char_size
        result_width = char_size * len(contours) + padding * (len(contours) - 1)
        result_img = np.zeros((result_height, result_width), dtype=np.uint8)
        
        # 创建调试图像
        debug_img = cv.cvtColor(binary.copy(), cv.COLOR_GRAY2BGR)
        
        # 处理每个轮廓（每个字符）
        for i, contour in enumerate(contours):
            # 获取字符区域
            x, y, w, h = cv.boundingRect(contour)
            char_img = binary[y:y+h, x:x+w]
            
            # 计算Hu矩
            moments = cv.moments(contour)
            if moments['m00'] != 0:
                # 计算重心
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
                
                # 计算主方向
                if moments['mu20'] != moments['mu02']:
                    theta = 0.5 * math.atan2(2*moments['mu11'], 
                                           moments['mu20'] - moments['mu02'])
                    angle = math.degrees(theta)
                    
                    # 根据宽高比调整角度
                    if h > w:
                        angle += 90
                        
                    # 旋转图像
                    M = cv.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                    char_img = cv.warpAffine(char_img, M, (w, h))
            
            # 添加边距以避免裁剪
            border_size = 5
            char_img = cv.copyMakeBorder(char_img, border_size, border_size, 
                                       border_size, border_size, 
                                       cv.BORDER_CONSTANT, value=0)
            
            # 缩放到固定大小，保持纵横比
            h, w = char_img.shape
            scale = min(char_size/w, char_size/h) * 0.8  # 缩小一点以留出边距
            new_w = int(w * scale)
            new_h = int(h * scale)
            normalized = cv.resize(char_img, (new_w, new_h))
            
            # 将字符放在中心位置
            char_region = np.zeros((char_size, char_size), dtype=np.uint8)
            y_offset = (char_size - new_h) // 2
            x_offset = (char_size - new_w) // 2
            char_region[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = normalized
            
            # 在结果图像中显示
            start_x = i * (char_size + padding)
            result_img[0:char_size, start_x:start_x+char_size] = char_region
            
            # 在调试图像上标记
            cv.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(debug_img, f"{i+1}", (x, y-5), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return result_img, debug_img
        
    except Exception as e:
        print(f"图像归一化失败: {str(e)}")
        traceback.print_exc()
        return None, None