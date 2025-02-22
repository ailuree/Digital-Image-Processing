import os
import numpy as np
import cv2

def get_all_files(dir_name):
    """获取目录下所有文件的路径"""
    file_list = []
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def compute_invariant_moments(img):
    """计算图像的不变矩特征"""
    # 确保图像是浮点型
    img = img.astype(np.float32)
    
    # 计算Hu矩
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments)
    
    # 对Hu矩进行对数处理以减小数值差异
    for i in range(7):
        if hu_moments[i] != 0:
            hu_moments[i] = -1 * np.sign(hu_moments[i]) * np.log10(abs(hu_moments[i]))
    
    return hu_moments.flatten()