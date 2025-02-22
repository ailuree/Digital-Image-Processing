import cv2 as cv
import time

def color_deal(img, deal_Type, ui, method):
    """根据用户的选择，对于图像做相应的处理"""
    # 取原始图片
    if deal_Type == 1:
        pass
    # 取反色处理
    if deal_Type == 2:
        if method == 0:
            # opencv求反色函数
            time1 = time.time()
            img = cv.bitwise_not(img)
            time2 = time.time()
            ui.update_debug_info(f"opencv取反色遍历时间：{(time2 - time1) * 1000} ms")
        else:

                        # 手写图像取反色处理
            time1 = time.time()
            # 获取图像的宽高  
            width = img.shape[1]
            height = img.shape[0]
            # 判断图像通道数 
            if img.shape[-1] == 3:         # 彩色图像
                for row in range(height):  # 遍历每一行
                    for col in range(width):  # 遍历每一列
                        for channel in range(3):  # 遍历每个通道（三个通道分别是BGR）
                            img[row][col][channel] = 255 - img[row][col][channel]       # 取反色处理 每一个像素点的每个通道都要取反
            else:                          # 灰度图像
                for row in range(height):  # 遍历每一行
                    for col in range(width):  # 遍历每一列
                        img[row][col] = 255 - img[row][col]

            time2 = time.time()
            ui.update_debug_info(f"数据检索取反色遍历时间：{(time2 - time1) * 1000} ms")

    elif deal_Type == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    elif deal_Type == 4:
        img = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    elif deal_Type == 5:
        img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    elif deal_Type == 6:
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    else:
        pass
    return img