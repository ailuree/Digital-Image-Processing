import cv2 as cv
import numpy as np
import time
import math as m

#---------图像缩放---------
# numpy手写缩放程序 最近邻插值
# def img_zoom(img, zm):
#     """对于传入的图像进行缩放操作, zm: 缩放因子"""
#     time1 = time.time()
#     h, w = img.shape[:2]
#     new_h, new_w = int(h * zm), int(w * zm)
#     # 使用最近邻插值
#     x_ratio = w / new_w
#     y_ratio = h / new_h
#     # 使用numpy的高级索引功能
#     x = (np.arange(new_w) * x_ratio).astype(int)
#     y = (np.arange(new_h) * y_ratio).astype(int)

#     # 使用numpy的高级索引功能，避免meshgrid
#     new_img = img[y[:, np.newaxis], x]
#     # 调整输出图像大小
#     if zm > 1:
#         new_img = new_img[:h, :w]
#     else:
#         new_img = np.pad(new_img, ((0, h - new_h), (0, w - new_w), (0, 0)), 
#                          mode='constant', constant_values=0)
#     time2 = time.time()
#     print("numpy-最近邻插值缩放程序处理时间： %.3f毫秒" % ((time2 - time1) * 1000))
#     return new_img
#---------图像缩放---------
#numpy手写缩放程序 双线性插值
# def img_zoom(img, zm):
#     """对于传入的图像进行缩放操作, zm: 缩放因子"""
#     time1 = time.time()
#     rows, cols = img.shape[:2]
#     new_rows, new_cols = int(rows * zm), int(cols * zm)
#     # 生成缩放后的坐标网格
#     x = np.linspace(0, rows - 1, new_rows)
#     y = np.linspace(0, cols - 1, new_cols)
#     x, y = np.meshgrid(x, y, indexing='ij')
#     # 计算原图中的对应坐标
#     x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
#     x1, y1 = np.clip(x0 + 1, 0, rows - 1), np.clip(y0 + 1, 0, cols - 1)
#     # 计算插值权重
#     wx, wy = x - x0, y - y0
#     # 使用双线性插值计算颜色值
#     new_img = ((1 - wx)[:, :, None] * (1 - wy)[:, :, None] * img[x0, y0] +wx[:, :, None] * (1 - wy)[:, :, None] * img[x1, y0] +(1 - wx)[:, :, None] * wy[:, :, None] * img[x0, y1] +wx[:, :, None] * wy[:, :, None] * img[x1, y1]).astype(np.uint8)
#     # 调整输出图像大小
#     if zm > 1:
#         new_img = new_img[:rows, :cols]
#     else:
#         new_img = np.pad(new_img, ((0, rows - new_rows), (0, cols - new_cols), (0, 0)), mode='constant', constant_values=0)
    
#     time2 = time.time()
#     print("numpy-双线性插值缩放程序处理时间： %.3f毫秒" % ((time2 - time1) * 1000))
#     return new_img
#---------图像缩放---------
# opencv缩放程序
# def img_zoom(img, zm):

#    time1 = time.time()
#    rows, cols = img.shape[:2]  #获取宽和高
#    new_img = np.zeros(img.shape, dtype=np.uint8) #新建同原图大小一致的空图像

#    # opencv图像缩放
#    img = cv.resize(img, (int(cols*zm), int(rows*zm)), interpolation=cv.INTER_CUBIC)
#    if zm>1: # 原图像大小显示
#       new_img = img[0:cols, 0:rows]
#    else:
#       new_img[0:img.shape[0], 0:img.shape[1]] = img
#    time2 = time.time()
#    print("opencv缩放程序处理时间：%.3f毫秒" %((time2 - time1) * 1000))

#    return new_img

# #---------图像平移---------
# # 手写图像平移程序
# def img_translation(img, trans):
#     """对于传入的图像进行左右平移, *trans:平移参数"""
#     time1 = time.time()
#     rows, cols = img.shape[:2]
#     new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像

#     # 手写图像平移代码
#     for i in range(rows):
#         for j in range(cols):
#             new_j = j + trans
#             if 0 <= new_j < cols:
#                 new_img[i, new_j] = img[i, j]

#     time2 = time.time()
#     print("手写图像平移程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
#     return new_img
# #---------图像平移---------
# numpy图像平移程序
# def img_translation(img, trans):
#     """对于传入的图像进行左右平移, *trans:平移参数"""
#     time1 = time.time()
#     rows, cols = img.shape[:2]
#     new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
#     # 使用NumPy进行图像平移
#     if trans > 0:
#         new_img[:, trans:] = img[:, :cols-trans]
#     elif trans < 0:
#         new_img[:, :cols+trans] = img[:, -trans:]
#     else:
#         new_img = img.copy()
#     time2 = time.time()
#     print("手写图像平移程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
#     return new_img
# #---------图像平移---------
# opencv图像平移程序
# def img_translation(img, trans):
#     """对于传入的图像进行左右平移, *trans:平移参数"""
#     time1 = time.time()
#     # 获取图像的高度和宽度
#     rows, cols = img.shape[:2]
#     # 定义平移矩阵
#     M = np.float32([[1, 0, trans], [0, 1, 0]])
#     # 使用cv.warpAffine进行图像平移
#     new_img = cv.warpAffine(img, M, (cols, rows))
#     time2 = time.time()
#     print("opencv图像平移时间：%.3f毫秒" % ((time2 - time1) * 1000))
#     return new_img

#---------图像镜像---------
# 手写函数实现图像镜像
# def img_imgMirror(img):
#    """对于传入的图像进行取镜像操作"""
#    time1 = time.time()
#    rows, cols = img.shape[:2]  # 获取宽和高
#    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
#    for col in range(cols):  # 遍历每一列
#       new_img[:, col] = img[:, cols-1-col]
#    time2 = time.time()
#    print("手写镜面变换程序处理时间：%.3f毫秒" %((time2 - time1) * 1000))
#    return new_img
#---------图像镜像---------
# numpy函数实现图像镜像
# def img_imgMirror(img):
#     """对于传入的图像进行取镜像操作"""
#     time1 = time.time()
#     rows, cols = img.shape[:2]  # 获取宽和高
    
#     # 使用NumPy进行镜像变换
#     new_img = img[:, ::-1]

#     time2 = time.time()
#     print("numpy实现镜像处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
#     return new_img
#---------图像镜像---------
# opencv函数实现图像镜像
# def img_imgMirror(img):
#     """对于传入的图像进行取镜像操作"""
#     time1 = time.time()
    
#     # 使用OpenCV进行镜像变换
#     new_img = cv.flip(img, 1)

#     time2 = time.time()
#     print("opencv镜面变换程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
#     return new_img

#---------图像旋转---------
# numpy函数实现图像旋转
def img_rotation(img, rot):
    """
    使用优化的NumPy实现对传入的图像进行旋转，可以绕图像中心旋转
    """
    rows, cols = img.shape[:2]
    rot_rad = np.radians(rot)
    time1 = time.time()
    # 计算旋转矩阵
    cos_val, sin_val = np.cos(rot_rad), np.sin(rot_rad)
    M = np.array([[cos_val, -sin_val, 0], [sin_val, cos_val, 0], [0, 0, 1]])
    
    # 计算新尺寸并调整旋转中心
    new_rows, new_cols = int(np.round(rows * np.abs(cos_val) + cols * np.abs(sin_val))), int(np.round(rows * np.abs(sin_val) + cols * np.abs(cos_val)))
    t = np.array([[1, 0, -cols/2], [0, 1, -rows/2], [0, 0, 1]])
    t_inv = np.array([[1, 0, new_cols/2], [0, 1, new_rows/2], [0, 0, 1]])
    M = t_inv @ M @ t

    # 创建网格坐标并应用逆变换
    y, x = np.indices((new_rows, new_cols))
    coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()])
    x_transformed, y_transformed = np.linalg.inv(M).dot(coords)[:2, :]
    x_transformed = x_transformed.reshape(new_rows, new_cols)
    y_transformed = y_transformed.reshape(new_rows, new_cols)

    # 使用双线性插值进行采样
    x0, y0 = np.floor(x_transformed).astype(int), np.floor(y_transformed).astype(int)
    x1, y1 = x0 + 1, y0 + 1

    x0 = np.clip(x0, 0, cols-1)
    x1 = np.clip(x1, 0, cols-1)
    y0 = np.clip(y0, 0, rows-1)
    y1 = np.clip(y1, 0, rows-1)

    wa = (x1 - x_transformed) * (y1 - y_transformed)
    wb = (x1 - x_transformed) * (y_transformed - y0)
    wc = (x_transformed - x0) * (y1 - y_transformed)
    wd = (x_transformed - x0) * (y_transformed - y0)

    new_img = (wa[:, :, np.newaxis] * img[y0, x0] +
               wb[:, :, np.newaxis] * img[y1, x0] +
               wc[:, :, np.newaxis] * img[y0, x1] +
               wd[:, :, np.newaxis] * img[y1, x1]).astype(img.dtype)

    time2 = time.time()
    print("优化后numpy旋转程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img

# opencv函数实现图像旋转
# def img_rotation(img, rot):
#    """对于传入的图像进行旋转，可以绕任一点旋转, *rot:旋转角度"""
#    rows, cols = img.shape[:2]  # 获取宽和高
#    b, a = rows / 2, cols / 2  # 设置旋转点位置
#    h, w = rows / 2, cols / 2  # 图像高宽的一半
#    time1 = time.time()
#    # opencv绕任一点旋转代码
#    # 第一个参数是旋转中心，第二个参数是旋转角度，第三个因子是旋转后的缩放因子
#    M = cv.getRotationMatrix2D((a, b), rot, 1)
#    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
#    new_cols = int((rows * sin) + (cols * cos))
#    new_rows = int((rows * cos) + (cols * sin))
#    M[0, 2] += (new_cols / 2) - w
#    M[1, 2] += (new_rows / 2) - h
#    new_img = cv.warpAffine(img, M, (new_cols, new_rows))  # 第三个参数是输出图像的尺寸中心，图像的宽和高
#    time2 = time.time()
#    print("opencv旋转程序处理时间：%.3f毫秒" %((time2 - time1) * 1000))
#    return new_img



def onmouse_pick_points(event, x, y, flags, l_ImgPot):
   """card_correction的鼠标回调函数, """
   if event == cv.EVENT_LBUTTONDOWN:
      print('x = %d, y = %d' % (x, y))
      l_ImgPot[2].append((x, y))
      cv.drawMarker(l_ImgPot[1], (x, y), (0, 0, 255))
   if event == cv.EVENT_RBUTTONDOWN:
      l_ImgPot[1] = l_ImgPot[0].copy() # 将没有画十字的原图重新赋值给显示图像
      if len(l_ImgPot[2]) != 0:
         l_ImgPot[2].pop() # 将最后一次绘制的标记清除
         for i in range(len(l_ImgPot[2])): # 重新绘制全部标记
            cv.drawMarker(l_ImgPot[1], l_ImgPot[2][i], (0, 0, 255))

def card_correction(img):
   """对于传入的图像进行鼠标交互，选择四个顶点进行名片矫正"""
   l_ImgPot = [img, img.copy(), []] # 记录画标识的图像和标识点  [0]原图 【1】处理后图
   cv.namedWindow('card', cv.WINDOW_AUTOSIZE)
   cv.setMouseCallback('card', onmouse_pick_points, l_ImgPot)
   while True:
      cv.imshow('card', l_ImgPot[1])
      key = cv.waitKey(30)
      if key == 27:  # ESC
         break
   cv.destroyAllWindows()
   time1 = time.time()

    # 透视变换矫正名片核心代码
   if len(l_ImgPot[2]) != 4:
       print("请选择4个角点")
       return img
   # 设置输出图像的大小
   width, height = 900, 540
   # 源图像中的四个角点
   pts1 = np.float32(l_ImgPot[2])
   # 目标图像中的四个角点
   pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
   # 计算透视变换矩阵
   matrix = cv.getPerspectiveTransform(pts1, pts2)
   # 进行透视变换
   new_img = cv.warpPerspective(img, matrix, (width, height))

   time2 = time.time()
   print("opencv名片矫正处理时间：%.3f毫秒" %((time2 - time1) * 1000))

   return new_img