import cv2 as cv
import numpy as np
import time
import math as m

def img_zoom(img, zm):
   """对于传入的图像进行缩放操作, *zm:缩放因子"""
   time1 = time.time()
   rows, cols = img.shape[:2]  #获取宽和高
   new_img = np.zeros(img.shape, dtype=np.uint8) #新建同原图大小一致的空图像
   # # 手写图像缩放代码
   ...
   # time2 = time.time()
   # print("手写缩放程序处理时间： %.3f毫秒" %((time2 - time1) * 1000))

   # opencv图像缩放
   img = cv.resize(img, (int(cols*zm), int(rows*zm)), interpolation=cv.INTER_CUBIC)
   if zm>1: # 原图像大小显示
      new_img = img[0:cols, 0:rows]
   else:
      new_img[0:img.shape[0], 0:img.shape[1]] = img
   time2 = time.time()
   print("opencv缩放程序处理时间：%.3f毫秒" %((time2 - time1) * 1000))
   return new_img

def img_translation(img, trans):
   """对于传入的图像进行左右, *trans:平移参数"""
   time1 = time.time()
   new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
   # # 手写图像平移代码
   ...

   time2 = time.time()
   print("手写图像平移程序处理时间：%.3f毫秒" %((time2 - time1) * 1000))
   return new_img

def img_imgMirror(img):
   """对于传入的图像进行左右, *trans:平移参数"""
   time1 = time.time()
   rows, cols = img.shape[:2]  # 获取宽和高
   new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
   for col in range(cols):  # 遍历每一列
      new_img[:, col] = img[:, cols-1-col]
   time2 = time.time()
   print("手写镜面变换程序处理时间：%.3f毫秒" %((time2 - time1) * 1000))
   return new_img

def img_rotation(img, rot):
   """对于传入的图像进行旋转，可以绕任一点旋转, *rot:旋转角度"""
   rows, cols = img.shape[:2]  # 获取宽和高
   b, a = rows / 2, cols / 2  # 设置旋转点位置
   h, w = rows / 2, cols / 2  # 图像高宽的一半
   time1 = time.time()

   # # 手写图像绕任一点旋转代码
   ...
   # time2 = time.time()
   # print("手写旋转程序处理时间：%.3f毫秒" %((time2 - time1) * 1000))

   # opencv绕任一点旋转代码
   # 第一个参数是旋转中心，第二个参数是旋转角度，第三个因子是旋转后的缩放因子
   M = cv.getRotationMatrix2D((a, b), rot, 1)
   cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
   new_cols = int((rows * sin) + (cols * cos))
   new_rows = int((rows * cos) + (cols * sin))
   M[0, 2] += (new_cols / 2) - w
   M[1, 2] += (new_rows / 2) - h
   new_img = cv.warpAffine(img, M, (new_cols, new_rows))  # 第三个参数是输出图像的尺寸中心，图像的宽和高
   time2 = time.time()
   print("opencv旋转程序处理时间：%.3f毫秒" %((time2 - time1) * 1000))
   return new_img

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

   # # 透视变换矫正名片核心代码(可opencv)
   ...
   
   time2 = time.time()
   print("opencv名片矫正处理时间：%.3f毫秒" %((time2 - time1) * 1000))

   return new_img