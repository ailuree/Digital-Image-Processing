import cv2 as cv
import numpy as np
import time
import math as m

#---------图像缩放---------
#numpy手写缩放程序 最近邻插值
# def img_zoom(img, zm):
#     """对于传入的图像进行缩放操作, zm: 缩放因子"""
#     time1 = time.time()
#     h, w = img.shape[:2]
#     new_h, new_w = int(h * zm), int(w * zm)
#     # 使用最近邻插值  计算缩放比例
#     x_ratio = w / new_w
#     y_ratio = h / new_h
#     # 使用numpy的高级索引功能得到新图像像素在原图的水平位置和垂直位置   np.arange()生成等差数组 生成0-new_w-1的数组
#     x = (np.arange(new_w) * x_ratio).astype(int)
#     y = (np.arange(new_h) * y_ratio).astype(int)  #将缩放后的图像的坐标映射到原图像的坐标值转变为整数  即最近邻

#     # 给y增加一个维度变成(new_h, 1) x是(new_w,) 利用广播机制生成(new_h, new_w)
#     new_img = img[y[:, np.newaxis], x]  
#     # 调整输出图像大小
#     if zm > 1:
#         new_img = new_img[:h, :w]   #如果缩放比例大于1，截取部分图像 
#     else:
#         new_img = np.pad(new_img, ((0, h - new_h), (0, w - new_w), (0, 0)),    # 把图像空的部分填充成黑色
#                          mode='constant', constant_values=0)
#     time2 = time.time()
#     print("numpy-最近邻插值缩放程序处理时间： %.3f毫秒" % ((time2 - time1) * 1000))
#     return new_img
#---------图像缩放---------
#numpy手写缩放程序 双线性插值
def img_zoom(img, zm):
    """对于传入的图像进行缩放操作, zm: 缩放因子"""
    time1 = time.time()
    rows, cols = img.shape[:2]   # 原图像的高和宽
    new_rows, new_cols = int(rows * zm), int(cols * zm)  # 新图像的高和宽
    # 生成缩放后的坐标网格
    x = np.linspace(0, rows - 1, new_rows)   # 把原图等间距垂直分成new_rows份  （分成多行）
    y = np.linspace(0, cols - 1, new_cols)   # 把原图等间距水平分成new_cols份  （分成多列）
    x, y = np.meshgrid(x, y, indexing='ij')  # 行列索引生成网格坐标
    # 计算原图中的对应坐标
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)   # 向下取整  所以是新图像中的点对应原图中左上角的点
    x1, y1 = np.clip(x0 + 1, 0, rows - 1), np.clip(y0 + 1, 0, cols - 1)   # 不超过图像边界 右下角的点
    # 计算插值权重 该点与左上角点的距离
    wx, wy = x - x0, y - y0
    # 使用双线性插值计算颜色值  分别是左上角、右上角、左下角、右下角的颜色贡献
    new_img = ((1 - wx)[:, :, None] * (1 - wy)[:, :, None] * img[x0, y0] + wx[:, :, None] * (1 - wy)[:, :, None] * img[x1, y0] + (1 - wx)[:, :, None] * wy[:, :, None] * img[x0, y1] + wx[:, :, None] * wy[:, :, None] * img[x1, y1]).astype(np.uint8)
    # 调整输出图像大小
    if zm > 1:
        new_img = new_img[:rows, :cols]
    else:
        new_img = np.pad(new_img, ((0, rows - new_rows), (0, cols - new_cols), (0, 0)), mode='constant', constant_values=0)
    
    time2 = time.time()
    print("numpy-双线性插值缩放程序处理时间： %.3f毫秒" % ((time2 - time1) * 1000))
    return new_img
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
#             new_j = j + trans  # 平移后的列坐标
#             if 0 <= new_j < cols:  # 判断是否越界
#                 new_img[i, new_j] = img[i, j]  # 越界时不处理

#     time2 = time.time()
#     print("手写图像平移程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
#     return new_img
# #---------图像平移---------
# numpy图像平移程序
def img_translation(img, trans):
    """对于传入的图像进行左右平移, *trans:平移参数"""
    time1 = time.time()
    rows, cols = img.shape[:2]
    new_img = np.zeros(img.shape, dtype=np.uint8)  # 新建同原图大小一致的空图像
    # 使用NumPy进行图像平移
    if trans > 0:
        new_img[:, trans:] = img[:, :cols-trans]  # 右移   把原图像的0到cols-trans列赋值给新图像的trans到最后列
    elif trans < 0:
        new_img[:, :cols+trans] = img[:, -trans:] # 左移   把原图像的-trans到最后列赋值给新图像的0到cols+trans列
    else:
        new_img = img.copy()
    time2 = time.time()
    print("手写图像平移程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img
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
#       new_img[:, col] = img[:, cols-1-col]  # :,表示所有行，cols-1-col表示镜像列
#    time2 = time.time()
#    print("手写镜面变换程序处理时间：%.3f毫秒" %((time2 - time1) * 1000))
#    return new_img
#---------图像镜像---------
# numpy函数实现图像镜像
def img_imgMirror(img):
    """对于传入的图像进行取镜像操作"""
    time1 = time.time()
    rows, cols = img.shape[:2]  # 获取宽和高
    
    # 使用NumPy进行镜像变换
    new_img = img[:, ::-1]   # 选中所有行，然后逆序选中所有列

    time2 = time.time()
    print("numpy实现镜像处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
    return new_img
#---------图像镜像---------
# opencv函数实现图像镜像
# def img_imgMirror(img):
#     """对于传入的图像进行取镜像操作"""
#     time1 = time.time()
    
#     # 使用OpenCV进行镜像变换
#     new_img = cv.flip(img, 1)  # 1表示水平镜像，0表示垂直镜像

#     time2 = time.time()
#     print("opencv镜面变换程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
#     return new_img

#---------图像旋转---------
# numpy函数实现图像旋转
def img_rotation(img, rot):
    """
    使用优化的NumPy实现对传入的图像进行旋转，可以绕图像任意一点旋转
    """
    rows, cols = img.shape[:2]   # 获取原图宽和高
    rot_rad = np.radians(rot)    # 将角度转换为弧度
    time1 = time.time()
    
    # 计算旋转矩阵
    cos_val, sin_val = np.cos(rot_rad), np.sin(rot_rad)
    M = np.array([[cos_val, -sin_val, 0], [sin_val, cos_val, 0], [0, 0, 1]])  # M就是旋转矩阵
    
    # 计算新尺寸并调整旋转中心
    new_rows, new_cols = int(np.round(rows * np.abs(cos_val) + cols * np.abs(sin_val))), int(np.round(rows * np.abs(sin_val) + cols * np.abs(cos_val)))
    
    # 指定旋转中心，可以修改为任意点
    # center = (cols / 2, rows / 2)  # 图像中心 (cols / 2, rows / 2)
    center = (200, 200)
    # 平移矩阵t 将原图像的旋转中心平移到原点
    t = np.array([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]])
    # 将原点平移到新图像的中心
    t_inv = np.array([[1, 0, new_cols / 2], [0, 1, new_rows / 2], [0, 0, 1]])
    M = t_inv @ M @ t  # @表示矩阵乘法

    # 将新图像的每个像素坐标映射回原图像坐标
    # 创建网格坐标并应用逆变换
    y, x = np.indices((new_rows, new_cols)) 
    # 解释:
      # np.indices() 创建一个表示新图像尺寸的坐标网格
      # 3x4的图像:  x表示列坐标，y表示行坐标
      # x = [[0 1 2 3]
      #      [0 1 2 3]
      #      [0 1 2 3]]
      # y = [[0 0 0 0]
      #      [1 1 1 1]
      #      [2 2 2 2]]
      # (y的第(i,j)位置的值，x的第(i,j)位置的值)表示原图像中的(i,j)位置的坐标
    coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()]) 
    # 解释:
      # - x.ravel() 和 y.ravel() 将坐标数组展平
      # - np.ones_like(x).ravel() 创建一个与x相同形状的全1数组,然后展平
      # - np.stack() 将这三个数组堆叠在一起
      # 结果是一个3xN的数组,其中N = new_rows * new_cols
      # coords = [[0 1 2 3 0 1 2 3 0 1 2 3]  # x 列
      #           [0 0 0 0 1 1 1 1 2 2 2 2]  # y 行
      #           [1 1 1 1 1 1 1 1 1 1 1 1]] # 齐次坐标
    x_transformed, y_transformed = np.linalg.inv(M).dot(coords)[:2, :]
    # - np.linalg.inv(M) 计算变换矩阵M的逆矩阵
      # - .dot(coords) 将逆矩阵应用到坐标上
      # - [:2, :] 只保留前两行(x和y坐标,丢弃齐次坐标)
      # 这一步计算出新图像中每个像素对应到原图像中的位置
    # 将变换后的x和y坐标重新调整为(new_rows, new_cols)的形状  注意此时x_transformed表示行，y_transformed表示列
    x_transformed = x_transformed.reshape(new_rows, new_cols)  # 重塑为新图像的形状  new_rows行 new_cols列
    y_transformed = y_transformed.reshape(new_rows, new_cols)  # 那么，x_transformed的（i，j）位置的值表示新图像中的（i，j）位置对应的原图像中的行坐标
    # (x_transformed的（i，j）位置的值，y_transformed的（i，j）位置的值) 表示新图像中的（i，j）位置对应的原图像中的坐标
    # 使用双线性插值进行采样
    x0, y0 = np.floor(x_transformed).astype(int), np.floor(y_transformed).astype(int)  # 向下取整 x0,y0表示采样点的左上角点
    x1, y1 = x0 + 1, y0 + 1    # 右下角点
    
    # 限制采样点在图像范围内
    x0 = np.clip(x0, 0, cols-1)
    x1 = np.clip(x1, 0, cols-1)
    y0 = np.clip(y0, 0, rows-1)
    y1 = np.clip(y1, 0, rows-1)

    # 计算插值权重
    wa = (x1 - x_transformed) * (y1 - y_transformed)
    wb = (x1 - x_transformed) * (y_transformed - y0)
    wc = (x_transformed - x0) * (y1 - y_transformed)
    wd = (x_transformed - x0) * (y_transformed - y0)

    # 分别计算左上角、右上角、左下角、右下角的颜色贡献
    new_img = (wa[:, :, np.newaxis] * img[y0, x0] +
               wb[:, :, np.newaxis] * img[y1, x0] +
               wc[:, :, np.newaxis] * img[y0, x1] +
               wd[:, :, np.newaxis] * img[y1, x1]).astype(img.dtype)

    time2 = time.time()
    print("numpy旋转程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))
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
#    M = cv.getRotationMatrix2D((a, b), rot, 1)  # 旋转矩阵
#    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1]) # 计算旋转后的图像尺寸
#    new_cols = int((rows * sin) + (cols * cos)) # 计算新图像的宽和高
#    new_rows = int((rows * cos) + (cols * sin)) 
#    # 更新旋转矩阵
#    M[0, 2] += (new_cols / 2) - w
#    M[1, 2] += (new_rows / 2) - h
#    # 进行仿射变换
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
   width, height = 900, 540  # 认为名片的大小是900x540
   # 源图像中的四个角点
   pts1 = np.float32(l_ImgPot[2])
   # 目标图像中的四个角点
   pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
   print(pts1)
   print(pts2)
   # 计算透视变换矩阵
   matrix = cv.getPerspectiveTransform(pts1, pts2)  # 直接用opencv的getPerspectiveTransform函数计算变换矩阵
   print(matrix)
   # getPerspectiveTransform()函数的参数是两个数组，分别是原图像中的四个点和目标图像中的四个点
   # 进行透视变换
   # cv.warpPerspective()函数的参数是原图像、变换矩阵和输出图像的大小 warpPerspective函数是opencv中的透视变换函数 直接实现了透视变换
   new_img = cv.warpPerspective(img, matrix, (width, height))

   time2 = time.time()
   print("opencv名片矫正处理时间：%.3f毫秒" %((time2 - time1) * 1000))

   return new_img