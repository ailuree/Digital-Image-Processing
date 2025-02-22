# log算子
def log(img):
    """*功能 : 根据LOG算子对应的卷积模板，对图像进行边缘检测"""
    time1 = time.time()  # 程序计时开始
    # # numpy实现 -- 使用FFT进行卷积操作
    # prw_conv = np.array([[-2, -4, -4, -4, -2], [-4, 0, 8, 0, -4], [-4, 8, 24, 8, -4],
    #                      [-4, 0, 8, 0, -4], [-2, -4, -4, -4, -2]], dtype=float)
    # # 高斯模糊
    # img = cv.GaussianBlur(img, (3, 3), 0)
    # # 使用快速傅里叶变换（FFT）进行卷积
    # img_pad = np.pad(img, ((2, 2), (2, 2)), mode='edge')
    # img_fft = np.fft.fft2(img_pad)
    # kernel_fft = np.fft.fft2(prw_conv, s=img_pad.shape)
    # new_img = np.real(np.fft.ifft2(img_fft * kernel_fft))[2:-2, 2:-2]
    # # 取绝对值并限制在0-255范围内
    # new_img = np.abs(new_img)
    # new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    # time2 = time.time()  # 程序计时结束
    # print("log算子边缘检测NumPy FFT优化程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # # numpy实现 -- 使用scipy的convolve2d进行卷积操作
    # prw_conv = np.array([[-2, -4, -4, -4, -2], [-4, 0, 8, 0, -4], [-4, 8, 24, 8, -4],
    #                      [-4, 0, 8, 0, -4], [-2, -4, -4, -4, -2]], dtype=float)
    # # 高斯模糊
    # img = cv.GaussianBlur(img, (3, 3), 0)
    # # 使用scipy的convolve2d进行卷积操作
    # new_img = convolve2d(img, prw_conv, mode='same', boundary='symm')
    # # 取绝对值并限制在0-255范围内
    # new_img = np.abs(new_img)
    # new_img = np.clip(new_img, 0, 255).astype(np.uint8)
    # time2 = time.time()  # 程序计时结束
    # print("log算子边缘检测NumPy scipy-convolve2d优化程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    # opencv实现
    # 高斯模糊
    img_blur = cv.GaussianBlur(img, (3, 3), 0)
    # 直接对原图应用log算子
    new_img = cv.Laplacian(img_blur, cv.CV_64F, ksize=5)
    # 取绝对值并限制在0-255范围内
    new_img = np.uint8(np.clip(np.abs(new_img), 0, 255))
    time2 = time.time()  # 程序计时结束
    print("log算子边缘检测CV程序处理时间：%.3f毫秒" % ((time2 - time1) * 1000))

    return new_img


    # img_blur = cv.GaussianBlur(img, (3, 3), 0)           # 高斯模糊 (3，3)高斯核大小 0是标准差
    # new_img = cv.Laplacian(img_blur, cv.CV_64F, ksize=5) # 直接对原图应用log算子  ksize使用5×5的算子 cv.CV_64F是输出图像的深度
    # new_img = np.uint8(np.clip(new_img, 0, 255))         # 限制结果在0-255范围内