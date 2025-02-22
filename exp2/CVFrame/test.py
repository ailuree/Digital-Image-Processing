import numpy as np

# 假设我们有一个3x3的图像,我们要将其旋转45度
original_image = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print("原始图像:")
print(original_image)

# 步骤1: 定义旋转角度和旋转中心
angle = 90  # 旋转角度(度)
center = (0, 0)  # 旋转中心(图像中心)

# 步骤2: 计算新图像的尺寸
rows, cols = original_image.shape
angle_rad = np.radians(angle)
cos_val = np.abs(np.cos(angle_rad))
sin_val = np.abs(np.sin(angle_rad))
new_rows = int(np.round(rows * cos_val + cols * sin_val))
new_cols = int(np.round(rows * sin_val + cols * cos_val))
print(f"新图像尺寸: {new_rows}x{new_cols}")

# 步骤3: 创建坐标网格
y, x = np.indices((new_rows, new_cols))
print("x坐标网格:")
print(x)
print("y坐标网格:")
print(y)

# 步骤4: 创建齐次坐标
coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()])
print("齐次坐标(全部):")
print(coords)

# 步骤5: 构建变换矩阵
t = np.array([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]])
rot = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1]])
t_inv = np.array([[1, 0, new_cols/2], [0, 1, new_rows/2], [0, 0, 1]])
M = t_inv @ rot @ t
print("变换矩阵M:")
print(M)

# 步骤6: 应用逆变换
M_inv = np.linalg.inv(M)
transformed_coords = M_inv @ coords
x_transformed, y_transformed = transformed_coords[:2, :]
print("变换后的坐标(全部):")
print(transformed_coords)

# 步骤7: 重塑变换后的坐标
x_transformed = x_transformed.reshape(new_rows, new_cols)
y_transformed = y_transformed.reshape(new_rows, new_cols)
print("重塑后的x坐标(全部):")
print(x_transformed)
print("重塑后的y坐标(全部):")
print(y_transformed)

# 输出新图像中的点在原图中的映射坐标
print("新图像中的点在原图中的映射坐标:")
for i in range(new_rows):
    for j in range(new_cols):
        print(f"新图像点 ({i}, {j}) -> 原图像点 ({x_transformed[i, j]:.2f}, {y_transformed[i, j]:.2f})")

# 步骤8: 使用双线性插值(简化版)
x0, y0 = np.floor(x_transformed).astype(int), np.floor(y_transformed).astype(int)
x1, y1 = x0 + 1, y0 + 1

# 确保坐标在图像范围内
x0 = np.clip(x0, 0, cols-1)
x1 = np.clip(x1, 0, cols-1)
y0 = np.clip(y0, 0, rows-1)
y1 = np.clip(y1, 0, rows-1)

# 计算插值权重
wa = (x1 - x_transformed) * (y1 - y_transformed)
wb = (x1 - x_transformed) * (y_transformed - y0)
wc = (x_transformed - x0) * (y1 - y_transformed)
wd = (x_transformed - x0) * (y_transformed - y0)

# 执行插值
new_image = (wa * original_image[y0, x0] + 
             wb * original_image[y1, x0] + 
             wc * original_image[y0, x1] + 
             wd * original_image[y1, x1])

print("旋转后的图像(全部):")
print(new_image.reshape(new_rows, new_cols))
