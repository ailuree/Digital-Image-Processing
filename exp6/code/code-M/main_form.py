import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from skimage.morphology import thin
from utils import get_all_files, compute_invariant_moments

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.fileurl = None
        self.img = None
        self.img_bw = None
        self.Ti = None
        
    def initUI(self):
        self.setWindowTitle('验证码识别系统')
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建主布局
        main_layout = QHBoxLayout()
        
        # 创建左侧显示区域的容器和布局
        left_widget = QWidget()
        left_layout = QGridLayout()
        left_widget.setLayout(left_layout)
        
        # 创建四个主显示区域
        self.axes1 = QLabel()  # 原始图像
        self.axes2 = QLabel()  # 去噪图像
        self.axes3 = QLabel()  # 定位图像
        self.axes4 = QLabel()  # 分割图像
        
        # 创建四个小框用于显示分割后的数字
        self.axes5 = QLabel()
        self.axes6 = QLabel()
        self.axes7 = QLabel()
        self.axes8 = QLabel()
        
        # 创建识别结果显示标签
        result_label = QLabel("识别结果：")
        self.text11 = QLabel()
        result_layout = QHBoxLayout()
        result_layout.addWidget(result_label)
        result_layout.addWidget(self.text11)
        result_layout.addStretch()
        
        # 设置所有显示区域的样式
        for axes in [self.axes1, self.axes2, self.axes3, self.axes4]:
            axes.setStyleSheet("background-color: cyan; border: 1px solid black;")
            axes.setMinimumSize(250, 200)
            axes.setAlignment(Qt.AlignCenter)
        
        # 设置小框的样式
        for axes in [self.axes5, self.axes6, self.axes7, self.axes8]:
            axes.setStyleSheet("background-color: cyan; border: 1px solid black;")
            axes.setMinimumSize(100, 100)
            axes.setMaximumSize(100, 100)
            axes.setAlignment(Qt.AlignCenter)
        
        # 添加四个主显示区域到布局
        left_layout.addWidget(self.axes1, 0, 0)
        left_layout.addWidget(self.axes2, 0, 1)
        left_layout.addWidget(self.axes3, 1, 0)
        left_layout.addWidget(self.axes4, 1, 1)
        
        # 创建底部布局
        bottom_layout = QHBoxLayout()
        
        # 添加四个小框到底部布局左侧
        small_boxes_layout = QHBoxLayout()
        for axes in [self.axes5, self.axes6, self.axes7, self.axes8]:
            small_boxes_layout.addWidget(axes)
        small_boxes_layout.addStretch()
        
        # 将小框和识别结果添加到底部布局
        bottom_widget = QWidget()
        bottom_widget.setLayout(bottom_layout)
        bottom_layout.addLayout(small_boxes_layout)
        bottom_layout.addLayout(result_layout)
        
        # 将底部布局添加到左侧主布局
        left_layout.addWidget(bottom_widget, 2, 0, 1, 2)
        
        # 创建右侧按钮布局
        right_layout = QVBoxLayout()
        
        # 创建按钮
        self.btn_load = QPushButton('打开')
        self.btn_denoise = QPushButton('去噪')
        self.btn_locate = QPushButton('定位')
        self.btn_segment = QPushButton('分割')
        self.btn_recognize = QPushButton('识别')
        
        # 设置按钮大小
        button_size = QSize(100, 40)
        for btn in [self.btn_load, self.btn_denoise, self.btn_locate, 
                    self.btn_segment, self.btn_recognize]:
            btn.setMinimumSize(button_size)
            btn.setMaximumSize(button_size)
            right_layout.addWidget(btn)
            right_layout.addSpacing(20)  # 添加按钮间距
        
        right_layout.addStretch()  # 在按钮下方添加弹性空间
        
        # 将左显示区域和右侧按钮添加到主布局
        main_layout.addWidget(left_widget, stretch=4)  # 左侧占据更多空间
        
        # 创建右侧按钮容器
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        main_layout.addWidget(right_widget, stretch=1)  # 右侧占据较少空间
        
        # 创建中心部件并设置主布局
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # 连接信号和槽
        self.btn_load.clicked.connect(self.load_image)
        self.btn_denoise.clicked.connect(self.denoise_image)
        self.btn_locate.clicked.connect(self.locate_chars)
        self.btn_segment.clicked.connect(self.segment_chars)
        self.btn_recognize.clicked.connect(self.recognize_chars)

    def load_image(self):
        """载入图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "载入验证码图像", 
            "test/test.jpg",
            "Image files (*.jpg *.png *.gif *.tif);;All files (*.*)"
        )
        
        if not file_path:
            return
            
        # 读取图像并保持原始清晰度
        self.img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if self.img is None:
            print("无法读取图像")
            return
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.show_image(self.axes1, self.img, "原图")
        self.fileurl = file_path
        
    def denoise_image(self):
        """图像去噪"""
        if self.img is None:
            return
            
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        # 定位噪点
        bw1 = (h > 0.16 * 180) & (h < 0.30 * 180)
        bw2 = (s > 0.65 * 255) & (s < 0.80 * 255)
        bw = bw1 & bw2
        
        # 过滤噪点
        img_denoised = self.img.copy()
        img_denoised[bw] = [255, 255, 255]
        
        # 显示结果
        self.img_bw = img_denoised
        self.show_image(self.axes2, img_denoised, "去噪结果")
        
    def locate_chars(self):
        """字符定位"""
        if self.img_bw is None:
            return
            
        # 灰度化和二值化
        gray = cv2.cvtColor(self.img_bw, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0.8 * 255, 255, cv2.THRESH_BINARY)
        
        # 获取图像尺寸和垂直投影
        sz = binary.shape
        cs = np.sum(binary, axis=0)  # 垂直投影
        maxcs = np.max(cs)
        
        # 初始化
        S1 = []  # 起始位置
        E1 = []  # 结束位置
        flag = 1
        s1 = 0
        tol = maxcs  # 使用最大投影值作为阈值
        
        # 通过投影找到字符的起始和结束位置
        while s1 < sz[1]:
            s2 = s1
            if flag == 1:
                while s2 < sz[1] and cs[s2] >= tol:
                    s2 += 1
                if s2 < sz[1]:
                    S1.append(s2)
                    flag = 2
            else:
                while s2 < sz[1] and cs[s2] < tol:
                    s2 += 1
                if s2 < sz[1]:
                    E1.append(s2)
                    flag = 1
            s1 = s2 + 1
        
        # 图像反色和细化
        binary = ~binary
        binary = thin(binary)
        
        # 处理每个字符
        self.Ti = []
        img_with_boxes = self.img_bw.copy()
        
        for i in range(len(S1)):
            # 裁剪字符区域
            Ibwi = binary[:, S1[i]:E1[i]]
            
            # 获取连通区域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                Ibwi.astype(np.uint8), connectivity=8
            )
            
            if num_labels < 2:
                continue
                
            # 获取最大连通区域（排除背景）
            areas = stats[1:, cv2.CC_STAT_AREA]  # 跳过背景
            max_area_idx = 1 + np.argmax(areas)
            
            # 获取边界框
            x = stats[max_area_idx, cv2.CC_STAT_LEFT]
            y = stats[max_area_idx, cv2.CC_STAT_TOP]
            w = stats[max_area_idx, cv2.CC_STAT_WIDTH]
            h = stats[max_area_idx, cv2.CC_STAT_HEIGHT]
            
            # 调整边界框位置
            x = S1[i] + x
            recti = [x, y, w, h]
            
            # 绘制边界框
            cv2.rectangle(img_with_boxes, 
                         (int(recti[0]), int(recti[1])), 
                         (int(recti[0] + recti[2]), int(recti[1] + recti[3])), 
                         (255, 0, 0), 1)
            
            # 提取并归一化字符
            char_img = binary[int(recti[1]):int(recti[1]+recti[3]), 
                             int(recti[0]):int(recti[0]+recti[2])]
            char_normalized = cv2.resize(char_img.astype(np.float32), (16, 16))
            self.Ti.append(char_normalized)
        
        self.show_image(self.axes3, img_with_boxes, "字符定位结果")

    def segment_chars(self):
        """字符分割"""
        if self.Ti is None:
            return
            
        # 设置显示参数
        char_width = 100   # 增大字符显示大小
        char_height = 100
        spacing = 40      # 增加间距
        total_width = len(self.Ti) * char_width + (len(self.Ti)-1) * spacing
        
        # 创建组合图像（白色背景）
        combined_image = np.ones((char_height, total_width), dtype=np.float32)
        
        # 组合所有字符到一个图像中
        for i, char in enumerate(self.Ti):
            # 反转颜色使字符为黑色
            char_inv = 1 - char
            
            # 调整字符大小，保持纵横比
            h, w = char_inv.shape
            scale = min(char_width/w, char_height/h) * 0.8  # 缩小一点以留出边距
            new_w = int(w * scale)
            new_h = int(h * scale)
            char_resized = cv2.resize(char_inv, (new_w, new_h))
            
            # 计算位置（居中放置）
            start_col = i * (char_width + spacing)
            y_offset = (char_height - new_h) // 2
            x_offset = (char_width - new_w) // 2
            
            # 放置字符
            combined_image[y_offset:y_offset+new_h, 
                         start_col+x_offset:start_col+x_offset+new_w] = char_resized
        
        # 显示组合图像
        img_with_lines = cv2.cvtColor((combined_image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # 添加分隔线
        for i in range(len(self.Ti)-1):
            x = (i+1) * (char_width + spacing) - spacing//2
            cv2.line(img_with_lines, (x, 0), (x, char_height), (255, 0, 0), 1)
        
        self.show_image(self.axes4, img_with_lines, "字符分割结果")
        
        # 在左下角显示单个字符
        axes_list = [self.axes5, self.axes6, self.axes7, self.axes8]
        for i, (char, axes) in enumerate(zip(self.Ti[:4], axes_list)):
            # 反转颜色
            char_inv = 1 - char
            
            # 调整大小时保持纵横比
            h, w = char_inv.shape
            scale = min(80/w, 80/h)  # 留出边距
            new_w = int(w * scale)
            new_h = int(h * scale)
            char_resized = cv2.resize(char_inv, (new_w, new_h))
            
            # 创建白色背景
            display_img = np.ones((100, 100), dtype=np.float32)
            # 居中放置字符
            y_offset = (100 - new_h) // 2
            x_offset = (100 - new_w) // 2
            display_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = char_resized
            
            self.show_image(axes, display_img, f"字符{i+1}")

    def recognize_chars(self):
        """字符识别"""
        if self.Ti is None:
            print('Ti为空，请先完成字符分割')
            return
            
        print(f'检测到 {len(self.Ti)} 个字符需要识别')
        
        # 检查模板库路径
        template_path = os.path.join(os.getcwd(), 'Database')
        if not os.path.exists(template_path):
            print(f'模板库路径不存在：{template_path}')
            return
            
        # 加载模板
        template_files = get_all_files(template_path)
        if not template_files:
            print('未找到模板文件')
            return
            
        print(f'找到 {len(template_files)} 个模板文件')
        
        # 模板匹配
        results = []
        for char_img in self.Ti:
            char_moments = compute_invariant_moments(char_img)
            
            # 与所有模板比较
            min_distance = float('inf')
            best_match = None
            
            for template_file in template_files:
                if not template_file.endswith('.jpg'):
                    continue
                    
                try:
                    # 读取模板
                    template = cv2.imread(template_file, cv2.IMREAD_GRAYSCALE)
                    template = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)[1]
                    template = template.astype(np.float32) / 255.0
                    
                    # 计算模板的不变矩
                    template_moments = compute_invariant_moments(template)
                    
                    # 计算距离
                    distance = np.linalg.norm(char_moments - template_moments)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_match = os.path.basename(os.path.dirname(template_file))
                        
                except Exception as e:
                    print(f'处理模板文件出错：{template_file}')
                    print(f'错误信息：{str(e)}')
                    
            if best_match is not None:
                results.append(best_match)
                print(f'字符识别结果：{best_match}')
                
        # 显示最终结果
        if results:
            final_result = ''.join(results)
            self.text11.setText(final_result)
            print(f'最终识别结果：{final_result}')
        else:
            print('识别失败')

    def show_image(self, label, img, title=""):
        """在QLabel中显示图像"""
        if len(img.shape) == 2:
            # 如果是二值图像，转换为RGB
            img = cv2.cvtColor(img.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)
            
        height, width = img.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(
            img.data, width, height, bytes_per_line, 
            QImage.Format_RGB888
        )
        
        # 调整图像大小以适应标签
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        label.setPixmap(scaled_pixmap)
        if title:
            label.setToolTip(title)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())