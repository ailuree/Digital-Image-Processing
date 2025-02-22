function varargout = MainForm(varargin)

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @MainForm_OpeningFcn, ...
    'gui_OutputFcn',  @MainForm_OutputFcn, ...
    'gui_LayoutFcn',  [] , ...
    'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end

function MainForm_OpeningFcn(hObject, eventdata, handles, varargin)

handles.output = hObject;
set(gcf,'name','hzy');
clc;
set(handles.axes1, 'Box', 'on', 'Color', 'c', 'XTickLabel', '', 'YTickLabel', '');
set(handles.axes2, 'Box', 'on', 'Color', 'c', 'XTickLabel', '', 'YTickLabel', '');
set(handles.axes3, 'Box', 'on', 'Color', 'c', 'XTickLabel', '', 'YTickLabel', '');
set(handles.axes4, 'Box', 'on', 'Color', 'c', 'XTickLabel', '', 'YTickLabel', '');
set(handles.axes5, 'Box', 'on', 'Color', 'c', 'XTickLabel', '', 'YTickLabel', '');
set(handles.axes6, 'Box', 'on', 'Color', 'c', 'XTickLabel', '', 'YTickLabel', '');
set(handles.axes7, 'Box', 'on', 'Color', 'c', 'XTickLabel', '', 'YTickLabel', '');
set(handles.axes8, 'Box', 'on', 'Color', 'c', 'XTickLabel', '', 'YTickLabel', '');

handles.fileurl = 0;
handles.Img = 0;
handles.Imgbw = 0;
handles.Ti = 0;
% Update handles structure
guidata(hObject, handles);

% --- Outputs from this function are returned to the command line.
function varargout = MainForm_OutputFcn(hObject, eventdata, handles)

% Get default command line output from handles structure
varargout{1} = handles.output;

function varargout = pushbutton1_CreateFcn(hObject, eventdata, handles)

GetDatabase();

% --- Executes on button press in pushbutton10.   打开图像
function pushbutton10_Callback(hObject, eventdata, handles)
file = fullfile(pwd, 'test/test.jpg');
[Filename, Pathname] = uigetfile({'*.jpg;*.tif;*.png;*.gif','All Image Files';...
    '*.*','All Files' }, '载入验证码图像',...
    file);
if isequal(Filename, 0) || isequal(Pathname, 0)
    return;
end
% 显示图像
axes(handles.axes1); cla reset;
axes(handles.axes2); cla reset;
set(handles.axes1, 'Box', 'on', 'Color', 'c', 'XTickLabel', '', 'YTickLabel', '');
set(handles.axes2, 'Box', 'on', 'Color', 'c', 'XTickLabel', '', 'YTickLabel', '');
set(handles.axes3, 'Box', 'on', 'Color', 'c', 'XTickLabel', '', 'YTickLabel', '');
set(handles.axes4, 'Box', 'on', 'Color', 'c', 'XTickLabel', '', 'YTickLabel', '');

%set(handles.text4, 'String', '');
% 存储
fileurl = fullfile(Pathname,Filename);
Img = imread(fileurl);
imshow(Img, [], 'Parent', handles.axes1);
title('原图')
handles.fileurl = fileurl;
handles.Img = Img;
guidata(hObject, handles);

% --- Executes on button press in pushbutton11.  图像去噪
function pushbutton11_Callback(hObject, eventdata, handles)
if isequal(handles.Img, 0)
    return;
end
Img = handles.Img;
% 颜色空间转换
hsv = rgb2hsv(Img);
h = hsv(:, :, 1);
s = hsv(:, :, 2);
v = hsv(:, :, 3);

bw1 = h > 0.16 & h < 0.30;
bw2 = s > 0.65 & s < 0.80;
bw = bw1 & bw2;
% 过滤噪音点
Imgr = Img(:, :, 1);
Imgg = Img(:, :, 2);
Imgb = Img(:, :, 3);
Imgr(bw) = 255;
Imgg(bw) = 255;
Imgb(bw) = 255;
% 去噪结果
Imgbw = cat(3, Imgr, Imgg, Imgb);
axes(handles.axes2)
imshow(Imgbw, [], 'Parent', handles.axes2);
%figure(3)
%imshow(Imgbw)
title('去噪结果图')
handles.Imgbw = Imgbw;
guidata(hObject, handles);

% --- Executes on button press in pushbutton12.   数字定位
function pushbutton12_Callback(hObject, eventdata, handles)
if isequal(handles.Imgbw, 0)
    return;
end
Imgbw = handles.Imgbw;
% 灰度化
Ig = rgb2gray(Imgbw);
% 二值化
Ibw = im2bw(Ig, 0.8);

% 常量参数
sz = size(Ibw);
cs = sum(Ibw, 1);  % 垂直投影
mincs = min(cs);
maxcs = max(cs);
masksize = 16;

% 初始化
S1 = []; E1 = [];
% 1对应开始，2对应结束
flag = 1;
s1 = 1;
tol = maxcs;

% 通过投影找到字符的起始和结束位置
while s1 < sz(2)
    for i = s1 : sz(2)
        % 移动游标
        s2 = i;
        if cs(s2) < tol && flag == 1
            % 达到起始位置
            flag = 2;
            S1 = [S1 s2-1];
            break;
        elseif cs(s2) >= tol && flag == 2
            % 达到结束位置
            flag = 1;
            E1 = [E1 s2];
            break;
        end
    end
    s1 = s2 + 1;
end

% 图像反色
Ibw = ~Ibw;

% 图像细化
Ibw = bwmorph(Ibw, 'thin', inf);

% 存储每个字符的边界框位置
Rect = cell(1, length(S1));

% 对每个分割出的字符进行处理
for i = 1 : length(S1)
    % 图像裁剪
    Ibwi = Ibw(:, S1(i):E1(i));
    % 连通区域标记
    [L, num] = bwlabel(Ibwi);
    stats = regionprops(L);
    % 获取积最大的连通区域
    Ar = cat(1, stats.Area);
    [maxAr, ind_maxAr] = max(Ar);
    % 获取边界框
    recti = stats(ind_maxAr).BoundingBox;
    recti(1) = recti(1) + S1(i) - 1;
    Rect{i} = recti;
    
    % 裁剪并归一化到指定大小
    Ibwi = imcrop(Ibw, recti);
    rate = masksize/max(size(Ibwi));
    Ibwi = imresize(Ibwi, rate, 'bilinear');
    ti = zeros(masksize, masksize);
    rsti = round((size(ti, 1)-size(Ibwi, 1))/2);
    csti = round((size(ti, 2)-size(Ibwi, 2))/2);
    ti(rsti+1:rsti+size(Ibwi,1), csti+1:csti+size(Ibwi,2)) = Ibwi;
    % 存储
    Ti{i} = ti;
end

% 显示定位结果
axes(handles.axes3)
imshow(Ibw); 
hold on;
% 在图像上绘制红色边界框
for i = 1 : length(Rect)
    rectangle('Position', Rect{i}, 'EdgeColor', 'r', 'LineWidth', 2);
end
hold off;
title('字符定位结果')

% 保存分割结果供后续识别使用
handles.Ti = Ti;
guidata(hObject, handles);

% --- Executes on button press in pushbutton13.   字符分割归一化
function pushbutton13_Callback(hObject, eventdata, handles)
if isequal(handles.Ti, 0)
    return;
end

Ti = handles.Ti;

% 在axes4中显示所有分割结果
axes(handles.axes4);
cla reset;

% 设置显示参数
char_width = 50;  % 每个字符的宽度
char_height = 50; % 每个字符的高度
spacing = 20;     % 字符间距
total_width = length(Ti) * char_width + (length(Ti)-1) * spacing;

% 创建组合图像
combined_image = ones(char_height, total_width);

% 组合所有字符到一个图像中
for i = 1:length(Ti)
    % 获取当前字符图像并调整大小
    char_img = imresize(Ti{i}, [char_height char_width]);
    
    % 计算字符位置
    start_col = (i-1)*(char_width + spacing) + 1;
    end_col = start_col + char_width - 1;
    
    % 放置字符
    combined_image(:, start_col:end_col) = char_img;
end

% 显示组合图像
imshow(combined_image, []);
hold on;

% 添加分隔线
for i = 1:(length(Ti)-1)
    x = i*(char_width + spacing) - spacing/2;
    line([x x], [0 char_height], 'Color', 'r', 'LineWidth', 1);
end
hold off;
title('字符分割结果');

% 在左下角的四个框中显示单个字符
axes_handles = [handles.axes5, handles.axes6, handles.axes7, handles.axes8];
for i = 1:min(length(Ti), 4)
    axes(axes_handles(i));
    cla reset;
    imshow(Ti{i}, []);
    title(['字符', num2str(i)]);
end

% 更新handles
handles.segmented_chars = Ti;
guidata(hObject, handles);

% --- Executes on button press in pushbutton15.   数字识别
function pushbutton15_Callback(hObject, eventdata, handles)
if isequal(handles.Ti, 0)
    disp('Ti为空，请先完成字符分割');
    return;
end

Ti = handles.Ti;
disp(['检测到 ', num2str(length(Ti)), ' 个字符需要识别']);

% 检查模板库路径
template_path = fullfile(pwd, 'Database');
if ~exist(template_path, 'dir')
    disp(['模板库路径不存在：', template_path]);
    return;
end

% 加载模板库中的所有模板图像
fileList = GetAllFiles(template_path);
if isempty(fileList)
    disp('未找到模板文件');
    return;
end
disp(['找到 ', num2str(length(fileList)), ' 个模板文件']);

Tj = [];

% 对每个模板进行特征提取和比对
for i = 1 : length(fileList)
    filenamei = fileList{i};
    [pathstr, name, ext] = fileparts(filenamei);
    if isequal(ext, '.jpg')
        try
            % 读取模板图像
            ti = imread(filenamei);
            ti = im2bw(ti, 0.5);
            ti = double(ti);
            % 提取模板的不变矩特征
            phii = invmoments(ti);
            
            % 与每个待识别字符进行比对
            OTj = [];
            for j = 1 : length(Ti)
                tij = double(Ti{j});
                % 提取待识别字符的不变矩特征
                phij = invmoments(tij);
                % 计算特征??离
                ad = norm(phii-phij);
                otij.filename = filenamei;
                otij.ad = ad;
                OTj = [OTj otij];
            end
            Tj = [Tj; OTj];
        catch ME
            disp(['处理模板文件出错：', filenamei]);
            disp(['错误信息：', ME.message]);
        end
    end
end

if isempty(Tj)
    disp('模板匹配失败');
    return;
end

% 识别结果
result = '';
for i = 1 : size(Tj, 2)
    ti = Tj(:, i);
    % 获取特征距离
    adi = cat(1, ti.ad);
    % 找到最小距离对应的模板
    [minadi, ind] = min(adi);
    filenamei = ti(ind).filename;
    % 从文件名中提取数字
    [pathstr, name, ext] = fileparts(filenamei);
    [~, folder, ~] = fileparts(pathstr);
    result = [result folder];
    disp(['第', num2str(i), '个字符识别结果：', folder]);
end

% 使用text11控件显示识别结果
text11_handle = findobj(gcf, 'Tag', 'text11');  % 获取text11控件的句柄
if ~isempty(text11_handle)
    set(text11_handle, 'String', result);
    disp(['最终识别结果：', result]);
else
    disp('找不到text11控件');
end

% 更新handles
handles.result = result;
guidata(hObject, handles);
