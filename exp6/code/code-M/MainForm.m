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

% --- Executes on button press in pushbutton10.   ��ͼ��
function pushbutton10_Callback(hObject, eventdata, handles)
file = fullfile(pwd, 'test/test.jpg');
[Filename, Pathname] = uigetfile({'*.jpg;*.tif;*.png;*.gif','All Image Files';...
    '*.*','All Files' }, '������֤��ͼ��',...
    file);
if isequal(Filename, 0) || isequal(Pathname, 0)
    return;
end
% ��ʾͼ��
axes(handles.axes1); cla reset;
axes(handles.axes2); cla reset;
set(handles.axes1, 'Box', 'on', 'Color', 'c', 'XTickLabel', '', 'YTickLabel', '');
set(handles.axes2, 'Box', 'on', 'Color', 'c', 'XTickLabel', '', 'YTickLabel', '');
set(handles.axes3, 'Box', 'on', 'Color', 'c', 'XTickLabel', '', 'YTickLabel', '');
set(handles.axes4, 'Box', 'on', 'Color', 'c', 'XTickLabel', '', 'YTickLabel', '');

%set(handles.text4, 'String', '');
% �洢
fileurl = fullfile(Pathname,Filename);
Img = imread(fileurl);
imshow(Img, [], 'Parent', handles.axes1);
title('ԭͼ')
handles.fileurl = fileurl;
handles.Img = Img;
guidata(hObject, handles);

% --- Executes on button press in pushbutton11.  ͼ��ȥ��
function pushbutton11_Callback(hObject, eventdata, handles)
if isequal(handles.Img, 0)
    return;
end
Img = handles.Img;
% ��ɫ�ռ�ת��
hsv = rgb2hsv(Img);
h = hsv(:, :, 1);
s = hsv(:, :, 2);
v = hsv(:, :, 3);

bw1 = h > 0.16 & h < 0.30;
bw2 = s > 0.65 & s < 0.80;
bw = bw1 & bw2;
% ����������
Imgr = Img(:, :, 1);
Imgg = Img(:, :, 2);
Imgb = Img(:, :, 3);
Imgr(bw) = 255;
Imgg(bw) = 255;
Imgb(bw) = 255;
% ȥ����
Imgbw = cat(3, Imgr, Imgg, Imgb);
axes(handles.axes2)
imshow(Imgbw, [], 'Parent', handles.axes2);
%figure(3)
%imshow(Imgbw)
title('ȥ����ͼ')
handles.Imgbw = Imgbw;
guidata(hObject, handles);

% --- Executes on button press in pushbutton12.   ���ֶ�λ
function pushbutton12_Callback(hObject, eventdata, handles)
if isequal(handles.Imgbw, 0)
    return;
end
Imgbw = handles.Imgbw;
% �ҶȻ�
Ig = rgb2gray(Imgbw);
% ��ֵ��
Ibw = im2bw(Ig, 0.8);

% ��������
sz = size(Ibw);
cs = sum(Ibw, 1);  % ��ֱͶӰ
mincs = min(cs);
maxcs = max(cs);
masksize = 16;

% ��ʼ��
S1 = []; E1 = [];
% 1��Ӧ��ʼ��2��Ӧ����
flag = 1;
s1 = 1;
tol = maxcs;

% ͨ��ͶӰ�ҵ��ַ�����ʼ�ͽ���λ��
while s1 < sz(2)
    for i = s1 : sz(2)
        % �ƶ��α�
        s2 = i;
        if cs(s2) < tol && flag == 1
            % �ﵽ��ʼλ��
            flag = 2;
            S1 = [S1 s2-1];
            break;
        elseif cs(s2) >= tol && flag == 2
            % �ﵽ����λ��
            flag = 1;
            E1 = [E1 s2];
            break;
        end
    end
    s1 = s2 + 1;
end

% ͼ��ɫ
Ibw = ~Ibw;

% ͼ��ϸ��
Ibw = bwmorph(Ibw, 'thin', inf);

% �洢ÿ���ַ��ı߽��λ��
Rect = cell(1, length(S1));

% ��ÿ���ָ�����ַ����д���
for i = 1 : length(S1)
    % ͼ��ü�
    Ibwi = Ibw(:, S1(i):E1(i));
    % ��ͨ������
    [L, num] = bwlabel(Ibwi);
    stats = regionprops(L);
    % ��ȡ��������ͨ����
    Ar = cat(1, stats.Area);
    [maxAr, ind_maxAr] = max(Ar);
    % ��ȡ�߽��
    recti = stats(ind_maxAr).BoundingBox;
    recti(1) = recti(1) + S1(i) - 1;
    Rect{i} = recti;
    
    % �ü�����һ����ָ����С
    Ibwi = imcrop(Ibw, recti);
    rate = masksize/max(size(Ibwi));
    Ibwi = imresize(Ibwi, rate, 'bilinear');
    ti = zeros(masksize, masksize);
    rsti = round((size(ti, 1)-size(Ibwi, 1))/2);
    csti = round((size(ti, 2)-size(Ibwi, 2))/2);
    ti(rsti+1:rsti+size(Ibwi,1), csti+1:csti+size(Ibwi,2)) = Ibwi;
    % �洢
    Ti{i} = ti;
end

% ��ʾ��λ���
axes(handles.axes3)
imshow(Ibw); 
hold on;
% ��ͼ���ϻ��ƺ�ɫ�߽��
for i = 1 : length(Rect)
    rectangle('Position', Rect{i}, 'EdgeColor', 'r', 'LineWidth', 2);
end
hold off;
title('�ַ���λ���')

% ����ָ���������ʶ��ʹ��
handles.Ti = Ti;
guidata(hObject, handles);

% --- Executes on button press in pushbutton13.   �ַ��ָ��һ��
function pushbutton13_Callback(hObject, eventdata, handles)
if isequal(handles.Ti, 0)
    return;
end

Ti = handles.Ti;

% ��axes4����ʾ���зָ���
axes(handles.axes4);
cla reset;

% ������ʾ����
char_width = 50;  % ÿ���ַ��Ŀ��
char_height = 50; % ÿ���ַ��ĸ߶�
spacing = 20;     % �ַ����
total_width = length(Ti) * char_width + (length(Ti)-1) * spacing;

% �������ͼ��
combined_image = ones(char_height, total_width);

% ��������ַ���һ��ͼ����
for i = 1:length(Ti)
    % ��ȡ��ǰ�ַ�ͼ�񲢵�����С
    char_img = imresize(Ti{i}, [char_height char_width]);
    
    % �����ַ�λ��
    start_col = (i-1)*(char_width + spacing) + 1;
    end_col = start_col + char_width - 1;
    
    % �����ַ�
    combined_image(:, start_col:end_col) = char_img;
end

% ��ʾ���ͼ��
imshow(combined_image, []);
hold on;

% ��ӷָ���
for i = 1:(length(Ti)-1)
    x = i*(char_width + spacing) - spacing/2;
    line([x x], [0 char_height], 'Color', 'r', 'LineWidth', 1);
end
hold off;
title('�ַ��ָ���');

% �����½ǵ��ĸ�������ʾ�����ַ�
axes_handles = [handles.axes5, handles.axes6, handles.axes7, handles.axes8];
for i = 1:min(length(Ti), 4)
    axes(axes_handles(i));
    cla reset;
    imshow(Ti{i}, []);
    title(['�ַ�', num2str(i)]);
end

% ����handles
handles.segmented_chars = Ti;
guidata(hObject, handles);

% --- Executes on button press in pushbutton15.   ����ʶ��
function pushbutton15_Callback(hObject, eventdata, handles)
if isequal(handles.Ti, 0)
    disp('TiΪ�գ���������ַ��ָ�');
    return;
end

Ti = handles.Ti;
disp(['��⵽ ', num2str(length(Ti)), ' ���ַ���Ҫʶ��']);

% ���ģ���·��
template_path = fullfile(pwd, 'Database');
if ~exist(template_path, 'dir')
    disp(['ģ���·�������ڣ�', template_path]);
    return;
end

% ����ģ����е�����ģ��ͼ��
fileList = GetAllFiles(template_path);
if isempty(fileList)
    disp('δ�ҵ�ģ���ļ�');
    return;
end
disp(['�ҵ� ', num2str(length(fileList)), ' ��ģ���ļ�']);

Tj = [];

% ��ÿ��ģ�����������ȡ�ͱȶ�
for i = 1 : length(fileList)
    filenamei = fileList{i};
    [pathstr, name, ext] = fileparts(filenamei);
    if isequal(ext, '.jpg')
        try
            % ��ȡģ��ͼ��
            ti = imread(filenamei);
            ti = im2bw(ti, 0.5);
            ti = double(ti);
            % ��ȡģ��Ĳ��������
            phii = invmoments(ti);
            
            % ��ÿ����ʶ���ַ����бȶ�
            OTj = [];
            for j = 1 : length(Ti)
                tij = double(Ti{j});
                % ��ȡ��ʶ���ַ��Ĳ��������
                phij = invmoments(tij);
                % ��������??��
                ad = norm(phii-phij);
                otij.filename = filenamei;
                otij.ad = ad;
                OTj = [OTj otij];
            end
            Tj = [Tj; OTj];
        catch ME
            disp(['����ģ���ļ�����', filenamei]);
            disp(['������Ϣ��', ME.message]);
        end
    end
end

if isempty(Tj)
    disp('ģ��ƥ��ʧ��');
    return;
end

% ʶ����
result = '';
for i = 1 : size(Tj, 2)
    ti = Tj(:, i);
    % ��ȡ��������
    adi = cat(1, ti.ad);
    % �ҵ���С�����Ӧ��ģ��
    [minadi, ind] = min(adi);
    filenamei = ti(ind).filename;
    % ���ļ�������ȡ����
    [pathstr, name, ext] = fileparts(filenamei);
    [~, folder, ~] = fileparts(pathstr);
    result = [result folder];
    disp(['��', num2str(i), '���ַ�ʶ������', folder]);
end

% ʹ��text11�ؼ���ʾʶ����
text11_handle = findobj(gcf, 'Tag', 'text11');  % ��ȡtext11�ؼ��ľ��
if ~isempty(text11_handle)
    set(text11_handle, 'String', result);
    disp(['����ʶ������', result]);
else
    disp('�Ҳ���text11�ؼ�');
end

% ����handles
handles.result = result;
guidata(hObject, handles);
