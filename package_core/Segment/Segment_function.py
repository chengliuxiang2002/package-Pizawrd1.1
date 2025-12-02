import cv2
import numpy as np
import os
import shutil
import json
from PIL import Image

def increment(data):
    if isinstance(data,dict):
        for key,value in data.items():
            if key == 'page' and isinstance(value,int):
                data[key] += 1
            else:
                increment(value)
    elif isinstance(data,list):
        for item in data:
            increment(item)
    return data
def get_neighbours(img, x, y):
    """
    查找邻接点是否符合条件
    :param img:
    :param x:
    :param y:
    :return:
    """
    neighbours = []
    rows, cols = img.shape
    for i in range(-3, 4):
        for j in range(-3, 4):
            nx, ny = x + i, y + j
            if 0 <= nx < rows and 0 <= ny < cols and img[nx, ny] == 255:
                neighbours.append((nx, ny))
    return neighbours

def dfs(img, x, y, contour):
    """
    路径搜索，获取最外层边框
    :param img:
    :param x:
    :param y:
    :param contour:
    :return:
    """
    stack = [(x, y)]
    img[x, y] = 128  # 将已经访问的坐标标记
    contour.append((x, y))

    while stack:
        current_x, current_y = stack.pop()
        neighbours = get_neighbours(img, current_x, current_y)

        for neighbour in neighbours:
            nx, ny = neighbour
            if img[nx, ny] == 255:
                stack.append((nx, ny))
                img[nx, ny] = 128
                contour.append((nx, ny))

def find_contour(img, start_x, start_y):
    """
    获得最外层轮廓的坐标点集
    :param img:
    :param start_x:
    :param start_y:
    :return:
    """
    contour = []
    dfs(img, start_x, start_y, contour)
    return np.array(contour)

def morphological_treatment(img):
    """
    对图像进行形态学处理
    :param img:
    :return:
    """
    ret, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    # 像素反转
    inverted = cv2.bitwise_not(thresh)
    # 膨胀
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(inverted, kernel, iterations=6)
    # 边缘检测
    edges = cv2.Canny(dilated_image, 50, 150)

    return edges, dilated_image

def np_where(thresh):
    """
    查找首个白色像素区域
    :param thresh:
    :return:
    """
    found_white_pixel = False
    white_pixel_coords = np.where(thresh == 255)
    if white_pixel_coords[0].size == 0:
        x = 0
        y = 0
        return x,y,found_white_pixel
    else:
        # 获取第一个白色像素点的坐标
        x, y = white_pixel_coords[0][0], white_pixel_coords[1][0]
        found_white_pixel = True
        return x,y,found_white_pixel

def side_combined(side_path,side1_path,save_path):
    """
    合并两个side图片
    :param side_path:
    :param side1_path:
    :param save_path:
    :return:
    """
    side = cv2.imread(side_path)
    side1 = cv2.imread(side1_path)

    # 获取图片的尺寸
    height1, width1 = side.shape[:2]
    height2, width2 = side1.shape[:2]

    # 垂直拼接
    # 创建一个空白图像，宽度为两张图片中较大的宽度，高度为两张图片的高度之和
    vertical_combined = np.zeros((height1 + height2, max(width1, width2), 3), dtype=np.uint8)

    # 将两张图片粘贴到新的图像上
    vertical_combined[:height1, :width1] = side
    vertical_combined[height1:height1 + height2, :width2] = side1

    os.remove(side_path)
    os.remove(side1_path)
    # 保存垂直拼接后的图像
    save = save_path + '/side.jpg'
    cv2.imwrite(save, vertical_combined)

def empty_file(directory):
    """
        若存在该文件目录，则清空directory路径下的文件夹里的内容，否则创建空文件目录
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)  # 删除目录下所有文件及目录
        os.makedirs(directory)  # 创建目录
    else:
        os.makedirs(directory)

def move_files(source_folder, destination_folder):
    # 获取源文件夹中的所有文件
    files = os.listdir(source_folder)

    # 遍历所有文件，并移动到目标文件夹
    for file in files:
        source_path = os.path.join(source_folder, file)
        destination_path = os.path.join(destination_folder, file)
        shutil.copy(source_path, destination_path)
        # print(f"Copy {file} to {destination_folder}")

def calculate_intersection(rect1, rect2):
    # rect1 和 rect2 都是形如 (x1, y1, x2, y2) 的矩形
    x_left = max(rect1[0], rect2[0])
    y_top = max(rect1[1], rect2[1])
    x_right = min(rect1[2], rect2[2])
    y_bottom = min(rect1[3], rect2[3])

    # 检查交集矩形是否有效
    if x_right >= x_left and y_bottom >= y_top:
        # 计算交集矩形的面积
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        return intersection_area
    else:
        # 没有交集
        return 0


def merge_rectangles(rects):
    # rects 是一个形如 [(x1, y1, x2, y2), ...] 的矩形列表
    if not rects:
        return None  # 如果没有矩形，则返回None

    # 初始化边界为第一个矩形的边界
    x_left = rects[0][0]
    y_top = rects[0][1]
    x_right = rects[0][2]
    y_bottom = rects[0][3]

    # 遍历所有矩形，更新边界
    for rect in rects:
        x_left = min(x_left, rect[0])
        y_top = min(y_top, rect[1])
        x_right = max(x_right, rect[2])
        y_bottom = max(y_bottom, rect[3])

        # 返回包含所有矩形的最小矩形
    return [x_left, y_top, x_right, y_bottom]


def jz(pred_box, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 240, 255, 0)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    height, width = img.shape[:2]
    binary_image = np.zeros((height, width, 1), dtype=np.uint8)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < pred_area * 0.64:
            cv2.rectangle(binary_image, (x - 3, y - 3), (x + w + 3, y + h + 3), 255, 1)

    contours2, hierarchy2 = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    rects = []
    for cnt in contours2:
        x, y, w, h = cv2.boundingRect(cnt)
        its = calculate_intersection((x, y, x + w, y + h), pred_box) / (w * h)
        if its > 0.8:
            rects.append((x + 3, y + 3, x + w - 3, y + h - 3))
    correct_box = merge_rectangles(rects)
    return correct_box

def get_type(package_information):
    type = {}
    for data in package_information:
        type[data['page']] = data['package_type']
    return type

def manage_json(package_information):
    """
    :param package_information: 传入的封装信息
    :return cla_lis: [{'package': []}, {'keyview': []}, {'top': []}, {'side': []}, {'side': []}, {'Note': []},{'Table': []}]
    """
    data = package_information
    sigle_dict = {}  # 存放每一页的封装信息
    table_dict = {}
    type = {}
    # 只对当前封装信息进行处理：
    package_coord = [int(i)*3 for i in data['rect']]
    sigle_dict[data['page']] = [{'package': package_coord}]
    table_dict[data['page']] = {}
    type[data['page']] = data['package_type']
    for parts in data['part_content']: # 此处可能需要注意一页之中包含多个类型的封装表
        # 添加名称转换逻辑
        if parts['part_name'] == 'TOPVIEW':
            parts['part_name'] = 'Top'
        elif parts['part_name'] == 'SON' or parts['part_name'] == 'DFN':
            parts['part_name'] = 'DFN_SON'
        elif parts['part_name'] == 'SIDEVIEW':
            parts['part_name'] = 'Side'
        elif parts['part_name'] == 'DETAIL':
            parts['part_name'] = 'Detail'
        # # 只添加与当前页码相同的部件信息
        if parts['page'] == data['page']:
            package_dict = {parts['part_name']: [int(i) * 3 for i in parts['rect']]}
            sigle_dict[data['page']].append(package_dict)

        if parts['part_name'] == 'Form':
            # table_dict[data['page']][parts['page']] = parts['rect']
            if tuple(parts['rect']) in table_dict[data['page']]:
                x = parts['rect']
                table_dict[data['page']][(x[0]-1, x[1],x[2],x[3])] = parts['page']
            else:
                table_dict[data['page']][tuple(parts['rect'])] = parts['page'] #key为坐标，values为表格所在页，这样就可以区分目标封装图匹配多个表的情况
    # print(sigle_dict) # 此处的页数+1才是实际PDF页数，比如说dict内是27，实际PDF页为28
    manage_package = [sigle_dict,table_dict,type]
    # print(manage_package)
    return manage_package

def hist(img, show_img_key):
    # 求出img 的最大最小值
    max_img = np.max(img)
    min_img = np.min(img)
    # 输出最小灰度级和最大灰度级
    o_min, o_max = 0, 255
    # 求 a, b
    a = float(o_max - o_min) / (max_img - min_img)
    b = o_min - a * min_img
    # 线性变换
    o = a * img + b
    o = o.astype(np.uint8)
    if show_img_key == 1:
        cv2.imshow('enhance-0', o)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return o

def set_image_size(filein):
    """
    改变图片大小
    :param filein: 输入图片
    :return:
    """
    max_length = 1000
    img = Image.open(filein)
    if img.size[0] > img.size[1]:  # 限制图片大小避免过大
        width = max_length
        height = int(img.size[1] * max_length / img.size[0])
    else:
        height = max_length
        width = int(img.size[0] * max_length / img.size[1])
    type = img.format
    out = img.resize((width, height), Image.LANCZOS)
    out = hist(out, show_img_key=0)
    out = Image.fromarray(np.uint8(out))
    out.save(filein, type)

def adjust_table_coordinates(current_page,table_info):
    """
    调整表格坐标信息，确保每页有三个表格坐标信息
    :param table_info: 字典，键为页码，值为表格坐标列表
    :return: 调整后的表格坐标信息字典
    """
    adjusted_info = {}
    page_numbers = sorted(table_info.values())
    coords =  table_info.keys()

    page_Number_List = list(table_info.values())
    Table_Coordinate_List =  list(table_info.keys())

    if len(page_numbers) == 3:
        pass
    elif len(page_numbers)==2 and page_numbers[0] != page_numbers[1]:
        if current_page == page_numbers[0]:
            page_Number_List.insert(0,page_numbers[0]-1)
            Table_Coordinate_List.insert(0,())
        elif current_page == page_numbers[1]:
            page_Number_List.append(page_numbers[1]+1)
            Table_Coordinate_List.append(())
    elif len(page_numbers)==2 and page_numbers[0] == page_numbers[1]: # 当前封装图匹配了多个同页表格
        page_Number_List.append(page_numbers[1] + 1)
        Table_Coordinate_List.append(())
    elif len(page_numbers)==1:
        if page_numbers[0] < current_page:
            #说明该表在前一页
            page_Number_List.append(page_numbers[0] + 1)
            Table_Coordinate_List.append(())
            page_Number_List.append(page_numbers[0] +2)
            Table_Coordinate_List.append(())
        elif page_numbers[0] == current_page:
            #该表在当前页
            page_Number_List.append(page_numbers[0] + 1)
            Table_Coordinate_List.append(())
            page_Number_List.insert(0, page_numbers[0] - 1)
            Table_Coordinate_List.insert(0, ())
        elif page_numbers[0] > current_page:
            #说明该表在后一页
            page_Number_List.insert(0,page_numbers[0] - 2)
            Table_Coordinate_List.insert(0,())
            page_Number_List.insert(1, page_numbers[0] - 1)
            Table_Coordinate_List.insert(1, ())



    # if len(page_numbers)==1:
    #     output = {page_numbers[0]-1:[],page_numbers[0]:list(coords[0]),page_numbers[0]+1:[]}
    # elif len(page_numbers)==2:
    #     if current_page == page_numbers[0]:
    #         output = {page_numbers[0]-1:[],page_numbers[0]:table_info[page_numbers[0]],page_numbers[0]+1:table_info[page_numbers[0]+1]}
    #     elif current_page == page_numbers[1]:
    #         output = {page_numbers[0]:table_info[page_numbers[0]],
    #                     page_numbers[1]:table_info[page_numbers[1]],
    #                     page_numbers[1]+1:[]}
    #     else:
    #         output = {page_numbers[0]:table_info[page_numbers[0]],current_page:[],page_numbers[1]:table_info[page_numbers[1]]}

    # 最后需要把页数+1，变为实际页数

    Table_list = []
    page_list = []
    for page in page_Number_List:
        page_list.append(page+1)
    for table in Table_Coordinate_List:
        Table_list.append(list(table))
    return page_list,Table_list
