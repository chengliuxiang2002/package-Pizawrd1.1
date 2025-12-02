#F2.分装图预处理流程模块
import os.path
from PIL import ImageEnhance
import matplotlib
import copy
import matplotlib.pyplot as plt
from package_core.PackageExtract.BGA_Function import f4_pipeline_runner
from package_core.PackageExtract.BGA_Function.Pin_process.QFN.QFN_extract_pins import QFN_extract_pins
from package_core.PackageExtract.BGA_Function.Pin_process.QFP.QFP_extract_pins import QFP_extract_pins
from package_core.PackageExtract.BGA_Function.Pin_process.SON.SON_extract_pins import SON_extract_pins
from package_core.PackageExtract.BGA_Function.Pin_process.SOP.SOP_extract_pins import SOP_extract_pins

matplotlib.use('TkAgg')
# 外部文件
from package_core.Segment.Segment_function import *
from package_core.PackageExtract import QFP_extract
from package_core.PackageExtract import SOP_extract
from package_core.PackageExtract import QFN_extract
from package_core.PackageExtract import SON_extract
from package_core.PackageExtract.BGA_Function.BGA_cal_pin import extract_BGA_PIN

try:
    from package_core.PackageExtract.yolox_onnx_py.model_paths import yolo_model_path, result_path
except ModuleNotFoundError:
    from pathlib import Path
    def yolo_model_path(*parts):
        return str(Path(__file__).resolve().parents[2] / 'model' / 'yolo_model' / Path(*parts))
    def result_path(*parts):
        return str(Path(__file__).resolve().parents[2] / 'Result' / Path(*parts))

# 使用统一的路径管理

BOTTOM_PACKAGE_TYPES = ['BGA','DFN','DFN_SON', 'SON', 'QFP', 'QFN', 'SOP',  'SOP']
DETR_RESULT = result_path('Package_view', 'DETRPage')
SEGMENT_RESULT = result_path('Package_view', 'page')
TEMP_SIDE = yolo_model_path('ExtractPackage', 'side.jpg')
TEMP_BOTTOM = yolo_model_path('ExtractPackage', 'bottom.jpg')
TEMP_TOP = yolo_model_path('ExtractPackage', 'top.jpg')
SEGMENT_SIDE = result_path('Package_view', 'page', 'side.jpg')
SEGMENT_BOTTOM = result_path('Package_view', 'page', 'bottom.jpg')
SEGMENT_TOP = result_path('Package_view', 'page', 'top.jpg')
DETR_IMG = result_path('PDF_extract', 'detr_img')

BGA_TABLE = ['Pitch x (el)', 'Pitch y (e)', 'Number of pins along X', 'Number of pins along Y',
             'Package Height (A)', 'Standoff (A1)', 'Body X (E)', 'Body Y (D)', 'Edge Fillet Radius',
             'Ball Diameter Normal (b)', 'Exclude Pins']
QFN_TABLE = ['Pitch x (el)', 'Pitch y (e)', 'Number of pins along X', 'Number of pins along Y',
             'Package Height (A)', 'Standoff (A1)', 'Pull Back (p)', 'Body X (E)', 'Body Y (D)',
             'Lead style', 'Pin Length (L)', 'Lead width (b)', 'Lead Height (c)', 'Exclude Pins',
             'Thermal X (E2)', 'Thermal Y (D2)']
QFP_TABLE = ['Number of pins along X', 'Number of pins along Y', 'Package Height (A)', 'Standoff (A1)',
             'Span X (E)', 'Span Y (D)', 'Body X (E1)', 'Body Y (D1)', 'Body draft (θ)', 'Edge Fillet radius',
             'Lead Length (L)', 'Lead width (b)', 'Lead Thickness (c)', 'Lead Radius (r)', 'Thermal X (E2)', 'Thermal Y (D2)']
SON_TABLE = ['Pitch (e)', 'Number of pins', 'Package Height (A)', 'Standoff (A1)', 'Pull Back (p)', 'Body X (E)',
             'Body Y (D)', 'Lead style', 'Lead Length (L)', 'Lead width (b)', 'Lead Height (c)', 'Exclude Pins', 'Thermal X (E2)', 'Thermal Y (D2)']


def package_coordinate_process(current_page, package_information):
    """
    此函数用来消除封装图上的表格、长文本信息像素，提取Package各视图
    :param current_page: 当前页图片坐标
    :param package_information: 整理后的封装信息
    :return: 返回干净的只包含封装信息的图片package_img
    """
    img_path = os.path.join(result_path('PDF_extract', 'detr_img'), f'{current_page + 1}.png')
    original_img = Image.open(img_path)
    # # 图像增强
    # enhancer = ImageEnhance.Contrast(original_img)
    # enhance_image = enhancer.enhance(factor=7)
    first_img = np.array(original_img)

    # 获取封装图各部分坐标信息
    class_information = package_information.get(current_page)
    class_dic = {}  # 存储视图中点坐标信息
    package_img = None  # 存储最终裁剪的封装图

    if not os.path.exists(f'{DETR_RESULT}{current_page}'):
        os.makedirs(f'{DETR_RESULT}{current_page}')

    # 1. 处理Form和Note区域（删除像素）
    for item in class_information:
        if 'Form' in item:
            form_coord = item['Form']
            x1, y1 = int(form_coord[0]), int(form_coord[1])
            x2, y2 = int(form_coord[2]), int(form_coord[3])
            for y in range(y1, y2 + 1):
                for x in range(x1, x2 + 1):
                    if 0 <= y < first_img.shape[0] and 0 <= x < first_img.shape[1]:
                        first_img[y, x] = [255, 255, 255]

        elif 'Note' in item:
            note_coord = item['Note']
            x1, y1 = int(note_coord[0]), int(note_coord[1])
            x2, y2 = int(note_coord[2]), int(note_coord[3])
            for y in range(y1, y2 + 1):
                for x in range(x1, x2 + 1):
                    if 0 <= y < first_img.shape[0] and 0 <= x < first_img.shape[1]:
                        first_img[y, x] = [255, 255, 255]

    # # 更新原图为处理后的图像
    original_img = Image.fromarray(first_img)
    # # 显示处理后的图像
    # plt.imshow(original_img)
    # plt.axis('off')  # 关闭坐标轴
    # plt.title("Processed Image")
    # plt.show()

    for item in class_information:
        # 处理封装图主区域
        if 'package' in item:
            person_coord = item['package']
            x1, y1 = int(person_coord[0]), int(person_coord[1])
            x2, y2 = int(person_coord[2]), int(person_coord[3])
            pred_box = (x1, y1, x2 + 30, y2)
            correct_box = jz(pred_box, first_img)
            x11, y11 = int(correct_box[0]), int(correct_box[1])
            x21, y21 = int(correct_box[2]), int(correct_box[3])

            # 裁剪获得封装图
            package_img = original_img.crop((int(x11), int(y11) - 10, int(x21), int(y21)))

        # 处理Side视图（按数量命名）
        elif 'Side' in item:
            # 区域中点坐标
            sx = item['Side'][0] + (item['Side'][2] - item['Side'][0]) // 2
            sy = item['Side'][1] + (item['Side'][3] - item['Side'][1]) // 2
            mix_s = [sx, sy]
            if class_dic.get('side') is None:
                side = original_img.crop(
                    (int(item['Side'][0]) - 20, int(item['Side'][1]) - 10, int(item['Side'][2] + 20),
                     int(item['Side'][3]) + 10))
                class_dic['side'] = mix_s
                cv2.imwrite(f'{DETR_RESULT}{current_page}/side.jpg', np.array(side))
            else:
                class_dic['side1'] = mix_s
                side = original_img.crop(
                    (int(item['Side'][0]) - 20, int(item['Side'][1]) - 10, int(item['Side'][2]) + 20,
                     int(item['Side'][3]) + 10))
                cv2.imwrite(f'{DETR_RESULT}{current_page}/side1.jpg', np.array(side))

        elif 'Detail' in item:
            # 区域中点坐标
            dx = item['Detail'][0] + (item['Detail'][2] - item['Detail'][0]) // 2
            dy = item['Detail'][1] + (item['Detail'][3] - item['Detail'][1]) // 2
            mix_t = [dx, dy]
            class_dic['detailed'] = mix_t
            top = original_img.crop(
                (int(item['Detail'][0]) - 20, int(item['Detail'][1]) - 10, int(item['Detail'][2]) + 20,
                 int(item['Detail'][3]) + 10))
            cv2.imwrite(f'{DETR_RESULT}{current_page}/detailed.jpg', np.array(top))

        # 处理Top视图
        elif 'Top' in item or 'TOPVIEW' in item:
            tx = item['Top'][0] + (item['Top'][2] - item['Top'][0]) // 2
            ty = item['Top'][1] + (item['Top'][3] - item['Top'][1]) // 2
            mix_t = [tx, ty]
            class_dic['top'] = mix_t
            top_img = original_img.crop(
                (int(item['Top'][0]) - 20, int(item['Top'][1]) - 10,
                 int(item['Top'][2]) + 20, int(item['Top'][3]) + 10)
            )
            cv2.imwrite(f'{DETR_RESULT}{current_page}/top.jpg', np.array(top_img))

        # 处理底部视图（BGA/DFN_SON/QFP/QFN）
        else:
            for key in item.keys():
                if key in BOTTOM_PACKAGE_TYPES:
                    bx = item[key][0] + (item[key][2] - item[key][0]) // 2
                    by = item[key][1] + (item[key][3] - item[key][1]) // 2
                    mix_b = [bx, by]
                    class_dic['bottom'] = mix_b

                    bottom_img = original_img.crop(
                        (int(item[key][0]) - 20, int(item[key][1]) - 10,
                         int(item[key][2]) + 20, int(item[key][3]) + 10)
                    )
                    cv2.imwrite(f'{DETR_RESULT}{current_page}/bottom.jpg', np.array(bottom_img))
                    break

    # 调整坐标原点为封装图左上角
    for key, value in class_dic.items():
        value[0] = value[0] - x11
        value[1] = value[1] - y11

    print(class_dic)
    return package_img, class_dic

def segment_package(package_img,class_dic, current_page):
    """
    :param package_img: 只包含封装信息的图片
    :param class_dic:DETR检测框的中点值{'bottom': [892, 227], 'side': [877, 580], 'side1': [281, 617], 'top': [305, 218]}
    :param current_page:当前页
    :return:
    """
    image1 = np.array(package_img)
    # 定义扩充的像素数，例如每个边缘都扩充10个像素
    top, bottom, left, right = 20, 20, 20, 20
    # 使用白色像素扩充边缘
    image = cv2.copyMakeBorder(image1, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    edges, dilated_image = morphological_treatment(image)
    # plt.imshow(dilated_image)
    # plt.show()
    # plt.imshow(edges)
    # plt.show()
    # ret, thresh = cv2.threshold(edges, 200, 255, cv2.THRESH_BINARY)
    result_image = np.zeros_like(image)
    kernel = np.ones((3, 3), np.uint8)
    view_flage = False  # 判断视图中是否有view命名的图片
    while True:
        y, x, found_white_pixel = np_where(edges)
        if found_white_pixel:
            start_x, start_y = y, x  # 选择一个多边形边上的像素点作为起点
            contour = find_contour(edges, start_x, start_y)  # 找到多边形的外轮廓
            processed_list = [[int(str(item[0]).lstrip()), item[1]] for item in contour]
            # 对最外层轮廓进行重写
            for coord in processed_list:
                x, y = coord
                result_image[x, y] = 255
            # 对上列表按照第二个元素从小到大排序
            x_list = sorted(processed_list, key=lambda x: x[1])  # 列排序
            y_list = sorted(processed_list, key=lambda y: y[0])  # 行排序
            x1,x2 = x_list[0][1],x_list[len(x_list) - 1][1]
            y1,y2 = y_list[0][0],y_list[len(y_list) - 1][0]

            area = (x2 - x1) * (y2 - y1)
            if area < 25000:
                # 清空该区域
                for i in range(y1, y2 + 1):
                    for j in range(x1, x2 + 1):
                        edges[i, j] = 0
                continue
            # ----------------------------------------------------
            result_image = cv2.dilate(result_image, kernel, iterations=2)  # 略微膨胀
            gray = cv2.cvtColor(result_image, cv2.COLOR_RGB2GRAY)
            # 应用阈值将图像二值化
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            # 查找轮廓
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(gray)
            # 在掩码图像上绘制所有轮廓，并填充轮廓内部
            cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
            # result = cv2.bitwise_and(image, image, mask=mask)
            white_background = np.ones_like(image) * 255
            # 将原始图像中掩码区域的内容复制到全白背景图像上
            inverse_mask = cv2.bitwise_not(mask)
            white_background[inverse_mask == 255] = [255, 255, 255]
            white_background[mask == 255] = image[mask == 255]
            edges[mask == 255] = [0]  # 清除已切割区域
            img1 = Image.fromarray(white_background)
            crop_img = img1.crop((x1 - 5, y1 - 5, x2 + 5, y2 + 5))
            crop_img = np.array(crop_img)
            # 确保生成保存文件夹
            if not os.path.exists(SEGMENT_RESULT):
                os.makedirs(SEGMENT_RESULT)
            # plt.imshow(crop_img)
            # plt.show()
            # DETR作为分类标准
            name_class = 'view'
            key_list = []
            test_list = ['bottom', 'side', 'side1', 'top']
            for key, value in class_dic.items():
                key_list.append(key)
                if key in test_list:
                    test_list.remove(key)
                if x1 < value[0] < x2 and y1 < value[1] < y2:
                    name_class = key
                    break
                else:
                    continue

            if name_class == 'view':
                view_flage = True
            cv2.imwrite(f'{SEGMENT_RESULT}/{name_class}.jpg', crop_img)  # DETRX分类保存
        else:
            break
    # 处理未命名view情况：
    if view_flage:
        # for name in os.listdir('Package_view/page'):
        if 'bottom.jpg' in os.listdir(SEGMENT_RESULT) and 'top.jpg' in os.listdir(
                SEGMENT_RESULT) and 'side.jpg' in os.listdir(SEGMENT_RESULT):
            pass
        elif 'bottom.jpg' in os.listdir(SEGMENT_RESULT) and 'top.jpg' in os.listdir(
                SEGMENT_RESULT):
            os.rename(f'{SEGMENT_RESULT}/view.jpg', SEGMENT_SIDE)  # side
        elif 'side.jpg' in os.listdir(SEGMENT_RESULT) and 'top.jpg' in os.listdir(
                SEGMENT_RESULT):
            os.rename(f'{SEGMENT_RESULT}/view.jpg', SEGMENT_BOTTOM)  # bottom

        elif 'bottom.jpg' in os.listdir(SEGMENT_RESULT) and 'side.jpg' in os.listdir(
                SEGMENT_RESULT):
            os.rename(f'{SEGMENT_RESULT}/view.jpg', SEGMENT_TOP)  # top

        elif 'side.jpg' in os.listdir(SEGMENT_RESULT) and 'top.jpg' not in os.listdir(
                SEGMENT_RESULT):
            os.rename(f'{SEGMENT_RESULT}/view.jpg', SEGMENT_BOTTOM)  # bottom
    # 添加top视图
    # if 'top.jpg' not in os.listdir('Package_view/page'):
    #     shutil.copy('DETRx_onnx/top.jpg','Package_view/page/top.jpg')
    #     shutil.copy('DETRx_onnx/top.jpg','data/top.jpg')
    if (not os.path.exists(SEGMENT_TOP)) and os.path.exists(
            f'{DETR_RESULT}{current_page}/top.jpg'):
        shutil.move(f'{DETR_RESULT}{current_page}/top.jpg', SEGMENT_TOP)
    elif (not os.path.exists(SEGMENT_TOP)) and not os.path.exists(
            f'{DETR_RESULT}{current_page}/top.jpg'):
        shutil.copy(TEMP_TOP, SEGMENT_TOP)

    if (not os.path.exists(SEGMENT_BOTTOM)) and os.path.exists(
            f'{DETR_RESULT}{current_page}/bottom.jpg'):
        shutil.move(f'{DETR_RESULT}{current_page}/bottom.jpg', SEGMENT_BOTTOM)
    elif (not os.path.exists(SEGMENT_BOTTOM)) and not os.path.exists(
            f'{DETR_RESULT}{current_page}/bottom.jpg'):
        shutil.copy(TEMP_BOTTOM, SEGMENT_BOTTOM)

    # 处理多side情况
    if 'side1.jpg' in os.listdir(SEGMENT_RESULT) and 'side.jpg' in os.listdir(SEGMENT_RESULT):
        side = f'{SEGMENT_RESULT}/side.jpg'
        side1 = f'{SEGMENT_RESULT}/side1.jpg'
        side_combined(side, side1, save_path=SEGMENT_RESULT)
    elif 'side1.jpg' in os.listdir(SEGMENT_RESULT) and 'side.jpg' not in os.listdir(
            SEGMENT_RESULT):
        side = f'{DETR_RESULT}{current_page}/side.jpg'
        side1 = f'{SEGMENT_RESULT}/side1.jpg'
        side_combined(side, side1, save_path=SEGMENT_RESULT)

def package_process(current_page,package_information):
    """
    对封装图片进行分割操作，并保存命名
    :param current_page: 当前页
    :param package_information: 整理后的封装信息
    :return:
    """
    if os.path.exists(SEGMENT_RESULT):
        shutil.rmtree(result_path('Package_view'))

    package_image, class_dic = package_coordinate_process(current_page,package_information)

    # 显示传入的图像
    # plt.imshow(package_image)
    # plt.show()
    # 分割函数
    segment_package(package_image,class_dic, current_page)

    # 处理意外分割情况
    if os.path.exists(f'{DETR_RESULT}{current_page}/top.jpg'):
       shutil.move(f'{DETR_RESULT}{current_page}/top.jpg',SEGMENT_TOP)

    if os.path.exists(f'{DETR_RESULT}{current_page}/bottom.jpg'):
       shutil.move(f'{DETR_RESULT}{current_page}/bottom.jpg',SEGMENT_BOTTOM)

    if ( not os.path.exists(SEGMENT_SIDE) )and os.path.exists(f'{DETR_RESULT}{current_page}/side.jpg'):
        shutil.move(f'{DETR_RESULT}{current_page}/side.jpg', SEGMENT_SIDE)
    elif ( not os.path.exists(SEGMENT_SIDE) )and not os.path.exists(f'{DETR_RESULT}{current_page}/side.jpg'):
        shutil.copy(TEMP_SIDE, SEGMENT_SIDE)
    #
    # # 对分割后的图片按照祁新源要求进行放大处理
    # set_image_size(SEGMENT_BOTTOM)
    # set_image_size(SEGMENT_TOP)
    # set_image_size(SEGMENT_SIDE)


def package_indentify(package_type, current_page):
    """
    进行封装视图的信息提取
    :param package_type: 该封装类型
    :return:
    """
    destination_folder_path = result_path('Package_extract', 'data')
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)
        print(f"文件夹 {destination_folder_path} 已创建")

    # 调用函数进行移动操作
    move_files(SEGMENT_RESULT, destination_folder_path)
    # 按照类型进行封装图信息提取
    if package_type == 'QFP':
        out_put = QFP_extract.extract_package(package_type, current_page)
    elif package_type == 'QFN':
        out_put = QFN_extract.extract_QFN(package_type, current_page)
    elif package_type == 'SOP':
        out_put = SOP_extract.extract_SOP(package_type, current_page)
    elif package_type == 'SON' or package_type == 'DFN' or package_type == 'DFN_SON':
        out_put = SON_extract.extract_SON(package_type, current_page)
    elif package_type == 'BGA':
        out_put = f4_pipeline_runner.run_f4_pipeline(destination_folder_path, package_type)
    else:
        print("未定义的封装类型")
        out_put = []
    return out_put


def extract_BGA_pins():
    # 提取BGA引脚数量
    pin_num_x_serial, pin_num_y_serial, loss_pin,loss_color = extract_BGA_PIN()

    return pin_num_x_serial, pin_num_y_serial, loss_pin,loss_color

def extract_QFP_pins():
    # 提取QFP行列引脚数量
    X, Y = QFP_extract_pins(SEGMENT_BOTTOM)
    return X, Y
def extract_QFN_pins():
    # 提取QFN行列引脚数量
    X, Y = QFN_extract_pins(SEGMENT_BOTTOM)
    return X, Y
def extract_SOP_pins():
    # 提取SOP引脚总数
    sum = SOP_extract_pins(SEGMENT_BOTTOM)
    return sum
def extract_SON_pins():
    # 提取SON引脚总数
    sum = SON_extract_pins(SEGMENT_BOTTOM)
    return sum

# 提供给迪浩的代码借口
def manage_result(out_put, package_type):
    # if len(out_put)==4:
    for out_list in out_put:
        del out_list[0]
    result = []
    for out_list in out_put:
        result1 =  []
        for item in out_list:
            if item == '' or item == '-':
                result1.append(None)
            else:
                result1.append(item)
        result.append(result1)

    record_json = {"pkg_type":None,"parameters":{}}
    # record_json[package_type] = {}
    if package_type == 'QFP':
        for i,key in enumerate(QFP_TABLE):
            record_json["pkg_type"] = package_type
            record_json["parameters"][key] = result[i]
    if package_type == 'QFN':
        for i, key in enumerate(QFN_TABLE):
            record_json["pkg_type"] = package_type
            record_json["parameters"][key] = result[i]
    if package_type == 'SON':
        for i, key in enumerate(SON_TABLE):
            record_json["pkg_type"] = package_type
            record_json["parameters"][key] = result[i]
    if package_type == 'BGA':
        for i, key in enumerate(BGA_TABLE):
            record_json["pkg_type"] = package_type
            record_json["parameters"][key] = result[i]

    return record_json
def reco_package(package_type,current_package,current_page,pdf_path):
    from package_core.Table_Processed import Table_extract

    Table_Coordinate_List = []
    page_Number_List = []
    result = None
    pin_num_x_serial = None
    pin_num_y_serial = None
    # 封装类型
  
    if package_type == 'DFN_SON' or package_type == 'DFN':
        package_type = 'SON'
    # 判断是自动搜索还是手动框选
    # 判断是否走数字流程有两个条件，一个是当前封装信息current_package内无Form；另一个是有Form但不是封装Form，这个就是在识别表格的时候才显示。
    if current_package['part_content'] is not None:
        manage_data = manage_json(current_package)
        package_process(current_page, manage_data[0])  # 分割流程
        exists = any(part['part_name'] == 'Form' for part in current_package['part_content'])
        if exists:
            # 表格提取
            current_table = manage_data[1][current_page]
            page_Number_List, Table_Coordinate_List = adjust_table_coordinates(current_page, current_table)
        else:
            print('数字提取')
            result = package_indentify(package_type, current_page)
    elif current_package['part_content'] is None and current_package['type'] == 'list':  # 说明是自动框表
        # 目前只考虑识别当前框选的表，暂不考虑识别多个框选的表
        Table_Coordinate_List = [[], current_package['rect'], []]
        page_Number_List = [current_page, current_page + 1, current_page + 2]
    elif current_package['part_content'] is None and current_package['type'] == 'img':  # 说明是自动框图
        # 框选图流程存在争议
        pass
    if len(page_Number_List) != 0 and len(Table_Coordinate_List) != 0:
        try:
            # 表格内容提取
            data = Table_extract.extract_table(pdf_path, page_Number_List, Table_Coordinate_List, package_type)
            if package_type == 'BGA':
                # 如果表格类型是BGA,运行数字提取BGA引脚数量
                pin_num_x_serial, pin_num_y_serial, loss_pin, loss_color = extract_BGA_pins()

            # 后续操作只考虑了BGA表格类型
            if package_type == 'QFP':
                if not data:
                    # 走数字提取流程
                    print("-----表格数据提取为空-----")
                    result = package_indentify(package_type, current_page)
                else:
                    result = data
            elif package_type == 'BGA':
                result = data[0:11]
                result[1] = copy.deepcopy(result[0])
                result[10][2] = str(loss_color)
                result[10][1] = str(loss_pin)
                if pin_num_x_serial != None and (result[2][2] == '' or result[2][2] == 0):
                    result[2] = ['', '', pin_num_x_serial, '']
                if pin_num_y_serial != None and (result[3][2] == '' or result[3][2] == 0):
                    result[3] = ['', '', pin_num_y_serial, '']

            elif package_type == 'SON':
                result = data[0:14]

            elif package_type == 'SOP':
                result = data[0:12]

            elif package_type == 'QFN':
                result = data
        except Exception as e:
            print(e)
            # 走数字提取流程
            result = package_indentify(package_type, current_page)
    out_put = manage_result(result, package_type)

    return out_put