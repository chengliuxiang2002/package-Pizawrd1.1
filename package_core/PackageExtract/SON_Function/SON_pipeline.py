from typing import Iterable
from package_core.PackageExtract.function_tool import *
from package_core.PackageExtract.get_pairs_data_present5_test import *
from package_core.PackageExtract import common_pipeline
from package_core.PackageExtract.BGA_Function.pre_extract import (
    other_match_dbnet,
    pin_match_dbnet,
    angle_match_dbnet,
    num_match_dbnet,
    num_direction,
    match_triple_factor,
    targeted_ocr
)

# SON数字提取流程

################################################################################################
def extract_pin_boxes_from_txt(file_path):
    """
    从txt文件中提取引脚框数据

    Args:
        file_path: txt文件路径

    Returns:
        tuple: (pin_box, pin_boxh, pin_boxv)
    """
    # 初始化变量
    pin_boxh = []
    pin_boxv = []

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

            for line in lines:
                line = line.strip()

                # 提取X数据
                if line.startswith('X:'):
                    # 去除'X: '前缀
                    x_data_str = line[2:].strip()
                    # 分割多个框
                    boxes_str = x_data_str.split('],[')

                    for i, box_str in enumerate(boxes_str):
                        # 清理字符串中的括号和空格
                        box_str = box_str.replace('[', '').replace(']', '').strip()
                        # 如果是第一个框且开头有逗号，需要进一步清理
                        if box_str.startswith(','):
                            box_str = box_str[1:]
                        # 分割数字并转换为float
                        coordinates = [float(coord.strip()) for coord in box_str.split(',')]
                        # 添加到pin_boxh
                        pin_boxh.append(coordinates)

                # 提取Y数据
                elif line.startswith('Y:'):
                    # 去除'Y: '前缀
                    y_data_str = line[2:].strip()
                    # 分割多个框
                    boxes_str = y_data_str.split('],[')

                    for box_str in boxes_str:
                        # 清理字符串中的括号和空格
                        box_str = box_str.replace('[', '').replace(']', '').strip()
                        # 如果是第一个框且开头有逗号，需要进一步清理
                        if box_str.startswith(','):
                            box_str = box_str[1:]
                        # 分割数字并转换为float
                        coordinates = [float(coord.strip()) for coord in box_str.split(',')]
                        # 添加到pin_boxv
                        pin_boxv.append(coordinates)

        # 从pin_boxh中提取第一个框作为pin_box
        if pin_boxh:
            pin_box = [pin_boxh[0]]  # 注意：格式化为列表的列表
        else:
            pin_box = []
            print("警告：X数据为空")

        return pin_box, pin_boxh, pin_boxv

    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return [], [], []
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return [], [], []

def extract_side_A_A1_A3(L3):
    side_yolox_num = find_list(L3, "side_yolox_num")
    side_ocr_data = find_list(L3, "side_ocr_data")
    side_dbnet_data = find_list(L3, "side_dbnet_data")
    print("调试信息如下")
    print(f'side_ocr_data:{side_ocr_data}')
    print(f'side_dbnet_data:{side_dbnet_data}')
    side_A, side_A3, side_A1 = extract_sorted_dimensions(side_ocr_data, side_yolox_num)
    return side_A, side_A3, side_A1

def extract_top_D_E(L3, triple_factor):
    top_ocr_data = find_list(L3, "top_ocr_data")
    bottom_ocr_data = find_list(L3, "bottom_ocr_data")
    top_border = find_list(L3, "top_border")
    bottom_border = find_list(L3, "bottom_border")
    top_dbnet_data = find_list(L3, "top_dbnet_data")
    print("调试信息如下")
    print(f'top_ocr_data:{top_ocr_data}')
    print(f'top_border_data:{top_border}')
    top_D, top_E = extract_top_dimensions(top_border, top_ocr_data, triple_factor, 0)
    if (np.all(np.array(top_D) == 0) or np.all(np.array(top_E) == 0)):
        top_D, top_E = extract_top_dimensions(bottom_border, bottom_ocr_data, triple_factor, 1)

    # if(top_D[1] > top_E[1]):
    #     top_D, top_E = top_E, top_D
    return top_D, top_E

def extract_bottom_D2_E2(L3, triple_factor, bottom_D, bottom_E):
    bottom_ocr_data = find_list(L3, "bottom_ocr_data")
    bottom_pad = find_list(L3, "bottom_pad")
    bottom_dbnet_data = find_list(L3, "bottom_dbnet_data")
    print(f'bottom_ocr_data:{bottom_ocr_data}')
    print(f'bottom_pad_data:{bottom_pad}')
    bottom_D2, bottom_E2 = extract_bottom_dimensions(bottom_D, bottom_E, bottom_pad, bottom_ocr_data, triple_factor)

    return bottom_D2, bottom_E2

def extract_bottom_b_L(L3, triple_factor, pin_boxs):
    bottom_ocr_data = find_list(L3, "bottom_ocr_data")
    bottom_dbnet_data = find_list(L3, "bottom_dbnet_data")
    print("调试信息如下")
    print(f'bottom_ocr_data:{bottom_ocr_data}')
    print(f'bottom_dbnet_data:{bottom_dbnet_data}')
    bottom_b, bottom_L = extract_pin_dimensions(pin_boxs, bottom_ocr_data, triple_factor)

    # if(bottom_D2[1] > bottom_E2[1]):
    #     bottom_D2, bottom_E2 = bottom_E2, bottom_D2

    return bottom_b, bottom_L

def extract_bottom_pitch(L3, triple_factor, pin_box):
    bottom_ocr_data = find_list(L3, "bottom_ocr_data")
    bottom_dbnet_data = find_list(L3, "bottom_dbnet_data")
    print(f'bottom_ocr_data:{bottom_ocr_data}')
    print(f'bottom_dbnet_data:{bottom_dbnet_data}')
    bottom_pitch = extract_pitch_dimension(pin_box, bottom_ocr_data, triple_factor)

    return bottom_pitch

def extract_sorted_dimensions(side_ocr_data_list, side_yolox_num):
    """
    处理多个OCR数据，每个YOLO框可能对应不同的OCR数据

    参数:
    side_ocr_data_list: OCR检测数据列表
    side_yolox_num: YOLO检测框数据，维度为[n, 4]

    返回:
    side_A, side_A3, side_A1: 排序后的前3个max_medium_min数组（仅处理中间值<2的）
    """
    # 存储所有匹配的尺寸数组和对应的中间值
    matched_dimensions = []

    if side_yolox_num is None or len(side_yolox_num) == 0:
        return [0, 0, 0], [0, 0, 0], [0, 0, 0]

    for yolo_box in side_yolox_num:
        best_match = None
        best_match_score = float('inf')

        # 为每个YOLO框找到最匹配的OCR数据
        for ocr_data in side_ocr_data_list:
            ocr_location = ocr_data.get('location', None)
            if ocr_location is not None and len(ocr_location) == 4:
                # 计算两个框的距离（中心点距离）
                yolo_center = [(yolo_box[0] + yolo_box[2]) / 2, (yolo_box[1] + yolo_box[3]) / 2]
                ocr_center = [(ocr_location[0] + ocr_location[2]) / 2, (ocr_location[1] + ocr_location[3]) / 2]
                distance = np.sqrt((yolo_center[0] - ocr_center[0]) ** 2 + (yolo_center[1] - ocr_center[1]) ** 2)

                if distance < best_match_score:
                    best_match_score = distance
                    best_match = ocr_data

        # 如果找到匹配的OCR数据，提取其尺寸数组
        if best_match is not None and best_match_score < 10:  # 设置一个阈值
            dimensions = best_match.get('max_medium_min', [])
            if len(dimensions) == 3:
                middle_value = dimensions[1]
                # 只处理中间值小于2的尺寸数组
                if middle_value < 2:
                    matched_dimensions.append((dimensions, middle_value))

    # 初始化输出值
    side_A = [0, 0, 0]
    side_A3 = [0, 0, 0]
    side_A1 = [0, 0, 0]

    # 按中间值从大到小排序并返回前3个
    if matched_dimensions:
        # 按中间值从大到小排序
        sorted_dims = sorted(matched_dimensions, key=lambda x: x[1], reverse=True)

        # 只取前3个完整的尺寸数组
        top_dims = [dim_array for dim_array, _ in sorted_dims[:3]]

        # 如果不足3个，补充[0,0,0]
        while len(top_dims) < 3:
            top_dims.append([0, 0, 0])

        # 分配给输出变量
        if len(top_dims) > 2:
            # 长度大于2时，保持原逻辑
            side_A = list(top_dims[0])
            side_A1 = list(top_dims[2])
            side_A3 = list(top_dims[1])
        else:
            # 长度小于等于2时，按顺序填充
            side_A = list(top_dims[0]) if len(top_dims) > 0 else [0, 0, 0]
            side_A1 = list(top_dims[1]) if len(top_dims) > 1 else [0, 0, 0]
            side_A3 = [0, 0, 0]

    return side_A, side_A3, side_A1


def extract_top_dimensions(border, top_ocr_data_list, triple_factor, key):
    """
    从top视图提取尺寸数据，处理多个OCR数据元素

    参数:
    border: 边界框，格式为[[x1, y1, x2, y2]]
    top_ocr_data_list: OCR检测数据列表，每个元素包含location和max_medium_min
    triple_factor: 嵌套的视图数据
    key: 控制提取'top'还是'bottom'元素

    返回:
    top_D: 水平方向尺寸数组 [最大, 标准, 最小]
    top_E: 竖直方向尺寸数组 [最大, 标准, 最小]
    """

    def extract_top_elements(data):
        """递归提取view_name为'top'或'bottom'的元素"""
        top_elements = []

        if isinstance(data, dict):
            if (key == 0):
                if data.get('view_name') == 'top':
                    top_elements.append(data)
                for value in data.values():
                    if isinstance(value, (dict, list)):
                        top_elements.extend(extract_top_elements(value))
            else:
                if data.get('view_name') == 'bottom':
                    top_elements.append(data)
                for value in data.values():
                    if isinstance(value, (dict, list)):
                        top_elements.extend(extract_top_elements(value))
        elif isinstance(data, list):
            for item in data:
                top_elements.extend(extract_top_elements(item))

        return top_elements

    print("=== extract_top_dimensions 开始执行 ===")

    # 初始化输出值
    top_D = [0, 0, 0]
    top_E = [0, 0, 0]

    # 检查输入数据
    if not top_ocr_data_list or len(top_ocr_data_list) == 0:
        print("警告: top_ocr_data_list为空，返回默认值")
        return top_D, top_E

    print(f"收到 {len(top_ocr_data_list)} 个OCR数据")

    # 提取triple_factor中的所有top元素
    top_elements = extract_top_elements(triple_factor)

    print(f"找到 {len(top_elements)} 个top元素")

    if not top_elements:
        print("警告: 没有找到top元素，使用OCR数据中的标准值排序")
        # 如果没有top元素，从OCR数据中按标准值排序取最大的
        all_max_medium_min = []
        for ocr_data in top_ocr_data_list:
            max_medium_min = ocr_data.get('max_medium_min', [])
            if len(max_medium_min) == 3:
                all_max_medium_min.append(max_medium_min)

        if all_max_medium_min:
            print(f"从 {len(all_max_medium_min)} 个OCR数据中提取max_medium_min")
            # 按标准值(中间值)排序
            all_max_medium_min.sort(key=lambda x: x[1], reverse=True)
            top_D = all_max_medium_min[0].copy()
            top_E = all_max_medium_min[0].copy()
            print(f"使用标准值排序结果: top_D={top_D}, top_E={top_E}")
        else:
            print("没有找到有效的max_medium_min数据")

        return top_D, top_E

    # 将top元素分为两类：有arrow_pairs和没有arrow_pairs的
    top_with_arrow = []
    top_without_arrow = []

    for element in top_elements:
        if element.get('arrow_pairs') is not None:
            top_with_arrow.append(element)
        else:
            top_without_arrow.append(element)

    print(f"有arrow_pairs的top元素: {len(top_with_arrow)} 个")
    print(f"无arrow_pairs的top元素: {len(top_without_arrow)} 个")

    # 为每个OCR数据找到匹配的top元素，创建融合结构B
    all_b_elements = []

    print(f"开始匹配OCR数据和top元素...")
    matched_count = 0

    # 使用更宽松的匹配阈值
    position_tolerance = 2.0  # 位置容差从0.001放宽到2.0

    for ocr_data in top_ocr_data_list:
        ocr_location = ocr_data.get('location', None)
        max_medium_min = ocr_data.get('max_medium_min', [])

        if ocr_location is None or len(ocr_location) != 4:
            continue

        # 确保max_medium_min是列表格式
        if isinstance(max_medium_min, np.ndarray):
            max_medium_min = max_medium_min.tolist()

        # 优先匹配有arrow_pairs的元素
        matched = False
        matched_element = None

        # 首先尝试匹配有arrow_pairs的元素
        for top_element in top_with_arrow:
            element_location = top_element.get('location', None)
            if element_location is not None and len(element_location) == 4:
                # 使用放宽的阈值比较location
                if (abs(ocr_location[0] - element_location[0]) < position_tolerance and
                        abs(ocr_location[1] - element_location[1]) < position_tolerance and
                        abs(ocr_location[2] - element_location[2]) < position_tolerance and
                        abs(ocr_location[3] - element_location[3]) < position_tolerance):
                    matched = True
                    matched_element = top_element
                    print(f"匹配成功(有箭头): OCR位置{ocr_location} 与 top位置{element_location}")
                    break

        # 如果没有匹配到有arrow_pairs的元素，再尝试匹配没有arrow_pairs的元素
        if not matched:
            for top_element in top_without_arrow:
                element_location = top_element.get('location', None)
                if element_location is not None and len(element_location) == 4:
                    # 使用放宽的阈值比较location
                    if (abs(ocr_location[0] - element_location[0]) < position_tolerance and
                            abs(ocr_location[1] - element_location[1]) < position_tolerance and
                            abs(ocr_location[2] - element_location[2]) < position_tolerance and
                            abs(ocr_location[3] - element_location[3]) < position_tolerance):
                        matched = True
                        matched_element = top_element
                        print(f"匹配成功(无箭头): OCR位置{ocr_location} 与 top位置{element_location}")
                        break

        # 如果匹配成功，创建融合结构B
        if matched and matched_element is not None:
            b_element = {
                'location': matched_element['location'],
                'direction': matched_element.get('direction', ''),
                'arrow_pairs': matched_element.get('arrow_pairs', None),
                'max_medium_min': max_medium_min  # 使用OCR的max_medium_min
            }
            all_b_elements.append(b_element)
            matched_count += 1

            # 从原始列表中移除已匹配的元素，避免重复匹配
            if matched_element in top_with_arrow:
                top_with_arrow.remove(matched_element)
            elif matched_element in top_without_arrow:
                top_without_arrow.remove(matched_element)

    print(f"匹配完成，共找到 {matched_count} 个匹配项")

    if not all_b_elements:
        print("警告: 没有找到匹配的B元素，使用OCR数据中的标准值排序")
        # 如果没有匹配的B元素，从OCR数据中按标准值排序取最大的
        all_max_medium_min = []
        for ocr_data in top_ocr_data_list:
            max_medium_min = ocr_data.get('max_medium_min', [])
            if len(max_medium_min) == 3:
                all_max_medium_min.append(max_medium_min)

        if all_max_medium_min:
            print(f"从 {len(all_max_medium_min)} 个OCR数据中提取max_medium_min")
            # 按标准值(中间值)排序
            all_max_medium_min.sort(key=lambda x: x[1], reverse=True)
            top_D = all_max_medium_min[0].copy()
            top_E = all_max_medium_min[0].copy()
            print(f"使用标准值排序结果: top_D={top_D}, top_E={top_E}")
        else:
            print("没有找到有效的max_medium_min数据")

        return top_D, top_E

    # 计算border的长宽
    border_width = 0
    border_height = 0
    if border is not None and len(border) > 0:
        try:
            border_box = border[0]
            border_width = abs(float(border_box[2]) - float(border_box[0]))  # x2 - x1
            border_height = abs(float(border_box[3]) - float(border_box[1]))  # y2 - y1
            print(f"border尺寸: 宽度={border_width:.2f}, 高度={border_height:.2f}")
        except Exception as e:
            print(f"错误: 计算border尺寸时出错: {e}")
            border_width = 0
            border_height = 0
    else:
        print("警告: border为空或无效")

    # 按照标准值(中间值)对all_b_elements排序
    all_b_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0, reverse=True)
    print(f"按标准值排序后，前3个B元素的max_medium_min: {[b['max_medium_min'] for b in all_b_elements[:3]]}")

    # 如果没有border或border尺寸无效，使用标准值排序方法
    if border_width == 0 or border_height == 0:
        print("警告: border尺寸无效，使用标准值排序方法")
        # 分别收集水平和竖直方向的元素
        horizontal_elements = []
        vertical_elements = []

        for element in all_b_elements:
            direction = element.get('direction', '').lower()

            # 根据direction判断方向
            if direction in ['horizontal', 'up', 'down']:  # 水平方向：up和down
                horizontal_elements.append(element)
            elif direction in ['vertical', 'left', 'right']:  # 竖直方向：left和right
                vertical_elements.append(element)
            else:
                # 方向未知，两个方向都考虑
                horizontal_elements.append(element)
                vertical_elements.append(element)

        print(f"水平方向元素: {len(horizontal_elements)} 个")
        print(f"竖直方向元素: {len(vertical_elements)} 个")

        # 获取每个方向的最大标准值元素
        if horizontal_elements:
            horizontal_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0,
                                     reverse=True)
            top_D = horizontal_elements[0]['max_medium_min'].copy()
            print(f"水平方向选择: max_medium_min={top_D}")
        else:
            top_D = all_b_elements[0]['max_medium_min'].copy()
            print(f"水平方向无指定元素，使用第一个: max_medium_min={top_D}")

        if vertical_elements:
            vertical_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0,
                                   reverse=True)
            top_E = vertical_elements[0]['max_medium_min'].copy()
            print(f"竖直方向选择: max_medium_min={top_E}")
        else:
            top_E = all_b_elements[0]['max_medium_min'].copy()
            print(f"竖直方向无指定元素，使用第一个: max_medium_min={top_E}")

        return top_D, top_E

    # 有有效的border，进行比对
    print("开始与border尺寸进行比对...")
    best_horizontal_match = None
    best_vertical_match = None
    min_horizontal_diff = float('inf')
    min_vertical_diff = float('inf')

    # 优先考虑有arrow_pairs的元素进行border匹配
    for idx, element in enumerate(all_b_elements):
        direction = element.get('direction', '').lower()
        arrow_pairs = element.get('arrow_pairs', None)

        # 对于没有arrow_pairs的元素，跳过border匹配
        if arrow_pairs is None or len(arrow_pairs) == 0:
            continue

        # 获取最后一位（引线之间距离）
        try:
            arrow_distance = float(arrow_pairs[-1])
        except Exception as e:
            continue

        # 计算与border尺寸的差异
        horizontal_diff = abs(arrow_distance - border_width)
        vertical_diff = abs(arrow_distance - border_height)

        print(f"元素{idx}(有箭头): 方向={direction}, 箭头距离={arrow_distance:.2f}, "
              f"水平差异={horizontal_diff:.2f}, 垂直差异={vertical_diff:.2f}")

        # 根据direction确定主要方向
        if direction in ['horizontal', 'up', 'down']:  # 水平方向
            if horizontal_diff < min_horizontal_diff:
                min_horizontal_diff = horizontal_diff
                best_horizontal_match = element
                print(f"  更新水平最佳匹配: 差异={horizontal_diff:.2f}")
        elif direction in ['vertical', 'left', 'right']:  # 竖直方向
            if vertical_diff < min_vertical_diff:
                min_vertical_diff = vertical_diff
                best_vertical_match = element
                print(f"  更新竖直最佳匹配: 差异={vertical_diff:.2f}")
        else:
            # 方向未知，根据差异最小值决定方向
            if horizontal_diff < vertical_diff and horizontal_diff < min_horizontal_diff:
                min_horizontal_diff = horizontal_diff
                best_horizontal_match = element
                print(f"  更新水平最佳匹配(自动判断): 差异={horizontal_diff:.2f}")
            elif vertical_diff < horizontal_diff and vertical_diff < min_vertical_diff:
                min_vertical_diff = vertical_diff
                best_vertical_match = element
                print(f"  更新竖直最佳匹配(自动判断): 差异={vertical_diff:.2f}")

    # 如果通过有arrow_pairs的元素没有找到匹配，再考虑没有arrow_pairs的元素
    if best_horizontal_match is None or best_vertical_match is None:
        print("通过有arrow_pairs的元素未找到足够匹配，考虑无arrow_pairs的元素...")
        for idx, element in enumerate(all_b_elements):
            # 跳过已经有arrow_pairs的元素（已经处理过）
            if element.get('arrow_pairs') is not None:
                continue

            direction = element.get('direction', '').lower()
            max_medium_min = element.get('max_medium_min', [])

            if len(max_medium_min) < 2:
                continue

            std_value = max_medium_min[1]  # 标准值

            # 计算与border尺寸的差异
            horizontal_diff = abs(std_value - border_width)
            vertical_diff = abs(std_value - border_height)

            print(f"元素{idx}(无箭头): 方向={direction}, 标准值={std_value:.2f}, "
                  f"水平差异={horizontal_diff:.2f}, 垂直差异={vertical_diff:.2f}")

            # 根据direction确定主要方向
            if direction in ['horizontal', 'up', 'down']:  # 水平方向
                if horizontal_diff < min_horizontal_diff:
                    min_horizontal_diff = horizontal_diff
                    best_horizontal_match = element
                    print(f"  更新水平最佳匹配: 差异={horizontal_diff:.2f}")
            elif direction in ['vertical', 'left', 'right']:  # 竖直方向
                if vertical_diff < min_vertical_diff:
                    min_vertical_diff = vertical_diff
                    best_vertical_match = element
                    print(f"  更新竖直最佳匹配: 差异={vertical_diff:.2f}")
            else:
                # 方向未知，根据差异最小值决定方向
                if horizontal_diff < vertical_diff and horizontal_diff < min_horizontal_diff:
                    min_horizontal_diff = horizontal_diff
                    best_horizontal_match = element
                    print(f"  更新水平最佳匹配(自动判断): 差异={horizontal_diff:.2f}")
                elif vertical_diff < horizontal_diff and vertical_diff < min_vertical_diff:
                    min_vertical_diff = vertical_diff
                    best_vertical_match = element
                    print(f"  更新竖直最佳匹配(自动判断): 差异={vertical_diff:.2f}")

    # 使用阈值判断是否"很相似"
    similarity_threshold = 0.2  # 从10%放宽到20%的误差
    border_width_threshold = border_width * similarity_threshold
    border_height_threshold = border_height * similarity_threshold

    print(f"\n相似性阈值: 水平={border_width_threshold:.2f}, 竖直={border_height_threshold:.2f}")

    # 判断水平方向是否有匹配
    if best_horizontal_match is not None and min_horizontal_diff <= border_width_threshold:
        top_D = best_horizontal_match['max_medium_min'].copy()
        has_arrow = best_horizontal_match.get('arrow_pairs') is not None
        print(
            f"水平方向找到{'有箭头' if has_arrow else '无箭头'}相似匹配: max_medium_min={top_D}, 差异={min_horizontal_diff:.2f}")
    else:
        # 没有匹配，使用标准值排序
        print(f'水平无相似匹配, 最小差异={min_horizontal_diff:.2f}, 阈值={border_width_threshold:.2f}')
        # 从all_b_elements中按标准值排序，取最大的水平方向元素或第一个元素
        horizontal_elements = [e for e in all_b_elements
                               if e.get('direction', '').lower() in ['horizontal', 'up', 'down']]
        if horizontal_elements:
            horizontal_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0,
                                     reverse=True)
            top_D = horizontal_elements[0]['max_medium_min'].copy()
            print(f"水平方向使用标准值排序: max_medium_min={top_D}")
        else:
            # 使用排序后第一个元素的max_medium_min
            top_D = all_b_elements[0]['max_medium_min'].copy()
            print(f"水平方向使用第一个元素: max_medium_min={top_D}")

    # 判断竖直方向是否有匹配
    if best_vertical_match is not None and min_vertical_diff <= border_height_threshold:
        top_E = best_vertical_match['max_medium_min'].copy()
        has_arrow = best_vertical_match.get('arrow_pairs') is not None
        print(
            f"竖直方向找到{'有箭头' if has_arrow else '无箭头'}相似匹配: max_medium_min={top_E}, 差异={min_vertical_diff:.2f}")
    else:
        # 没有匹配，使用标准值排序
        print(f'竖直无相似匹配, 最小差异={min_vertical_diff:.2f}, 阈值={border_height_threshold:.2f}')
        # 从all_b_elements中按标准值排序，取最大的竖直方向元素或第二个元素
        vertical_elements = [e for e in all_b_elements
                             if e.get('direction', '').lower() in ['vertical', 'left', 'right']]
        if vertical_elements:
            vertical_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0,
                                   reverse=True)
            top_E = vertical_elements[0]['max_medium_min'].copy()
            print(f"竖直方向使用标准值排序: max_medium_min={top_E}")
        else:
            # 使用排序后第二个元素的max_medium_min（如果存在）
            if len(all_b_elements) > 1:
                top_E = all_b_elements[1]['max_medium_min'].copy()
                print(f"竖直方向使用第二个元素: max_medium_min={top_E}")
            else:
                # 如果只有一个元素，使用同一个元素的max_medium_min
                top_E = all_b_elements[0]['max_medium_min'].copy()
                print(f"竖直方向使用第一个元素: max_medium_min={top_E}")

    print(f"\n最终结果: top_D={top_D}, top_E={top_E}")
    print("=== extract_top_dimensions 执行结束 ===\n")

    return top_D, top_E



def extract_bottom_dimensions(bottom_D, bottom_E, pad, bottom_ocr_data_list, triple_factor):
    """
    从bottom视图提取尺寸数据，处理多个OCR数据元素

    参数:
    bottom_D: 水平方向尺寸数组 [最大, 标准, 最小]
    bottom_E: 竖直方向尺寸数组 [最大, 标准, 最小]
    pad: 边界框，格式为[[x1, y1, x2, y2]]
    bottom_ocr_data_list: OCR检测数据列表，每个元素包含location和max_medium_min
    triple_factor: 嵌套的视图数据

    返回:
    bottom_D2: 水平方向尺寸数组 [最大, 标准, 最小]
    bottom_E2: 竖直方向尺寸数组 [最大, 标准, 最小]
    """

    def extract_bottom_elements(data):
        """递归提取view_name为'bottom'的元素"""
        bottom_elements = []

        if isinstance(data, dict):
            if data.get('view_name') == 'bottom':
                bottom_elements.append(data)
            for value in data.values():
                if isinstance(value, (dict, list)):
                    bottom_elements.extend(extract_bottom_elements(value))
        elif isinstance(data, list):
            for item in data:
                bottom_elements.extend(extract_bottom_elements(item))

        return bottom_elements

    print("=== extract_bottom_dimensions 开始执行 ===")

    # 初始化输出值
    bottom_D2 = [0, 0, 0]
    bottom_E2 = [0, 0, 0]

    # 检查pad是否存在
    if pad is None or len(pad) == 0:
        print("警告: pad为空，返回默认值[0,0,0]")
        return bottom_D2, bottom_E2

    print(f"输入参数: bottom_D={bottom_D}, bottom_E={bottom_E}")
    print(f"pad: {pad}")

    # 检查输入数据
    if not bottom_ocr_data_list or len(bottom_ocr_data_list) == 0:
        print("警告: bottom_ocr_data_list为空，返回默认值")
        return bottom_D2, bottom_E2

    print(f"收到 {len(bottom_ocr_data_list)} 个bottom OCR数据")

    # 提取triple_factor中的所有bottom元素
    bottom_elements = extract_bottom_elements(triple_factor)

    print(f"找到 {len(bottom_elements)} 个bottom元素")

    if not bottom_elements:
        print("警告: 没有找到bottom元素，返回默认值[0,0,0]")
        return bottom_D2, bottom_E2

    # 将bottom元素分为两类：有arrow_pairs和没有arrow_pairs的
    bottom_with_arrow = []
    bottom_without_arrow = []

    for element in bottom_elements:
        if element.get('arrow_pairs') is not None:
            bottom_with_arrow.append(element)
        else:
            bottom_without_arrow.append(element)

    print(f"有arrow_pairs的bottom元素: {len(bottom_with_arrow)} 个")
    print(f"无arrow_pairs的bottom元素: {len(bottom_without_arrow)} 个")

    # 为每个OCR数据找到匹配的bottom元素，创建融合结构B
    all_b_elements = []

    print(f"开始匹配OCR数据和bottom元素...")
    matched_count = 0

    # 使用更宽松的匹配阈值
    position_tolerance = 2.0  # 位置容差从0.001放宽到2.0

    for ocr_data in bottom_ocr_data_list:
        ocr_location = ocr_data.get('location', None)
        max_medium_min = ocr_data.get('max_medium_min', [])

        if ocr_location is None or len(ocr_location) != 4:
            continue

        # 确保max_medium_min是列表格式
        if isinstance(max_medium_min, np.ndarray):
            max_medium_min = max_medium_min.tolist()

        # 优先匹配有arrow_pairs的元素
        matched = False
        matched_element = None

        # 首先尝试匹配有arrow_pairs的元素
        for bottom_element in bottom_with_arrow:
            element_location = bottom_element.get('location', None)
            if element_location is not None and len(element_location) == 4:
                # 使用放宽的阈值比较location
                if (abs(ocr_location[0] - element_location[0]) < position_tolerance and
                        abs(ocr_location[1] - element_location[1]) < position_tolerance and
                        abs(ocr_location[2] - element_location[2]) < position_tolerance and
                        abs(ocr_location[3] - element_location[3]) < position_tolerance):
                    matched = True
                    matched_element = bottom_element
                    print(f"匹配成功(有箭头): OCR位置{ocr_location} 与 bottom位置{element_location}")
                    break

        # 如果没有匹配到有arrow_pairs的元素，再尝试匹配没有arrow_pairs的元素
        if not matched:
            for bottom_element in bottom_without_arrow:
                element_location = bottom_element.get('location', None)
                if element_location is not None and len(element_location) == 4:
                    # 使用放宽的阈值比较location
                    if (abs(ocr_location[0] - element_location[0]) < position_tolerance and
                            abs(ocr_location[1] - element_location[1]) < position_tolerance and
                            abs(ocr_location[2] - element_location[2]) < position_tolerance and
                            abs(ocr_location[3] - element_location[3]) < position_tolerance):
                        matched = True
                        matched_element = bottom_element
                        print(f"匹配成功(无箭头): OCR位置{ocr_location} 与 bottom位置{element_location}")
                        break

        # 如果匹配成功，创建融合结构B
        if matched and matched_element is not None:
            b_element = {
                'location': matched_element['location'],
                'direction': matched_element.get('direction', ''),
                'arrow_pairs': matched_element.get('arrow_pairs', None),
                'max_medium_min': max_medium_min  # 使用OCR的max_medium_min
            }
            all_b_elements.append(b_element)
            matched_count += 1

            # 从原始列表中移除已匹配的元素，避免重复匹配
            if matched_element in bottom_with_arrow:
                bottom_with_arrow.remove(matched_element)
            elif matched_element in bottom_without_arrow:
                bottom_without_arrow.remove(matched_element)

    print(f"匹配完成，共找到 {matched_count} 个匹配项")

    if not all_b_elements:
        print("警告: 没有找到匹配的B元素，返回默认值[0,0,0]")
        return bottom_D2, bottom_E2

    # 计算pad的长宽
    pad_width = 0
    pad_height = 0
    if pad is not None and len(pad) >= 1:
        try:
            if len(pad) == 1:
                # 单个边界框的情况
                pad_box = pad[0]
                pad_width = abs(float(pad_box[2]) - float(pad_box[0]))  # x2 - x1
                pad_height = abs(float(pad_box[3]) - float(pad_box[1]))  # y2 - y1
            else:
                # 多个边界框的情况：计算总体包围盒
                # 初始化最小值和最大值
                x_min = float('inf')
                y_min = float('inf')
                x_max = float('-inf')
                y_max = float('-inf')

                # 遍历所有边界框，找到总体的最小和最大坐标
                for box in pad:
                    if len(box) >= 4:
                        x1 = float(box[0])
                        y1 = float(box[1])
                        x2 = float(box[2])
                        y2 = float(box[3])

                        x_min = min(x_min, x1, x2)
                        y_min = min(y_min, y1, y2)
                        x_max = max(x_max, x1, x2)
                        y_max = max(y_max, y1, y2)

                # 计算总体尺寸
                pad_width = x_max - x_min
                pad_height = y_max - y_min

            print(f"pad尺寸: 宽度={pad_width:.2f}, 高度={pad_height:.2f}")

        except Exception as e:
            print(f"错误: 计算pad尺寸时出错: {e}")
            pad_width = 0
            pad_height = 0
    else:
        pad_width = 0
        pad_height = 0

    # 按照标准值(中间值)对all_b_elements排序（降序）
    all_b_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0, reverse=True)
    print(f"按标准值排序后，所有B元素的max_medium_min: {[b['max_medium_min'] for b in all_b_elements]}")

    # 记录是否通过引线找到匹配
    horizontal_matched_by_arrow = False
    vertical_matched_by_arrow = False

    # 如果没有有效的pad尺寸，使用标准值排序方法
    if pad_width == 0 or pad_height == 0:
        print("警告: pad尺寸无效，使用标准值排序方法")
        # 分别收集水平和竖直方向的元素
        horizontal_elements = []
        vertical_elements = []

        for element in all_b_elements:
            direction = element.get('direction', '').lower()

            # 根据direction判断方向
            if direction in ['horizontal', 'up', 'down']:  # 水平方向：up和down
                horizontal_elements.append(element)
            elif direction in ['vertical', 'left', 'right']:  # 竖直方向：left和right
                vertical_elements.append(element)
            else:
                # 方向未知，两个方向都考虑
                horizontal_elements.append(element)
                vertical_elements.append(element)

        print(f"水平方向元素: {len(horizontal_elements)} 个")
        print(f"竖直方向元素: {len(vertical_elements)} 个")

        # 获取每个方向的最大标准值元素，但需要跳过与输入参数相同的值
        if horizontal_elements:
            horizontal_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0,
                                     reverse=True)
            # 寻找第一个与bottom_D不同的元素
            for element in horizontal_elements:
                candidate = element['max_medium_min'].copy()
                if not np.array_equal(candidate, bottom_D):
                    bottom_D2 = candidate
                    print(f"水平方向选择: max_medium_min={bottom_D2}")
                    break
            else:
                # 如果没有找到不同的元素，使用最大值
                bottom_D2 = horizontal_elements[0]['max_medium_min'].copy()
                print(f"水平方向所有元素都与bottom_D相同，使用最大值: max_medium_min={bottom_D2}")
        else:
            # 从所有元素中找与bottom_D不同的最大值
            for element in all_b_elements:
                candidate = element['max_medium_min'].copy()
                if not np.array_equal(candidate, bottom_D):
                    bottom_D2 = candidate
                    print(f"水平方向无指定元素，使用与bottom_D不同的第一个元素: max_medium_min={bottom_D2}")
                    break
            else:
                print("水平方向没有与bottom_D不同的元素，返回[0,0,0]")
                bottom_D2 = [0, 0, 0]

        if vertical_elements:
            vertical_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0,
                                   reverse=True)
            # 寻找第一个与bottom_E不同的元素
            for element in vertical_elements:
                candidate = element['max_medium_min'].copy()
                if not np.array_equal(candidate, bottom_E):
                    bottom_E2 = candidate
                    print(f"竖直方向选择: max_medium_min={bottom_E2}")
                    break
            else:
                # 如果没有找到不同的元素，使用最大值
                bottom_E2 = vertical_elements[0]['max_medium_min'].copy()
                print(f"竖直方向所有元素都与bottom_E相同，使用最大值: max_medium_min={bottom_E2}")
        else:
            # 从所有元素中找与bottom_E不同的最大值
            for element in all_b_elements:
                candidate = element['max_medium_min'].copy()
                if not np.array_equal(candidate, bottom_E):
                    bottom_E2 = candidate
                    print(f"竖直方向无指定元素，使用与bottom_E不同的第一个元素: max_medium_min={bottom_E2}")
                    break
            else:
                print("竖直方向没有与bottom_E不同的元素，返回[0,0,0]")
                bottom_E2 = [0, 0, 0]

        return bottom_D2, bottom_E2

    # 开始与pad尺寸进行比对
    print("开始与pad尺寸进行比对...")
    best_horizontal_match = None
    best_vertical_match = None
    min_horizontal_diff = float('inf')
    min_vertical_diff = float('inf')

    # 优先考虑有arrow_pairs的元素进行pad匹配
    for idx, element in enumerate(all_b_elements):
        direction = element.get('direction', '').lower()
        arrow_pairs = element.get('arrow_pairs', None)

        # 对于没有arrow_pairs的元素，先跳过
        if arrow_pairs is None or len(arrow_pairs) == 0:
            continue

        # 获取最后一位（引线之间距离）
        try:
            arrow_distance = float(arrow_pairs[-1])
        except Exception as e:
            continue

        # 计算与pad尺寸的差异
        horizontal_diff = abs(arrow_distance - pad_width)
        vertical_diff = abs(arrow_distance - pad_height)

        print(f"元素{idx}(有箭头): 方向={direction}, 箭头距离={arrow_distance:.2f}, "
              f"水平差异={horizontal_diff:.2f}, 垂直差异={vertical_diff:.2f}")

        # 根据direction确定主要方向
        if direction in ['horizontal', 'up', 'down']:  # 水平方向
            if horizontal_diff < min_horizontal_diff:
                min_horizontal_diff = horizontal_diff
                best_horizontal_match = element
                print(f"  更新水平最佳匹配: 差异={horizontal_diff:.2f}")
        elif direction in ['vertical', 'left', 'right']:  # 竖直方向
            if vertical_diff < min_vertical_diff:
                min_vertical_diff = vertical_diff
                best_vertical_match = element
                print(f"  更新竖直最佳匹配: 差异={vertical_diff:.2f}")
        else:
            # 方向未知，根据差异最小值决定方向
            if horizontal_diff < vertical_diff and horizontal_diff < min_horizontal_diff:
                min_horizontal_diff = horizontal_diff
                best_horizontal_match = element
                print(f"  更新水平最佳匹配(自动判断): 差异={horizontal_diff:.2f}")
            elif vertical_diff < horizontal_diff and vertical_diff < min_vertical_diff:
                min_vertical_diff = vertical_diff
                best_vertical_match = element
                print(f"  更新竖直最佳匹配(自动判断): 差异={vertical_diff:.2f}")

    # 如果通过有arrow_pairs的元素没有找到匹配，再考虑没有arrow_pairs的元素
    if best_horizontal_match is None or best_vertical_match is None:
        print("通过有arrow_pairs的元素未找到足够匹配，考虑无arrow_pairs的元素...")
        for idx, element in enumerate(all_b_elements):
            # 跳过已经有arrow_pairs的元素（已经处理过）
            if element.get('arrow_pairs') is not None:
                continue

            direction = element.get('direction', '').lower()
            max_medium_min = element.get('max_medium_min', [])

            if len(max_medium_min) < 2:
                continue

            std_value = max_medium_min[1]  # 标准值

            # 计算与pad尺寸的差异
            horizontal_diff = abs(std_value - pad_width)
            vertical_diff = abs(std_value - pad_height)

            print(f"元素{idx}(无箭头): 方向={direction}, 标准值={std_value:.2f}, "
                  f"水平差异={horizontal_diff:.2f}, 垂直差异={vertical_diff:.2f}")

            # 根据direction确定主要方向
            if direction in ['horizontal', 'up', 'down']:  # 水平方向
                if horizontal_diff < min_horizontal_diff:
                    min_horizontal_diff = horizontal_diff
                    best_horizontal_match = element
                    print(f"  更新水平最佳匹配: 差异={horizontal_diff:.2f}")
            elif direction in ['vertical', 'left', 'right']:  # 竖直方向
                if vertical_diff < min_vertical_diff:
                    min_vertical_diff = vertical_diff
                    best_vertical_match = element
                    print(f"  更新竖直最佳匹配: 差异={vertical_diff:.2f}")
            else:
                # 方向未知，根据差异最小值决定方向
                if horizontal_diff < vertical_diff and horizontal_diff < min_horizontal_diff:
                    min_horizontal_diff = horizontal_diff
                    best_horizontal_match = element
                    print(f"  更新水平最佳匹配(自动判断): 差异={horizontal_diff:.2f}")
                elif vertical_diff < horizontal_diff and vertical_diff < min_vertical_diff:
                    min_vertical_diff = vertical_diff
                    best_vertical_match = element
                    print(f"  更新竖直最佳匹配(自动判断): 差异={vertical_diff:.2f}")

    # 使用阈值判断是否"很相似"
    similarity_threshold = 0.2  # 从10%放宽到20%的误差
    pad_width_threshold = pad_width * similarity_threshold
    pad_height_threshold = pad_height * similarity_threshold

    print(f"\n相似性阈值: 水平={pad_width_threshold:.2f}, 竖直={pad_height_threshold:.2f}")

    # 判断水平方向是否有匹配
    if best_horizontal_match is not None and min_horizontal_diff <= pad_width_threshold:
        candidate = best_horizontal_match['max_medium_min'].copy()
        # 检查是否与bottom_D相同
        if not np.array_equal(candidate, bottom_D):
            bottom_D2 = candidate
            has_arrow = best_horizontal_match.get('arrow_pairs') is not None
            horizontal_matched_by_arrow = has_arrow  # 记录是否通过引线找到
            print(
                f"水平方向找到{'有箭头' if has_arrow else '无箭头'}相似匹配: max_medium_min={bottom_D2}, 差异={min_horizontal_diff:.2f}")
        else:
            print(f"水平方向找到相似匹配，但与bottom_D相同，跳过该匹配")
            # 继续寻找其他匹配
            best_horizontal_match = None
            horizontal_matched_by_arrow = False

    # 如果水平方向没有匹配或匹配值与bottom_D相同
    if best_horizontal_match is None or np.array_equal(bottom_D2, [0, 0, 0]):
        print(f'水平无有效相似匹配, 最小差异={min_horizontal_diff:.2f}, 阈值={pad_width_threshold:.2f}')
        # 从all_b_elements中按标准值排序，寻找与bottom_D不同的元素
        horizontal_elements = [e for e in all_b_elements
                               if e.get('direction', '').lower() in ['horizontal', 'up', 'down']]
        if horizontal_elements:
            horizontal_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0,
                                     reverse=True)
            # 寻找第一个与bottom_D不同的元素
            for element in horizontal_elements:
                candidate = element['max_medium_min'].copy()
                if not np.array_equal(candidate, bottom_D):
                    bottom_D2 = candidate
                    print(f"水平方向使用标准值排序且与bottom_D不同的元素: max_medium_min={bottom_D2}")
                    break
            else:
                # 如果所有候选都与bottom_D相同，则从所有元素中找与bottom_D不同的元素
                print("水平方向所有候选都与bottom_D相同，从所有元素中寻找")
                for element in all_b_elements:
                    candidate = element['max_medium_min'].copy()
                    if not np.array_equal(candidate, bottom_D):
                        bottom_D2 = candidate
                        print(f"水平方向使用所有元素中与bottom_D不同的元素: max_medium_min={bottom_D2}")
                        break
                else:
                    print("水平方向所有元素都与bottom_D相同，返回[0,0,0]")
                    bottom_D2 = [0, 0, 0]
        else:
            # 从所有元素中寻找与bottom_D不同的元素
            for element in all_b_elements:
                candidate = element['max_medium_min'].copy()
                if not np.array_equal(candidate, bottom_D):
                    bottom_D2 = candidate
                    print(f"水平方向使用与bottom_D不同的第一个元素: max_medium_min={bottom_D2}")
                    break
            else:
                print("水平方向没有与bottom_D不同的元素，返回[0,0,0]")
                bottom_D2 = [0, 0, 0]

    # 判断竖直方向是否有匹配
    if best_vertical_match is not None and min_vertical_diff <= pad_height_threshold:
        candidate = best_vertical_match['max_medium_min'].copy()
        # 检查是否与bottom_E相同
        if not np.array_equal(candidate, bottom_E):
            bottom_E2 = candidate
            has_arrow = best_vertical_match.get('arrow_pairs') is not None
            vertical_matched_by_arrow = has_arrow  # 记录是否通过引线找到
            print(
                f"竖直方向找到{'有箭头' if has_arrow else '无箭头'}相似匹配: max_medium_min={bottom_E2}, 差异={min_vertical_diff:.2f}")
        else:
            print(f"竖直方向找到相似匹配，但与bottom_E相同，跳过该匹配")
            # 继续寻找其他匹配
            best_vertical_match = None
            vertical_matched_by_arrow = False

    # 如果竖直方向没有匹配或匹配值与bottom_E相同
    if best_vertical_match is None or np.array_equal(bottom_E2, [0, 0, 0]):
        print(f'竖直无有效相似匹配, 最小差异={min_vertical_diff:.2f}, 阈值={pad_height_threshold:.2f}')
        # 从all_b_elements中按标准值排序，寻找与bottom_E不同的元素
        vertical_elements = [e for e in all_b_elements
                             if e.get('direction', '').lower() in ['vertical', 'left', 'right']]
        if vertical_elements:
            vertical_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0,
                                   reverse=True)
            # 寻找第一个与bottom_E不同的元素
            for element in vertical_elements:
                candidate = element['max_medium_min'].copy()
                if not np.array_equal(candidate, bottom_E):
                    bottom_E2 = candidate
                    print(f"竖直方向使用标准值排序且与bottom_E不同的元素: max_medium_min={bottom_E2}")
                    break
            else:
                # 如果所有候选都与bottom_E相同，则从所有元素中找与bottom_E不同的元素
                print("竖直方向所有候选都与bottom_E相同，从所有元素中寻找")
                for element in all_b_elements:
                    candidate = element['max_medium_min'].copy()
                    if not np.array_equal(candidate, bottom_E):
                        bottom_E2 = candidate
                        print(f"竖直方向使用所有元素中与bottom_E不同的元素: max_medium_min={bottom_E2}")
                        break
                else:
                    print("竖直方向所有元素都与bottom_E相同，返回[0,0,0]")
                    bottom_E2 = [0, 0, 0]
        else:
            # 从所有元素中寻找与bottom_E不同的元素
            for element in all_b_elements:
                candidate = element['max_medium_min'].copy()
                if not np.array_equal(candidate, bottom_E):
                    bottom_E2 = candidate
                    print(f"竖直方向使用与bottom_E不同的第一个元素: max_medium_min={bottom_E2}")
                    break
            else:
                print("竖直方向没有与bottom_E不同的元素，返回[0,0,0]")
                bottom_E2 = [0, 0, 0]

    # 应用新规则：如果一边通过引线找到匹配，另一边没有，则没有的一方使用找到引线一方的值
    print(
        f"\n匹配状态: 水平方向通过引线匹配={horizontal_matched_by_arrow}, 竖直方向通过引线匹配={vertical_matched_by_arrow}")

    if horizontal_matched_by_arrow and not vertical_matched_by_arrow:
        # 只有水平方向通过引线找到匹配，竖直方向没有
        if not np.array_equal(bottom_D2, [0, 0, 0]) and np.array_equal(bottom_E2, [0, 0, 0]):
            bottom_E2 = bottom_D2.copy()
            print(f"水平方向通过引线找到匹配，竖直方向没有，设置bottom_E2=bottom_D2: {bottom_E2}")
        elif not np.array_equal(bottom_D2, [0, 0, 0]) and not np.array_equal(bottom_E2, [0, 0, 0]):
            # 如果竖直方向已经有值，但水平方向是通过引线找到的，仍然使用水平方向的值
            print(f"水平方向通过引线找到匹配，竖直方向已有其他值，仍然使用水平方向的值")
            bottom_E2 = bottom_D2.copy()
    elif vertical_matched_by_arrow and not horizontal_matched_by_arrow:
        # 只有竖直方向通过引线找到匹配，水平方向没有
        if not np.array_equal(bottom_E2, [0, 0, 0]) and np.array_equal(bottom_D2, [0, 0, 0]):
            bottom_D2 = bottom_E2.copy()
            print(f"竖直方向通过引线找到匹配，水平方向没有，设置bottom_D2=bottom_E2: {bottom_D2}")
        elif not np.array_equal(bottom_E2, [0, 0, 0]) and not np.array_equal(bottom_D2, [0, 0, 0]):
            # 如果水平方向已经有值，但竖直方向是通过引线找到的，仍然使用竖直方向的值
            print(f"竖直方向通过引线找到匹配，水平方向已有其他值，仍然使用竖直方向的值")
            bottom_D2 = bottom_E2.copy()
    elif not horizontal_matched_by_arrow and not vertical_matched_by_arrow:
        print("水平和竖直方向都没有通过引线找到匹配，保持各自的排序结果")

    print(f"\n最终结果: bottom_D2={bottom_D2}, bottom_E2={bottom_E2}")
    print("=== extract_bottom_dimensions 执行结束 ===\n")

    return bottom_D2, bottom_E2



def extract_pin_dimensions(pin_boxs, bottom_ocr_data_list, triple_factor):
    """
    从bottom视图提取与pin相关的尺寸数据

    参数:
    pin_boxs: pin角坐标，只有一个框[x1, y1, x2, y2]
    bottom_ocr_data_list: OCR检测数据列表，每个元素包含location和max_medium_min
    triple_factor: 嵌套的视图数据

    返回:
    bottom_b: 短边方向尺寸数组 [最大, 标准, 最小]
    bottom_L: 长边方向尺寸数组 [最大, 标准, 最小]
    """

    def extract_bottom_elements(data):
        """递归提取view_name为'bottom'的元素"""
        bottom_elements = []

        if isinstance(data, dict):
            if data.get('view_name') == 'bottom':
                bottom_elements.append(data)
            for value in data.values():
                if isinstance(value, (dict, list)):
                    bottom_elements.extend(extract_bottom_elements(value))
        elif isinstance(data, list):
            for item in data:
                bottom_elements.extend(extract_bottom_elements(item))

        return bottom_elements

    print("=== extract_pin_dimensions 开始执行 ===")

    # 初始化输出值
    bottom_b = [0, 0, 0]
    bottom_L = [0, 0, 0]

    # 检查输入数据
    if not bottom_ocr_data_list or len(bottom_ocr_data_list) == 0:
        print("警告: bottom_ocr_data_list为空，返回默认值")
        return bottom_b, bottom_L

    print(f"收到 {len(bottom_ocr_data_list)} 个bottom OCR数据")

    # 提取triple_factor中的所有bottom元素
    bottom_elements = extract_bottom_elements(triple_factor)

    print(f"找到 {len(bottom_elements)} 个bottom元素")

    if not bottom_elements:
        print("警告: 没有找到bottom元素，返回默认值")
        return bottom_b, bottom_L

    # 将bottom元素分为两类：有arrow_pairs和没有arrow_pairs的
    bottom_with_arrow = []
    bottom_without_arrow = []

    for element in bottom_elements:
        if element.get('arrow_pairs') is not None:
            bottom_with_arrow.append(element)
        else:
            bottom_without_arrow.append(element)

    print(f"有arrow_pairs的bottom元素: {len(bottom_with_arrow)} 个")
    print(f"无arrow_pairs的bottom元素: {len(bottom_without_arrow)} 个")

    # 为每个OCR数据找到匹配的bottom元素，创建融合结构B
    all_b_elements = []

    print(f"开始匹配OCR数据和bottom元素...")
    matched_count = 0

    # 使用更宽松的匹配阈值
    position_tolerance = 5.0  # 位置容差从0.001放宽到2.0

    for ocr_data in bottom_ocr_data_list:
        ocr_location = ocr_data.get('location', None)
        max_medium_min = ocr_data.get('max_medium_min', [])

        if ocr_location is None or len(ocr_location) != 4:
            continue

        # 确保max_medium_min是列表格式
        if isinstance(max_medium_min, np.ndarray):
            max_medium_min = max_medium_min.tolist()

        # 优先匹配有arrow_pairs的元素
        matched = False
        matched_element = None

        # 首先尝试匹配有arrow_pairs的元素
        for bottom_element in bottom_with_arrow:
            element_location = bottom_element.get('location', None)
            if element_location is not None and len(element_location) == 4:
                # 使用放宽的阈值比较location
                if (abs(ocr_location[0] - element_location[0]) < position_tolerance and
                        abs(ocr_location[1] - element_location[1]) < position_tolerance and
                        abs(ocr_location[2] - element_location[2]) < position_tolerance and
                        abs(ocr_location[3] - element_location[3]) < position_tolerance):
                    matched = True
                    matched_element = bottom_element
                    print(f"匹配成功(有箭头): OCR位置{ocr_location} 与 bottom位置{element_location}")
                    break

        # 如果没有匹配到有arrow_pairs的元素，再尝试匹配没有arrow_pairs的元素
        if not matched:
            for bottom_element in bottom_without_arrow:
                element_location = bottom_element.get('location', None)
                if element_location is not None and len(element_location) == 4:
                    # 使用放宽的阈值比较location
                    if (abs(ocr_location[0] - element_location[0]) < position_tolerance and
                            abs(ocr_location[1] - element_location[1]) < position_tolerance and
                            abs(ocr_location[2] - element_location[2]) < position_tolerance and
                            abs(ocr_location[3] - element_location[3]) < position_tolerance):
                        matched = True
                        matched_element = bottom_element
                        print(f"匹配成功(无箭头): OCR位置{ocr_location} 与 bottom位置{element_location}")
                        break

        # 如果匹配成功，创建融合结构B
        if matched and matched_element is not None:
            b_element = {
                'location': matched_element['location'],
                'direction': matched_element.get('direction', ''),
                'arrow_pairs': matched_element.get('arrow_pairs', None),
                'max_medium_min': max_medium_min  # 使用OCR的max_medium_min
            }
            all_b_elements.append(b_element)
            matched_count += 1

            # 从原始列表中移除已匹配的元素，避免重复匹配
            if matched_element in bottom_with_arrow:
                bottom_with_arrow.remove(matched_element)
            elif matched_element in bottom_without_arrow:
                bottom_without_arrow.remove(matched_element)

    print(f"匹配完成，共找到 {matched_count} 个匹配项")

    if not all_b_elements:
        print("警告: 没有找到匹配的B元素，返回默认值")
        return bottom_b, bottom_L

    # 按照标准值(中间值)对all_b_elements排序（升序）
    all_b_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0)
    print(f"按标准值排序后，所有B元素的max_medium_min: {[b['max_medium_min'] for b in all_b_elements]}")

    # 检查pin_boxs是否存在
    if pin_boxs is None or len(pin_boxs) == 0:
        print("警告: pin_boxs为空，使用标准值排序方法")
        # 使用排序后第一个元素的max_medium_min作为bottom_b
        if all_b_elements:
            bottom_b = all_b_elements[0]['max_medium_min'].copy()
            print(f"bottom_b使用第一个元素: max_medium_min={bottom_b}")

        # bottom_L使用最后一个元素的max_medium_min（如果存在且大于bottom_b），否则使用第二个
        if len(all_b_elements) >= 2:
            # 判断最后一个元素的标准值是否大于第一个元素
            last_std = all_b_elements[-1]['max_medium_min'][1] if len(all_b_elements[-1]['max_medium_min']) > 1 else 0
            first_std = all_b_elements[0]['max_medium_min'][1] if len(all_b_elements[0]['max_medium_min']) > 1 else 0

            if last_std > first_std:
                bottom_L = all_b_elements[-1]['max_medium_min'].copy()
                print(f"bottom_L使用最后一个元素: max_medium_min={bottom_L}")
            else:
                bottom_L = all_b_elements[1]['max_medium_min'].copy()
                print(f"bottom_L使用第二个元素: max_medium_min={bottom_L}")
        elif all_b_elements:
            bottom_L = all_b_elements[0]['max_medium_min'].copy()
            print(f"bottom_L只有一个元素可用，使用第一个元素: max_medium_min={bottom_L}")

        print(f"\n最终结果: bottom_b={bottom_b}, bottom_L={bottom_L}")
        print("=== extract_pin_dimensions 执行结束 ===\n")
        return bottom_b, bottom_L

    # 计算pin_boxs的尺寸（只有一个框）
    try:
        pin_box = pin_boxs[0] if isinstance(pin_boxs, list) else pin_boxs
        pin_width = abs(float(pin_box[2]) - float(pin_box[0]))  # x2 - x1
        pin_height = abs(float(pin_box[3]) - float(pin_box[1]))  # y2 - y1

        # 判断短边和长边
        if pin_width <= pin_height:
            # 宽度是短边，高度是长边
            pin_short = pin_width  # 短边
            pin_long = pin_height  # 长边
            print(f"pin_boxs尺寸: 宽度={pin_width:.2f}(短边), 高度={pin_height:.2f}(长边)")
        else:
            # 高度是短边，宽度是长边
            pin_short = pin_height  # 短边
            pin_long = pin_width  # 长边
            print(f"pin_boxs尺寸: 宽度={pin_width:.2f}(长边), 高度={pin_height:.2f}(短边)")

    except Exception as e:
        print(f"错误: 计算pin_boxs尺寸时出错: {e}")
        # 使用标准值排序方法
        if all_b_elements:
            bottom_b = all_b_elements[0]['max_medium_min'].copy()
            if len(all_b_elements) >= 2:
                bottom_L = all_b_elements[-1]['max_medium_min'].copy() if all_b_elements[-1]['max_medium_min'][1] > \
                                                                          all_b_elements[0]['max_medium_min'][1] else \
                all_b_elements[1]['max_medium_min'].copy()
            else:
                bottom_L = all_b_elements[0]['max_medium_min'].copy()

        return bottom_b, bottom_L

    # 开始与pin_boxs尺寸进行比对
    print("开始与pin_boxs尺寸进行比对...")
    best_short_match = None
    best_long_match = None
    min_short_diff = float('inf')
    min_long_diff = float('inf')

    # 优先选择有arrow_pairs的元素
    for idx, element in enumerate(all_b_elements):
        arrow_pairs = element.get('arrow_pairs', None)

        if arrow_pairs is None or len(arrow_pairs) == 0:
            continue  # 跳过没有arrow_pairs的元素

        # 获取最后一位（引线之间距离）
        try:
            arrow_distance = float(arrow_pairs[-1])
        except Exception as e:
            continue

        # 计算与短边和长边的差异
        short_diff = abs(arrow_distance - pin_short)
        long_diff = abs(arrow_distance - pin_long)

        print(f"元素{idx}(有箭头): 箭头距离={arrow_distance:.2f}, "
              f"与短边差异={short_diff:.2f}, 与长边差异={long_diff:.2f}")

        # 寻找与短边最相似的元素
        if short_diff < min_short_diff:
            min_short_diff = short_diff
            best_short_match = element
            print(f"  更新短边最佳匹配: 差异={short_diff:.2f}")

        # 寻找与长边最相似的元素
        if long_diff < min_long_diff:
            min_long_diff = long_diff
            best_long_match = element
            print(f"  更新长边最佳匹配: 差异={long_diff:.2f}")

    # 如果通过有arrow_pairs的元素没有找到匹配，再考虑没有arrow_pairs的元素
    if best_short_match is None or best_long_match is None:
        print("通过有arrow_pairs的元素未找到足够匹配，考虑无arrow_pairs的元素...")
        for idx, element in enumerate(all_b_elements):
            if element.get('arrow_pairs') is not None:
                continue  # 跳过已经有arrow_pairs的元素

            # 对于没有arrow_pairs的元素，使用max_medium_min的标准值进行匹配
            max_medium_min = element.get('max_medium_min', [])
            if len(max_medium_min) < 2:
                continue

            std_value = max_medium_min[1]  # 标准值

            # 计算与短边和长边的差异
            short_diff = abs(std_value - pin_short)
            long_diff = abs(std_value - pin_long)

            print(f"元素{idx}(无箭头): 标准值={std_value:.2f}, "
                  f"与短边差异={short_diff:.2f}, 与长边差异={long_diff:.2f}")

            # 寻找与短边最相似的元素
            if short_diff < min_short_diff:
                min_short_diff = short_diff
                best_short_match = element
                print(f"  更新短边最佳匹配: 差异={short_diff:.2f}")

            # 寻找与长边最相似的元素
            if long_diff < min_long_diff:
                min_long_diff = long_diff
                best_long_match = element
                print(f"  更新长边最佳匹配: 差异={long_diff:.2f}")

    # 使用阈值判断是否"很相似"
    similarity_threshold = 0.2  # 从10%放宽到20%的误差
    pin_short_threshold = pin_short * similarity_threshold
    pin_long_threshold = pin_long * similarity_threshold

    print(f"\n相似性阈值: 短边={pin_short_threshold:.2f}, 长边={pin_long_threshold:.2f}")

    # 记录是否通过引线找到匹配
    short_matched = False
    long_matched = False

    # 判断短边是否有匹配
    if best_short_match is not None and min_short_diff <= pin_short_threshold:
        bottom_b = best_short_match['max_medium_min'].copy()
        short_matched = True
        has_arrow = best_short_match.get('arrow_pairs') is not None
        print(
            f"短边找到{'有箭头' if has_arrow else '无箭头'}匹配: max_medium_min={bottom_b}, 差异={min_short_diff:.2f}")
    else:
        # 没有匹配，使用标准值排序取最小
        print(f'短边无相似匹配, 最小差异={min_short_diff:.2f}, 阈值={pin_short_threshold:.2f}')
        if all_b_elements:
            bottom_b = all_b_elements[0]['max_medium_min'].copy()
            print(f"短边使用标准值排序最小: max_medium_min={bottom_b}")

    # 判断长边是否有匹配
    if best_long_match is not None and min_long_diff <= pin_long_threshold:
        # 如果长边匹配的元素与短边匹配的元素相同，且短边已经匹配，则我们需要找另一个元素
        if best_long_match == best_short_match and short_matched:
            print("长边匹配的元素与短边相同，且短边已匹配，为长边寻找次佳匹配")
            # 在剩余元素中寻找与长边最相似的元素
            second_best_long_match = None
            second_min_long_diff = float('inf')

            for idx, element in enumerate(all_b_elements):
                if element == best_short_match:
                    continue  # 跳过已经被短边使用的元素

                # 根据是否有arrow_pairs选择比较方式
                if element.get('arrow_pairs') is not None:
                    try:
                        arrow_distance = float(element['arrow_pairs'][-1])
                        long_diff = abs(arrow_distance - pin_long)
                    except:
                        continue
                else:
                    max_medium_min = element.get('max_medium_min', [])
                    if len(max_medium_min) < 2:
                        continue
                    long_diff = abs(max_medium_min[1] - pin_long)

                if long_diff < second_min_long_diff:
                    second_min_long_diff = long_diff
                    second_best_long_match = element

            # 检查次佳匹配是否满足阈值
            if second_best_long_match is not None and second_min_long_diff <= pin_long_threshold:
                bottom_L = second_best_long_match['max_medium_min'].copy()
                long_matched = True
                has_arrow = second_best_long_match.get('arrow_pairs') is not None
                print(
                    f"长边找到{'有箭头' if has_arrow else '无箭头'}次佳匹配: max_medium_min={bottom_L}, 差异={second_min_long_diff:.2f}")
            else:
                # 没有次佳匹配，使用标准值排序
                print(f'长边无次佳相似匹配')
                if len(all_b_elements) >= 2:
                    # 使用排序后的最后一个元素（最大值）
                    bottom_L = all_b_elements[-1]['max_medium_min'].copy()
                    long_matched = False
                    print(f"长边使用标准值排序最大: max_medium_min={bottom_L}")
                elif all_b_elements:
                    bottom_L = all_b_elements[0]['max_medium_min'].copy()
                    long_matched = False
                    print(f"长边只有一个元素可用，使用第一个: max_medium_min={bottom_L}")
        else:
            bottom_L = best_long_match['max_medium_min'].copy()
            long_matched = True
            has_arrow = best_long_match.get('arrow_pairs') is not None
            print(
                f"长边找到{'有箭头' if has_arrow else '无箭头'}匹配: max_medium_min={bottom_L}, 差异={min_long_diff:.2f}")
    else:
        # 没有匹配，使用标准值排序
        print(f'长边无相似匹配, 最小差异={min_long_diff:.2f}, 阈值={pin_long_threshold:.2f}')
        if len(all_b_elements) >= 2:
            # 使用排序后的最后一个元素（最大值）
            bottom_L = all_b_elements[-1]['max_medium_min'].copy()
            long_matched = False
            print(f"长边使用标准值排序最大: max_medium_min={bottom_L}")
        elif all_b_elements:
            bottom_L = all_b_elements[0]['max_medium_min'].copy()
            long_matched = False
            print(f"长边只有一个元素可用，使用第一个: max_medium_min={bottom_L}")

    print(f"\n最终结果: bottom_b={bottom_b}, bottom_L={bottom_L}")
    print("=== extract_pin_dimensions 执行结束 ===\n")

    return bottom_b, bottom_L



def extract_pitch_dimension(pin_box, bottom_ocr_data_list, triple_factor):
    """
    提取pitch尺寸数据
    重构逻辑：严格按照 pin_box 确定的方向进行定向筛选和计算
    """

    # --- 内部辅助函数 ---
    def get_sort_value(element):
        """获取用于排序的标准值（中间值）"""
        vals = element.get('max_medium_min', [])
        return vals[1] if len(vals) > 1 else (vals[0] if len(vals) > 0 else 0)

    def extract_bottom_elements(data):
        """递归提取view_name为'bottom'的元素"""
        bottom_elements = []

        if isinstance(data, dict):
            if data.get('view_name') == 'bottom':
                bottom_elements.append(data)
            for value in data.values():
                if isinstance(value, (dict, list)):
                    bottom_elements.extend(extract_bottom_elements(value))
        elif isinstance(data, list):
            for item in data:
                bottom_elements.extend(extract_bottom_elements(item))

        return bottom_elements

    print("=== extract_pitch_dimensions 开始执行 ===")

    # 初始化输出值
    pitch = [0, 0, 0]

    # 检查输入数据
    if not bottom_ocr_data_list or len(bottom_ocr_data_list) == 0:
        print("警告: bottom_ocr_data_list为空，返回默认值")
        return pitch

    print(f"收到 {len(bottom_ocr_data_list)} 个bottom OCR数据")

    # 提取triple_factor中的所有bottom元素
    bottom_elements = extract_bottom_elements(triple_factor)

    print(f"找到 {len(bottom_elements)} 个bottom元素")

    if not bottom_elements:
        print("警告: 没有找到bottom元素，返回默认值")
        return pitch

    # 将bottom元素分为两类：有arrow_pairs和没有arrow_pairs的
    bottom_with_arrow = []
    bottom_without_arrow = []

    for element in bottom_elements:
        if element.get('arrow_pairs') is not None:
            bottom_with_arrow.append(element)
        else:
            bottom_without_arrow.append(element)

    print(f"有arrow_pairs的bottom元素: {len(bottom_with_arrow)} 个")
    print(f"无arrow_pairs的bottom元素: {len(bottom_without_arrow)} 个")

    # 为每个OCR数据找到匹配的bottom元素，创建融合结构B
    all_b_elements = []

    print(f"开始匹配OCR数据和bottom元素...")
    matched_count = 0

    # 使用更宽松的匹配阈值
    position_tolerance = 2.0  # 位置容差从0.001放宽到2.0

    for ocr_data in bottom_ocr_data_list:
        ocr_location = ocr_data.get('location', None)
        max_medium_min = ocr_data.get('max_medium_min', [])

        if ocr_location is None or len(ocr_location) != 4:
            continue

        # 确保max_medium_min是列表格式
        if isinstance(max_medium_min, np.ndarray):
            max_medium_min = max_medium_min.tolist()

        # 优先匹配有arrow_pairs的元素
        matched = False
        matched_element = None

        # 首先尝试匹配有arrow_pairs的元素
        for bottom_element in bottom_with_arrow:
            element_location = bottom_element.get('location', None)
            if element_location is not None and len(element_location) == 4:
                # 使用放宽的阈值比较location
                if (abs(ocr_location[0] - element_location[0]) < position_tolerance and
                        abs(ocr_location[1] - element_location[1]) < position_tolerance and
                        abs(ocr_location[2] - element_location[2]) < position_tolerance and
                        abs(ocr_location[3] - element_location[3]) < position_tolerance):
                    matched = True
                    matched_element = bottom_element
                    print(f"匹配成功(有箭头): OCR位置{ocr_location} 与 bottom位置{element_location}")
                    break

        # 如果没有匹配到有arrow_pairs的元素，再尝试匹配没有arrow_pairs的元素
        if not matched:
            for bottom_element in bottom_without_arrow:
                element_location = bottom_element.get('location', None)
                if element_location is not None and len(element_location) == 4:
                    # 使用放宽的阈值比较location
                    if (abs(ocr_location[0] - element_location[0]) < position_tolerance and
                            abs(ocr_location[1] - element_location[1]) < position_tolerance and
                            abs(ocr_location[2] - element_location[2]) < position_tolerance and
                            abs(ocr_location[3] - element_location[3]) < position_tolerance):
                        matched = True
                        matched_element = bottom_element
                        print(f"匹配成功(无箭头): OCR位置{ocr_location} 与 bottom位置{element_location}")
                        break

        # 如果匹配成功，创建融合结构B
        if matched and matched_element is not None:
            b_element = {
                'location': matched_element['location'],
                'direction': matched_element.get('direction', ''),
                'arrow_pairs': matched_element.get('arrow_pairs', None),
                'max_medium_min': max_medium_min  # 使用OCR的max_medium_min
            }
            all_b_elements.append(b_element)
            matched_count += 1

            # 从原始列表中移除已匹配的元素，避免重复匹配
            if matched_element in bottom_with_arrow:
                bottom_with_arrow.remove(matched_element)
            elif matched_element in bottom_without_arrow:
                bottom_without_arrow.remove(matched_element)

    print(f"匹配完成，共找到 {matched_count} 个匹配项")

    if not all_b_elements:
        print("警告: 未找到匹配的融合元素")
        return pitch

    # 3. 【核心逻辑】判定 Pin Box 的方向和参考距离
    # 默认为 UNKNOWN，如果有 pin_box 则计算
    target_direction = "UNKNOWN"  # 'HORIZONTAL' 或 'VERTICAL'
    target_pin_distance = 0.0

    if pin_box is not None:
        try:
            # 数据清洗，兼容不同格式
            box1, box2 = None, None
            if isinstance(pin_box, list):
                if len(pin_box) == 2 and isinstance(pin_box[0], (list, np.ndarray)):
                    box1, box2 = pin_box[0], pin_box[1]
                elif len(pin_box) >= 8 and all(isinstance(x, (int, float)) for x in pin_box[:8]):
                    box1, box2 = pin_box[:4], pin_box[4:8]

            if box1 is not None and box2 is not None:
                # 计算中心点
                c1_x, c1_y = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
                c2_x, c2_y = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2

                dx = abs(c2_x - c1_x)
                dy = abs(c2_y - c1_y)

                # 判定方向
                if dx > dy:
                    target_direction = "HORIZONTAL"
                    target_pin_distance = dx
                    print(f"Pin方向判定: 水平 (dx={dx:.2f} > dy={dy:.2f}),pitch={target_pin_distance}")
                else:
                    target_direction = "VERTICAL"
                    target_pin_distance = dy
                    print(f"Pin方向判定: 垂直 (dy={dy:.2f} >= dx={dx:.2f}),pitch={target_pin_distance}")
            else:
                print(f"Pin Box 格式无法解析: {pin_box}")
        except Exception as e:
            print(f"Pin Box 计算出错: {e}")

    # 4. 根据判定出的方向，筛选候选元素 (Candidate Selection)
    candidates = []

    # 定义方向关键词
    h_keys = ['horizontal', 'up', 'down']
    v_keys = ['vertical', 'left', 'right']

    if target_direction == "HORIZONTAL":
        # 筛选水平方向的元素（包含未知方向作为备选）
        candidates = [e for e in all_b_elements if e['direction'].lower() in h_keys]
        if not candidates:  # 兜底：如果没有明确水平的，尝试包含方向未知的
            candidates = [e for e in all_b_elements if e['direction'].lower() not in v_keys]
        print(f"执行水平逻辑，筛选出候选元素: {candidates} ")

    elif target_direction == "VERTICAL":
        # 筛选垂直方向的元素
        candidates = [e for e in all_b_elements if e['direction'].lower() in v_keys]
        if not candidates:
            candidates = [e for e in all_b_elements if e['direction'].lower() not in h_keys]
        print(f"执行垂直逻辑，筛选出候选元素: {candidates} ")

    else:
        # 方向未知，使用所有元素
        candidates = all_b_elements
        print(f"方向未知，使用所有 {len(candidates)} 个元素")

    if not candidates:
        print("筛选后无候选元素，返回默认值")
        return pitch

    # 5. 【统一算法】步骤A：尝试通过引线距离匹配 (Arrow Matching)
    found_match = False
    similarity_threshold = 0.15  # 10% 误差

    # 只有当计算出了有效的 pin_distance 时才尝试匹配
    if target_pin_distance > 0:
        best_match = None
        min_diff = float('inf')

        print(f"尝试匹配引线距离 (目标: {target_pin_distance:.2f})...")

        for e in candidates:
            pairs = e.get('arrow_pairs')
            if pairs is None or len(pairs) == 0:
                continue
            try:
                val = float(pairs[-1])  # 取最后一个距离
                diff = abs(val - target_pin_distance)
                if diff < min_diff:
                    min_diff = diff
                    best_match = e
            except:
                continue

        # 判断最佳匹配是否在阈值内
        limit = target_pin_distance * similarity_threshold
        if best_match and min_diff <= limit:
            pitch = best_match['max_medium_min'].copy()
            found_match = True
            print(f"匹配成功! 差异 {min_diff:.2f} <= 阈值 {limit:.2f}. Pitch={pitch}")
        else:
            print(f"匹配失败或无合适引线 (最小差异 {min_diff:.2f}). 转入排序逻辑.")

    # 6. 【统一算法】步骤B：排序兜底
    # 如果步骤A没有找到结果，或者根本没有 pin_box 信息
    if not found_match:
        print("执行排序逻辑: 按标准值排序取次小值...")
        # 按中间值排序
        candidates.sort(key=get_sort_value)

        # 打印调试信息
        debug_vals = [get_sort_value(x) for x in candidates]
        print(f"候选值排序结果: {debug_vals}")

        if len(candidates) >= 2:
            pitch = candidates[1]['max_medium_min'].copy()
            print("取第2小的值")
        else:
            pitch = candidates[0]['max_medium_min'].copy()
            print("元素不足2个，取第1个值")

    print(f"最终结果: pitch={pitch}")
    print("=== extract_pitch_dimensions 执行结束 ===\n")
    return pitch


###############################################################################################################################



def compute_SON_parameters(L3):
    top_serial_numbers_data = find_list(L3, 'top_serial_numbers_data')
    bottom_serial_numbers_data = find_list(L3, 'bottom_serial_numbers_data')
    top_ocr_data = find_list(L3, 'top_ocr_data')
    bottom_ocr_data = find_list(L3, 'bottom_ocr_data')
    side_ocr_data = find_list(L3, 'side_ocr_data')
    detailed_ocr_data = find_list(L3, 'detailed_ocr_data')
    yolox_pairs_top = find_list(L3, 'yolox_pairs_top')
    yolox_pairs_bottom = find_list(L3, 'yolox_pairs_bottom')
    top_yolox_pairs_length = find_list(L3, 'top_yolox_pairs_length')
    bottom_yolox_pairs_length = find_list(L3, 'bottom_yolox_pairs_length')
    top_border = find_list(L3, 'top_border')
    bottom_border = find_list(L3, 'bottom_border')
    bottom_pad = find_list(L3, 'bottom_pad')
    print("bottom_pad:", bottom_pad)
    # 输出序号nx,ny
    nx, ny = get_serial(top_serial_numbers_data, bottom_serial_numbers_data)
    # 输出body_x、body_y
    body_x, body_y = get_body(yolox_pairs_top, top_yolox_pairs_length, yolox_pairs_bottom,
                                  bottom_yolox_pairs_length, top_border, bottom_border, top_ocr_data,
                                  bottom_ocr_data)
    # 初始化参数列表
    SON_parameter_list = get_SON_parameter_list(top_ocr_data, bottom_ocr_data, side_ocr_data, detailed_ocr_data,body_x, body_y)
    # 整理参数列表
    SON_parameter_list = resort_parameter_list_2(SON_parameter_list)


    # 输出高
    if len(SON_parameter_list[4]) > 1:
        high = get_QFP_high(SON_parameter_list[4]['maybe_data'])
        if len(high) > 0:
            SON_parameter_list[4]['maybe_data'] = high
            SON_parameter_list[4]['maybe_data_num'] = len(high)
    # 输出pitch
    if len(SON_parameter_list[5]['maybe_data']) > 1 or len(SON_parameter_list[6]['maybe_data']) > 1:
        pitch_x, pitch_y = get_QFP_pitch(SON_parameter_list[5]['maybe_data'], body_x, body_y, nx, ny)
        if len(pitch_x) > 0:
            SON_parameter_list[5]['maybe_data'] = pitch_x
            SON_parameter_list[5]['maybe_data_num'] = len(pitch_x)
        if len(pitch_y) > 0:
            SON_parameter_list[6]['maybe_data'] = pitch_y
            SON_parameter_list[6]['maybe_data_num'] = len(pitch_y)
    # 整理参数列表
    SON_parameter_list = resort_parameter_list_2(SON_parameter_list)

    return SON_parameter_list, nx, ny

def get_SON_parameter_list(top_ocr_data, bottom_ocr_data, side_ocr_data, detailed_ocr_data, body_x, body_y):
    '''
    D/E 10~35
    D1/E1
    D2/E2
    A
    A1
    e
    b
    θ
    L
    c:引脚厚度
    '''
    SON_parameter_list = []
    dic_D = {'parameter_name': 'D', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    dic_E = {'parameter_name': 'E', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    # D_max = 10
    # D_min = 1.0
    # E_max = D_max
    # E_min = D_min

    dic_D1 = {'parameter_name': 'D1', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    dic_E1 = {'parameter_name': 'E1', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    D1_max = 10
    D1_min = 1.0
    E1_max = D1_max
    E1_min = D1_min

    dic_L = {'parameter_name': 'L', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    L_max = 1.5
    L_min = 0.2


    dic_GAGE_PLANE = {'parameter_name': 'GAGE_PLANE', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    GAGE_PLANE_max = 0.25
    GAGE_PLANE_min = 0.25

    dic_c = {'parameter_name': 'c', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    c_max = 0.3
    c_min = 0.1

    dic_θ = {'parameter_name': 'θ', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    θ_max = 10
    θ_min = 0

    dic_θ1 = {'parameter_name': 'θ1', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    θ1_max = 14
    θ1_min = 0

    dic_θ2 = {'parameter_name': 'θ2', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    θ2_max = 16
    θ2_min = 11

    dic_θ3 = {'parameter_name': 'θ3', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    θ3_max = 16
    θ3_min = 11

    dic_A = {'parameter_name': 'A', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    A_max = 1.2
    A_min = 0.5
    dic_A1 = {'parameter_name': 'A1', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    A1_max = 0.10
    A1_min = 0
    dic_e = {'parameter_name': 'e', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    e_max = 0.65
    e_min = 0.35
    dic_b = {'parameter_name': 'b', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    b_max = 0.6
    b_min = 0.15
    dic_D2 = {'parameter_name': 'D2', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    dic_E2 = {'parameter_name': 'E2', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    D2_max = 6.0
    D2_min = 0.8
    E2_max = 6.0
    E2_min = 0.15
    SON_parameter_list.append(dic_D)
    SON_parameter_list.append(dic_E)
    SON_parameter_list.append(dic_D1)
    SON_parameter_list.append(dic_E1)
    SON_parameter_list.append(dic_A)
    SON_parameter_list.append(dic_A1)
    SON_parameter_list.append(dic_e)
    SON_parameter_list.append(dic_b)
    SON_parameter_list.append(dic_D2)
    SON_parameter_list.append(dic_E2)
    SON_parameter_list.append(dic_L)
    SON_parameter_list.append(dic_GAGE_PLANE)
    SON_parameter_list.append(dic_c)
    SON_parameter_list.append(dic_θ)
    SON_parameter_list.append(dic_θ1)
    SON_parameter_list.append(dic_θ2)
    SON_parameter_list.append(dic_θ3)

    for i in range(len(top_ocr_data)):

        if D1_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= D1_max:
            if len(body_x) > 0:
                SON_parameter_list[2]['maybe_data'] = body_x
            else:
                SON_parameter_list[2]['maybe_data'].append(top_ocr_data[i])
                SON_parameter_list[2]['maybe_data_num'] += 1
        if E1_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= E1_max:
            if len(body_y) > 0:
                SON_parameter_list[3]['maybe_data'] = body_y
            else:
                SON_parameter_list[3]['maybe_data'].append(top_ocr_data[i])
                SON_parameter_list[3]['maybe_data_num'] += 1

    for i in range(len(bottom_ocr_data)):
        if e_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= e_max:
            SON_parameter_list[6]['maybe_data'].append(bottom_ocr_data[i])
            SON_parameter_list[6]['maybe_data_num'] += 1
        if b_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= b_max:
            SON_parameter_list[7]['maybe_data'].append(bottom_ocr_data[i])
            SON_parameter_list[7]['maybe_data_num'] += 1
        if L_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= L_max:
            SON_parameter_list[10]['maybe_data'].append(bottom_ocr_data[i])
            SON_parameter_list[10]['maybe_data_num'] += 1


        if D2_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= D2_max:
            # if len(pad_x) > 0:
            #     SON_parameter_list[8]['maybe_data'] = pad_x
            # else:
            SON_parameter_list[8]['maybe_data'].append(bottom_ocr_data[i])
            SON_parameter_list[8]['maybe_data_num'] += 1
        if E2_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= E2_max:
            # if len(pad_y) > 0:
            #     SON_parameter_list[9]['maybe_data'] = pad_y
            # else:
            SON_parameter_list[9]['maybe_data'].append(bottom_ocr_data[i])
            SON_parameter_list[9]['maybe_data_num'] += 1

    # 检查是否满足特殊逻辑：只有3条数据
    if len(side_ocr_data) == 3:
        # 按照 max_medium_min[0] (即 max 值) 进行降序排序 (从大到小)
        sorted_side_data = sorted(side_ocr_data, key=lambda x: x['max_medium_min'][0], reverse=True)
        # 第1大 -> Index 4 (通常是 A 总高)
        SON_parameter_list[4]['maybe_data'].append(sorted_side_data[0])
        SON_parameter_list[4]['maybe_data_num'] += 1
        # 第2大 -> Index 12 (对应 c 参数)
        SON_parameter_list[12]['maybe_data'].append(sorted_side_data[1])
        SON_parameter_list[12]['maybe_data_num'] += 1
        # 第3大 -> Index 5 (对应 A1 Standoff)
        SON_parameter_list[5]['maybe_data'].append(sorted_side_data[2])
        SON_parameter_list[5]['maybe_data_num'] += 1

    else:
        # --- 原有逻辑保持不变 ---
        for i in range(len(side_ocr_data)):
            if A_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= A_max:
                SON_parameter_list[4]['maybe_data'].append(side_ocr_data[i])
                SON_parameter_list[4]['maybe_data_num'] += 1
            if side_ocr_data[i]['max_medium_min'][0] <= A1_max:
                SON_parameter_list[5]['maybe_data'].append(side_ocr_data[i])
                SON_parameter_list[5]['maybe_data_num'] += 1
            if c_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= c_max:
                SON_parameter_list[12]['maybe_data'].append(side_ocr_data[i])
                SON_parameter_list[12]['maybe_data_num'] += 1
            if side_ocr_data[i].get('Absolutely') == 'angle':
                if θ2_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= θ2_max:
                    SON_parameter_list[15]['maybe_data'].append(side_ocr_data[i])
                    SON_parameter_list[15]['maybe_data_num'] += 1

    for i in range(len(detailed_ocr_data)):
        if A_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= A_max:
            SON_parameter_list[4]['maybe_data'].append(detailed_ocr_data[i])
            SON_parameter_list[4]['maybe_data_num'] += 1
        if L_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= L_max:
            SON_parameter_list[10]['maybe_data'].append(detailed_ocr_data[i])
            SON_parameter_list[10]['maybe_data_num'] += 1
        if GAGE_PLANE_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= GAGE_PLANE_max:
            SON_parameter_list[11]['maybe_data'].append(detailed_ocr_data[i])
            SON_parameter_list[11]['maybe_data_num'] += 1
        if c_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= c_max:
            SON_parameter_list[12]['maybe_data'].append(detailed_ocr_data[i])
            SON_parameter_list[12]['maybe_data_num'] += 1
        if A1_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= A1_max:
            SON_parameter_list[5]['maybe_data'].append(detailed_ocr_data[i])
            SON_parameter_list[5]['maybe_data_num'] += 1

        if detailed_ocr_data[i]['Absolutely'] == 'angle':
            if θ_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= θ_max:
                SON_parameter_list[13]['maybe_data'].append(detailed_ocr_data[i])
                SON_parameter_list[13]['maybe_data_num'] += 1
            if θ1_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= θ1_max:
                SON_parameter_list[14]['maybe_data'].append(detailed_ocr_data[i])
                SON_parameter_list[14]['maybe_data_num'] += 1

            if θ2_min < detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= θ2_max:
                SON_parameter_list[15]['maybe_data'].append(detailed_ocr_data[i])
                SON_parameter_list[15]['maybe_data_num'] += 1
            if θ3_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= θ3_max:
                SON_parameter_list[16]['maybe_data'].append(detailed_ocr_data[i])
                SON_parameter_list[16]['maybe_data_num'] += 1

    for i in range(len(SON_parameter_list)):
        print("***/", SON_parameter_list[i]['parameter_name'],"/***")

        for j in range(len(SON_parameter_list[i]['maybe_data'])):
            print(SON_parameter_list[i]['maybe_data'][j]['max_medium_min'])
    #
    # for i in range(len(SON_parameter_list)):
    #     print(SON_parameter_list[i]['maybe_data_num'])

    return SON_parameter_list



