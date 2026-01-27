# 外部文件：
from ultralytics import YOLO
import os
import cv2
import json
import numpy as np
from typing import Iterable, List, Dict, Any

from package_core.PackageExtract.function_tool import *
from package_core.PackageExtract.get_pairs_data_present5_test import *
from package_core.PackageExtract import common_pipeline
import os

from package_core.PackageExtract.BGA_Function.pre_extract import (
    other_match_dbnet,
    pin_match_dbnet,
    angle_match_dbnet,
    num_match_dbnet,
    num_direction,
    match_triple_factor,
    targeted_ocr
)


# SOP数字提取
##############################################################################
def extract_pin_boxes_from_txt(file_path):
    """
    从txt文件中提取引脚框数据
    适配格式: [x1, y1, x2, y2],[x3, y3, x4, y4]...
    """
    pin_boxes = []
    pin_box = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # 读取全部内容并去除首尾空白
            content = file.read().strip()

            # 判空处理
            if not content:
                return [], []

            # --- 核心修改逻辑 ---
            # 使用正则表达式查找所有被 [] 包裹的内容
            # r'\[([^\]]+)\]' 的含义是：匹配 '[' 开始，中间是非 ']' 的任意字符，以 ']' 结束
            matches = re.findall(r'\[([^\]]+)\]', content)

            for match in matches:
                # match 的内容类似于 "319.00, 105.00, 346.00, 142.00"
                try:
                    # 1. 按逗号分割
                    # 2. 去除每个数字字符串的空格
                    # 3. 转换为浮点数
                    # 4. 过滤空字符
                    coords = [float(x.strip()) for x in match.split(',') if x.strip()]

                    # 确保解析出来的数据不为空
                    if coords:
                        pin_boxes.append(coords)
                except ValueError:
                    print(f"警告：无法解析的数据片段 -> {match}")
                    continue

        # --- 结果分配逻辑 ---
        # pin_boxes: 包含文件里所有的框 [[319...], [359...]]
        # pin_box: 按照您之前的逻辑，通常取第一个框作为主要对象，且包一层列表

        if pin_boxes:
            pin_box = [pin_boxes[0]]  # 取第一个框，结果如 [[319.00, ...]]
        else:
            pin_box = []

        return pin_box, pin_boxes

    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return [], []
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return [], []


def extract_SOP_D_E(L3, triple_factor):
    top_ocr_data = find_list(L3, "top_ocr_data")
    bottom_ocr_data = find_list(L3, "bottom_ocr_data")
    top_border = find_list(L3, "top_border")
    bottom_border = find_list(L3, "bottom_border")
    side_ocr_data = find_list(L3, "side_ocr_data")
    print(f'bottom_ocr_data:{bottom_ocr_data}')
    print(f'top_ocr_data:{top_ocr_data}')
    print(f'top_border:{top_border}')
    print(f'bottom_border:{bottom_border}')
    print(f'side_ocr_data:{side_ocr_data}')

    if len(bottom_border) != 0:
        top_D, top_E = extract_top_dimensions(bottom_border, bottom_ocr_data,side_ocr_data, triple_factor, 1)
    else:
        top_D, top_E = extract_top_dimensions(top_border, top_ocr_data, side_ocr_data,triple_factor, 0)

    return top_D, top_E


def extract_span(L3, triple_factor, bottom_D, bottom_E, direction):
    top_ocr_data = find_list(L3, "top_ocr_data")
    bottom_ocr_data = find_list(L3, "bottom_ocr_data")
    span = extract_span_dimensions(bottom_ocr_data, triple_factor, bottom_D, bottom_E, direction,1)
    if np.all(np.array(span) == 0):
        span = extract_span_dimensions(top_ocr_data, triple_factor, bottom_D, bottom_E, direction,0)
    return span


def extract_bottom_D2_E2(L3, triple_factor, bottom_D, bottom_E):
    top_ocr_data = find_list(L3, "top_ocr_data")
    bottom_ocr_data = find_list(L3, "bottom_ocr_data")
    top_pad = find_list(L3, "top_pad")
    bottom_pad = find_list(L3, "bottom_pad")
    print(f'bottom_ocr_data:{bottom_ocr_data}')
    print(f'bottom_pad_data:{bottom_pad}')
    print(f'top_ocr_data:{top_ocr_data}')
    print(f'top_pad_data:{top_pad}')
    if len(bottom_pad) != 0:
        bottom_D2, bottom_E2 = extract_bottom_dimensions(bottom_D, bottom_E, bottom_pad, bottom_ocr_data, triple_factor,1)
    else:
        bottom_D2, bottom_E2 = extract_bottom_dimensions(bottom_D, bottom_E, top_pad, top_ocr_data, triple_factor,0)
    return bottom_D2, bottom_E2


def extract_SOP_A_A1(L3, triple_factor, direction):
    side_ocr_data = find_list(L3, "side_ocr_data")
    detail_ocr_data = find_list(L3, "detailed_ocr_data")
    A, A1 = extract_sop_side_dimensions(side_ocr_data, triple_factor, direction, view_name="side")

    is_A_empty = np.all(np.array(A) == 0)
    is_A1_empty = np.all(np.array(A1) == 0)
    if is_A_empty and is_A1_empty:
        print("Side视图未找到有效 A, A1, c 数据，正在尝试从 Detailed 视图提取...")
        A, A1 = extract_sop_side_dimensions(detail_ocr_data, triple_factor, "垂直方向", view_name="detail")
    return A, A1


def extract_SOP_L(L3, triple_factor, bottom_direction, top_D, top_E, span):
    detail_ocr_data = find_list(L3, "detailed_ocr_data")
    L = extract_pin_L(detail_ocr_data, triple_factor, bottom_direction, top_D, top_E, span)
    return L

def extract_bottom_b_L(L3, triple_factor, pin_boxs,bottom_direction,side_direction):
    bottom_ocr_data = find_list(L3, "bottom_ocr_data")
    side_ocr_data = find_list(L3, "side_ocr_data")
    print("调试信息如下")
    print(f'bottom_ocr_data:{bottom_ocr_data}')
    print(f'side_ocr_data:{side_ocr_data}')
    bottom_b, bottom_L = extract_pin_dimensions(pin_boxs, bottom_ocr_data,triple_factor,bottom_direction,0)
    if np.all(np.array(bottom_b) == 0):
        bottom_b, bottom_L = extract_pin_dimensions(pin_boxs, side_ocr_data, triple_factor,side_direction,1)
    # if(bottom_D2[1] > bottom_E2[1]):
    #     bottom_D2, bottom_E2 = bottom_E2, bottom_D2

    return bottom_b, bottom_L

def extract_bottom_pitch(L3, triple_factor, pin_box,bottom_direction,side_direction):
    bottom_ocr_data = find_list(L3, "bottom_ocr_data")
    side_ocr_data = find_list(L3, "side_ocr_data")
    print(f'bottom_ocr_data:{bottom_ocr_data}')
    print(f'side_ocr_data:{side_ocr_data}')
    bottom_pitch = extract_pitch_dimension(pin_box, bottom_ocr_data, triple_factor,bottom_direction,0)
    if np.all(np.array(bottom_pitch) == 0):
        bottom_pitch = extract_pitch_dimension(pin_box, side_ocr_data, triple_factor,side_direction,1)
    return bottom_pitch


# --- 内部函数: 递归提取视图元素 ---
def extract_view_elements(data, key):
        """
        根据 key 递归提取目标视图的元素
        key == 0: target 'top' and 'side'
        key == 1: target 'bottom' and 'side'
        """
        if key == 0:
            target_views = {'top', 'side'}
        elif key == 1:
            target_views = {'bottom', 'side'}
        else:
            print(f"Warning: Unknown key {key}, returning empty list")
            return []

        collected_elements = []

        def _traverse(obj):
            if isinstance(obj, dict):
                # 检查当前字典是否匹配目标视图
                # 注意：有些数据结构可能在顶层就有 view_name，有些可能在 children 里
                if obj.get('view_name') in target_views:
                    collected_elements.append(obj)

                # 继续递归遍历值
                for value in obj.values():
                    if isinstance(value, (dict, list)):
                        _traverse(value)

            elif isinstance(obj, list):
                for item in obj:
                    _traverse(item)

        _traverse(data)
        return collected_elements

def extract_top_dimensions(border, top_ocr_data_list, side_ocr_list, triple_factor, key):
    """
    从top/bottom视图及side视图提取尺寸数据，处理多个OCR数据元素并进行匹配。

    参数:
    border: 边界框，格式为[[x1, y1, x2, y2]] (本代码中主要用于上下文，暂未直接用于裁剪)
    top_ocr_data_list: Top/Bottom视图的OCR检测数据列表
    side_ocr_list: Side视图的OCR检测数据列表 (新增参数)
    triple_factor: 嵌套的视图数据结构 (包含 view_name, direction, arrow_pairs 等信息)
    key: 控制提取模式 (0: top+side, 1: bottom+side)

    返回:
    result_D: 水平方向尺寸数组 [最大, 标准, 最小]
    result_E: 竖直方向尺寸数组 [最大, 标准, 最小]
    """

    # --- 0. 数据预处理 ---
    # 合并 OCR 数据，处理 None 的情况
    top_ocr = top_ocr_data_list if top_ocr_data_list else []
    side_ocr = side_ocr_list if side_ocr_list else []
    ocr_data_list = top_ocr + side_ocr

    print("=== extract_top_dimensions 开始执行 ===")
    print(f"Key: {key} (0=Top+Side, 1=Bottom+Side)")
    print(f"Total OCR items: {len(ocr_data_list)} (Top/Bot: {len(top_ocr)}, Side: {len(side_ocr)})")

    # 初始化返回值
    result_D = [0, 0, 0]
    result_E = [0, 0, 0]

    # 检查输入数据
    if not ocr_data_list:
        print("Warning: ocr_data_list is empty, returning default values")
        return result_D, result_E

    # --- Step 1: 提取相关视图元素 ---
    view_elements = extract_view_elements(triple_factor, key)
    print(f"Found {len(view_elements)} elements matching target views inside triple_factor")

    # --- Step 2: 处理未找到元素的情况 (Fallback) ---
    if len(view_elements)==0:
        print("Warning: No matching view elements found. Sorting purely by OCR standard values.")
        all_max_medium_min = []
        for ocr_data in ocr_data_list:
            max_medium_min = ocr_data.get('max_medium_min', [])
            if len(max_medium_min) == 3:
                all_max_medium_min.append(max_medium_min)

        if all_max_medium_min:
            # 按标准值(索引1)降序排序，直接取最大值作为 D 和 E 的兜底
            all_max_medium_min.sort(key=lambda x: x[1], reverse=True)
            result_D = all_max_medium_min[0].copy()
            result_E = all_max_medium_min[0].copy()
            print(f"Fallback Result: D={result_D}, E={result_E}")
        else:
            print("No valid max_medium_min data found in OCR list.")

        return result_D, result_E

    # --- Step 3: 元素分类 (有箭头 vs 无箭头) ---
    top_with_arrow = []
    top_without_arrow = []


    for element in view_elements:
        if element.get('arrow_pairs') is not None:
            top_with_arrow.append(element)
        else:
            top_without_arrow.append(element)

    print(f"Elements with arrow_pairs: {len(top_with_arrow)}")
    print(f"Elements without arrow_pairs: {len(top_without_arrow)}")

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


def extract_bottom_dimensions(bottom_D, bottom_E, pad, bottom_ocr_data_list, triple_factor,key):
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
    bottom_elements = extract_top_elements(triple_factor)

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



def extract_pin_L(detail_ocr_data, triple_factor, bottom_direction, top_D, top_E, span):
    """
    提取 SOP/QFP 等封装的 Pin 长度 L。
    基于 Detailed 视图，利用 Span 和 Body 尺寸计算限制条件 L1，筛选出符合条件的尺寸。
    """

    # --- 内部函数：递归提取 view_name='detailed' 的元素 ---
    def extract_detailed_elements_recursive(data):
        elements = []
        if isinstance(data, dict):
            # 兼容 detailed 或 detail 命名
            if data.get('view_name') in ['detailed', 'detail']:
                elements.append(data)
            for value in data.values():
                if isinstance(value, (dict, list)):
                    elements.extend(extract_detailed_elements_recursive(value))
        elif isinstance(data, list):
            for item in data:
                elements.extend(extract_detailed_elements_recursive(item))
        return elements

    print(f"=== extract_pin_L 开始执行 (Bottom方向: {bottom_direction}) ===")

    # 初始化默认输出值
    L = [0, 0, 0]

    # --- 1. 基础数据准备 ---
    if not detail_ocr_data:
        print("警告: detail_ocr_data 为空，返回默认值")
        return L

    # 提取结构元素
    detail_elements = extract_detailed_elements_recursive(triple_factor)
    print(f"找到 {len(detail_elements)} 个 detailed 结构元素")

    if not detail_elements:
        return L

    # 将 structural 元素分为有/无 arrow_pairs，优先匹配有箭头的

    det_with_arrow = [e for e in detail_elements
        if e.get('arrow_pairs') is not None and len(e.get('arrow_pairs')) > 0
    ]
    det_without_arrow = [
        e for e in detail_elements
        if e.get('arrow_pairs') is None or len(e.get('arrow_pairs')) == 0
    ]

    all_b_elements = []
    position_tolerance = 5.0  # 匹配容差

    print("开始匹配 OCR 数据...")

    # --- 2. 融合 OCR 和 Structural 数据 (含方向修正) ---
    matched_count = 0
    for ocr_data in detail_ocr_data:
        ocr_location = ocr_data.get('location')
        max_medium_min = ocr_data.get('max_medium_min', [])

        if not ocr_location or len(ocr_location) != 4:
            continue

        # 确保 max_medium_min 是列表
        if isinstance(max_medium_min, np.ndarray):
            max_medium_min = max_medium_min.tolist()

        # 过滤无效数据
        if len(max_medium_min) < 2:
            continue

        matched = False
        matched_element = None

        # 2.1 优先匹配有箭头的元素
        for el in det_with_arrow:
            el_loc = el.get('location', [])
            if len(el_loc) == 4:
                if all(abs(ocr_location[i] - el_loc[i]) < position_tolerance for i in range(4)):
                    matched = True
                    matched_element = el
                    break

        # 2.2 其次匹配无箭头的元素
        if not matched:
            for el in det_without_arrow:
                el_loc = el.get('location', [])
                if len(el_loc) == 4:
                    if all(abs(ocr_location[i] - el_loc[i]) < position_tolerance for i in range(4)):
                        matched = True
                        matched_element = el
                        break

        # 2.3 构建融合数据并进行【方向修正】
        if matched and matched_element:
            # 获取原始方向
            raw_direction = matched_element.get('direction', '')
            calculated_direction = raw_direction  # 默认为原始值

            arrow_pairs_data = matched_element.get('arrow_pairs', None)

            # --- 关键修改：通过几何形状强制修正方向 ---
            # 如果 arrow_pairs 存在，计算 Δx 和 Δy 来判断水平/垂直
            if arrow_pairs_data is not None and len(arrow_pairs_data) >= 4:
                try:
                    # 假设格式为 [x1, y1, x2, y2]
                    ax1, ay1, ax2, ay2 = map(float, arrow_pairs_data[:4])
                    adx = abs(ax2 - ax1)
                    ady = abs(ay2 - ay1)

                    if adx > ady:
                        calculated_direction = 'horizontal'
                    elif ady > adx:
                        calculated_direction = 'vertical'
                    # 如果 adx == ady，保持原 direction 或视为无效
                except (ValueError, IndexError):
                    pass  # 数据异常则不修改

            b_element = {
                'location': matched_element['location'],
                'direction': str(calculated_direction).lower(),  # 统一转小写
                'arrow_pairs': arrow_pairs_data,
                'max_medium_min': max_medium_min
            }
            all_b_elements.append(b_element)
            matched_count += 1

            # 防止重复匹配（可选，视具体业务需求而定，建议加上）
            if matched_element in det_with_arrow:
                det_with_arrow.remove(matched_element)
            elif matched_element in det_without_arrow:
                det_without_arrow.remove(matched_element)

    print(f"匹配完成，生成融合数据: {matched_count} 个")

    if not all_b_elements:
        return L

    # --- 3. 计算 L1 (理论 Pin 长上限) ---
    # 公式: L1 < (Span - Body) / 2
    # 注意：这里的 Span 和 Body 需要取“标准值”(索引1)
    span_val = span[1] if (span and len(span) > 1) else 0
    body_val = 0

    # 根据 Bottom 视图的主方向来决定 Body 对应的尺寸 (D 或 E)
    if bottom_direction == '垂直方向' or bottom_direction == 'vertical':
        # 垂直分布意味着 Span 是高度方向，Body 取 E (高度)
        body_val = top_E[1] if (top_E and len(top_E) > 1) else 0
        print(f"L1 计算基准 (垂直): Span={span_val}, Body(E)={body_val}")
    else:
        # 默认为水平方向，Span 是宽度方向，Body 取 D (宽度)
        body_val = top_D[1] if (top_D and len(top_D) > 1) else 0
        print(f"L1 计算基准 (水平): Span={span_val}, Body(D)={body_val}")

    if span_val <= 0:
        print("警告: Span 值为 0，无法计算有效 L1，设置 L1 为无限大")
        L1 = float('inf')
    else:
        # 如果 body_val 为 0，可能只有单侧引脚或数据缺失，仍计算一半
        L1 = (span_val - body_val) / 2
        print(f"计算出的 Pin 长上限 L1 = ({span_val} - {body_val}) / 2 = {L1:.4f}")

    if L1 <= 0:
        print(f"警告: 计算出的 L1 ({L1}) <= 0，逻辑异常")
        return L

    # --- 4. 筛选与排序 (只保留 horizontal 且 < L1 的数据) ---
    candidates = []

    # 定义合法的水平方向标识
    valid_horizontal_keys = ['horizontal']

    for element in all_b_elements:
        direction = element.get('direction', '')
        # 安全获取标准值
        val = element['max_medium_min'][1] if len(element['max_medium_min']) > 1 else 0

        # 条件 1: 必须是水平方向 (Horizontal)
        is_horizontal = any(k in direction for k in valid_horizontal_keys)

        if is_horizontal:
            # 条件 2: 数值必须合理 (0 < val < L1)
            # 放宽一点点 L1 上限 (L1 * 1.05) 以防浮点误差，或者严格使用 L1
            if 0 < val < L1:
                candidates.append(element)
            else:
                pass
                # print(f"排除: 值 {val} 超出范围 (0, {L1})")

    print(f"筛选后符合条件的候选数量: {len(candidates)}")

    if not candidates:
        print("没有找到符合条件的 L 候选值")
        return L

    # --- 5. 最终排序取值 ---
    # 按照标准值 (max_medium_min[1]) 降序排列
    # reverse=True 表示 [最大值, ..., 最小值]
    candidates.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0, reverse=True)

    # 打印调试信息
    # print(f"Top 3 candidates: {[c['max_medium_min'][1] for c in candidates[:3]]}")

    # 取最大的符合条件的值 (index 0)
    # 通常 Pin 长取最大可行值是比较安全的（避免取到引脚上的局部小特征）
    best_candidate = candidates[-1]
    L = best_candidate['max_medium_min']

    print(f"最终选定的 Pin L: {L}")
    print("=== extract_pin_L 执行结束 ===\n")

    return L


def extract_sop_side_dimensions(side_ocr_data, triple_factor, direction, view_name):
    """
    提取 Side 视图的尺寸，根据方向筛选并返回前三个主要尺寸 (A, A1)
    注意：如果这不是类的方法，请删除第一个参数 self。
    """

    # --- 内部函数：递归提取 view_name='side' 的元素 ---
    def extract_side_elements_recursive(data, view_name):
        elements = []
        if isinstance(data, dict):
            # 这里检查 'side' 视图
            if data.get('view_name') == view_name:
                elements.append(data)
            for value in data.values():
                if isinstance(value, (dict, list)):
                    # 【修复】这里必须传递 view_name，否则会报错
                    elements.extend(extract_side_elements_recursive(value, view_name))
        elif isinstance(data, list):
            for item in data:
                # 【修复】这里必须传递 view_name，否则会报错
                elements.extend(extract_side_elements_recursive(item, view_name))

        return elements

    # 初始化输出值
    A = [0, 0, 0]
    A1 = [0, 0, 0]

    # 1. 基础数据检查
    if not side_ocr_data or len(side_ocr_data) == 0:
        print("警告: side_ocr_data 为空，返回默认值")
        return A, A1

    print(f"收到 {len(side_ocr_data)} 个 side OCR数据")

    # 2. 从 triple_factor 提取所有 side 元素
    side_elements = extract_side_elements_recursive(triple_factor, view_name)
    print(f"找到 {len(side_elements)} 个 side 结构元素")

    if not side_elements:
        print("警告: triple_factor 中没有找到 side 元素，返回默认值")
        return A, A1

    # 3. 将 side 元素分为有/无 arrow_pairs
    side_with_arrow = []
    side_without_arrow = []
    for element in side_elements:
        if element.get('arrow_pairs') is not None:
            side_with_arrow.append(element)
        else:
            side_without_arrow.append(element)

    # 4. 匹配 OCR 数据与结构化元素 (Merge)
    all_b_elements = []
    matched_count = 0
    position_tolerance = 5.0  # 宽松匹配阈值

    print(f"开始匹配 OCR 数据...")

    for ocr_data in side_ocr_data:
        ocr_location = ocr_data.get('location', None)
        max_medium_min = ocr_data.get('max_medium_min', [])

        if ocr_location is None or len(ocr_location) != 4:
            continue

        if isinstance(max_medium_min, np.ndarray):
            max_medium_min = max_medium_min.tolist()

        # 跳过无效的尺寸数据
        if not max_medium_min or len(max_medium_min) < 2:
            continue

        matched = False
        matched_element = None

        # 4.1 优先匹配有箭头的数据
        for side_element in side_with_arrow:
            el_loc = side_element.get('location', None)
            if el_loc is not None and len(el_loc) == 4:
                if (abs(ocr_location[0] - el_loc[0]) < position_tolerance and
                        abs(ocr_location[1] - el_loc[1]) < position_tolerance and
                        abs(ocr_location[2] - el_loc[2]) < position_tolerance and
                        abs(ocr_location[3] - el_loc[3]) < position_tolerance):
                    matched = True
                    matched_element = side_element
                    break

        # 4.2 其次匹配无箭头的数据
        if not matched:
            for side_element in side_without_arrow:
                el_loc = side_element.get('location', None)
                if el_loc is not None and len(el_loc) == 4:
                    if (abs(ocr_location[0] - el_loc[0]) < position_tolerance and
                            abs(ocr_location[1] - el_loc[1]) < position_tolerance and
                            abs(ocr_location[2] - el_loc[2]) < position_tolerance and
                            abs(ocr_location[3] - el_loc[3]) < position_tolerance):
                        matched = True
                        matched_element = side_element
                        break

        # 4.3 存储匹配结果
        if matched and matched_element is not None:
            b_element = {
                'location': matched_element['location'],
                'direction': matched_element.get('direction', ''),
                'arrow_pairs': matched_element.get('arrow_pairs', None),
                'max_medium_min': max_medium_min
            }
            all_b_elements.append(b_element)
            matched_count += 1

            # 避免重复匹配同一个结构元素
            if matched_element in side_with_arrow:
                side_with_arrow.remove(matched_element)
            elif matched_element in side_without_arrow:
                side_without_arrow.remove(matched_element)

    print(f"匹配完成，共找到 {matched_count} 个匹配项")

    if not all_b_elements:
        return A, A1

    # 5. 根据 direction 进行方向筛选
    horizontal_keys = ['horizontal', 'up', 'down']
    vertical_keys = ['vertical', 'left', 'right']

    target_candidates = []

    if direction == "垂直方向":
        # 筛选水平方向的元素
        target_candidates = [e for e in all_b_elements if e['direction'].lower() in horizontal_keys]
        print(f"方向筛选(水平): 剩余 {len(target_candidates)} 个候选")
    elif direction == "水平方向":
        # 筛选垂直方向的元素
        target_candidates = [e for e in all_b_elements if e['direction'].lower() in vertical_keys]
        print(f"方向筛选(垂直): 剩余 {len(target_candidates)} 个候选")
    else:
        print(f"错误: 未知方向 '{direction}'，不进行筛选")
        return A, A1

    if not target_candidates:
        print(f"该方向下无候选元素，返回默认值")
        return A, A1

    # 6. 排序逻辑
    # 按标准值(max_medium_min[1]) 降序排列 (reverse=True)
    target_candidates.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0, reverse=True)

    # 提取 max_medium_min 数组
    sorted_dims = [e['max_medium_min'] for e in target_candidates]
    print(f"排序后的候选尺寸: {sorted_dims}")

    # 如果不足3个，补充[0,0,0]
    while len(sorted_dims) < 3:
        sorted_dims.append([0, 0, 0])

    # 8. 分配变量 (A, A1)
    A = sorted_dims[0]  # 最大值
    A1 = sorted_dims[-1]  # 如果只有两个值，这取的是最小值；如果有三个值，这取的是第三大的值

    print(f"最终提取结果: A={A}, A1={A1}")
    print("=== extract_side_dimensions 执行结束 ===\n")

    return A, A1


def extract_span_dimensions(bottom_ocr_data_list, triple_factor,bottom_D, bottom_E, direction,key):
    """
    从bottom视图提取Pitch相关尺寸数据，优先匹配带引线的元素
    """

    # --- 内部辅助函数 ---
    def get_arrow_length(element):
        """安全获取引线物理长度 (arrow_pairs的最后一个值)"""
        pairs = element.get('arrow_pairs')
        try:
            if pairs is not None and len(pairs) > 0:
                if isinstance(pairs, np.ndarray):
                    return float(pairs[-1])
                elif isinstance(pairs, list):
                    return float(pairs[-1])
        except:
            pass
        return 0.0

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

    print("=== extract_span_dimensions 开始执行 ===")

    # 初始化输出
    span = [0, 0, 0]  # 默认返回值

    # 1. 基础数据检查
    if not bottom_ocr_data_list:
        print("警告: bottom_ocr_data_list为空")
        return span  # 或者返回空列表 []，视你的后续逻辑而定

    print(f"收到 {len(bottom_ocr_data_list)} 个bottom OCR数据")

    # 2. 提取 triple_factor 中的 bottom 元素
    bottom_elements = extract_top_elements(triple_factor)
    print(f"找到 {len(bottom_elements)} 个bottom元素")

    if not bottom_elements:
        print("警告: 没有找到bottom元素")
        return span

    # 3. 元素分类：有箭头 vs 无箭头
    # 优先匹配有箭头的元素，因为它们通常更具有几何代表性
    bottom_with_arrow = []
    bottom_without_arrow = []

    for element in bottom_elements:
        # 注意：这里检查 arrow_pairs 是否存在且非空
        if element.get('arrow_pairs') is not None:
            bottom_with_arrow.append(element)
        else:
            bottom_without_arrow.append(element)

    print(f"分类完成 - 有箭头: {len(bottom_with_arrow)} 个, 无箭头: {len(bottom_without_arrow)} 个")

    # 4. 执行匹配逻辑 (OCR -> Element)
    all_b_elements = []
    position_tolerance = 2.0  # 坐标容差
    matched_count = 0

    print(f"开始匹配OCR数据和bottom元素...")

    for ocr_data in bottom_ocr_data_list:
        ocr_loc = ocr_data.get('location')
        max_medium_min = ocr_data.get('max_medium_min', [])

        # 校验 OCR 数据有效性
        if ocr_loc is None or len(ocr_loc) != 4:
            continue

        # 格式化 max_medium_min
        if isinstance(max_medium_min, np.ndarray):
            max_medium_min = max_medium_min.tolist()

        matched = False
        matched_element = None
        # 4.1 优先级 1: 尝试匹配有箭头的元素
        for element in bottom_with_arrow:
            el_loc = element.get('location')
            if el_loc and len(el_loc) == 4:
                # 检查四个坐标是否都在容差范围内
                if all(abs(ocr_loc[i] - el_loc[i]) < position_tolerance for i in range(4)):
                    matched = True
                    matched_element = element
                    # print(f"匹配成功(有箭头): OCR {ocr_loc} -> Element {el_loc}")
                    break
        # 4.2 优先级 2: 如果未匹配，尝试匹配无箭头的元素
        if not matched:
            for element in bottom_without_arrow:
                el_loc = element.get('location')
                if el_loc and len(el_loc) == 4:
                    if all(abs(ocr_loc[i] - el_loc[i]) < position_tolerance for i in range(4)):
                        matched = True
                        matched_element = element
                        # print(f"匹配成功(无箭头): OCR {ocr_loc} -> Element {el_loc}")
                        break
        # 4.3 构建融合数据
        if matched and matched_element:
            # --- 动态计算方向 ---
            # 默认使用元素自带的 direction
            calculated_direction = matched_element.get('direction', '')
            arrow_pairs_data = matched_element.get('arrow_pairs', [])

            # 如果有箭头数据，尝试通过几何形状计算方向 (水平/垂直)
            if arrow_pairs_data is not None and len(arrow_pairs_data) >= 4:
                try:
                    # 假设 arrow_pairs 前4个值为 x1, y1, x2, y2 (引线端点)
                    ax1, ay1, ax2, ay2 = map(float, arrow_pairs_data[:4])
                    adx, ady = abs(ax2 - ax1), abs(ay2 - ay1)

                    if adx > ady:
                        calculated_direction = 'horizontal'
                    elif ady > adx:
                        calculated_direction = 'vertical'
                except (ValueError, IndexError):
                    pass  # 计算失败则保持原有 direction

            # 添加到结果列表
            all_b_elements.append({
                'location': matched_element['location'],
                'direction': calculated_direction,
                'arrow_pairs': matched_element.get('arrow_pairs', None),
                'max_medium_min': max_medium_min
            })
            matched_count += 1

            # --- 关键步骤：移除已匹配元素 (防止重复匹配) ---
            if matched_element in bottom_with_arrow:
                bottom_with_arrow.remove(matched_element)
            elif matched_element in bottom_without_arrow:
                bottom_without_arrow.remove(matched_element)

    print(f"匹配完成，共生成 {len(all_b_elements)} 个融合数据")


    if not all_b_elements:
        print("警告: 未找到匹配的融合元素")
        return span

    # 3. 准备筛选参数
    target_candidates = []
    target_reference_value = []

    horizontal_keys = ['horizontal']
    vertical_keys = ['vertical']

    if direction == "水平方向":
        target_candidates = [e for e in all_b_elements if e['direction'].lower() in vertical_keys]
        target_reference_value = bottom_D
        print(f"执行水平逻辑: 参考值(bottom_D)={target_reference_value}")

    elif direction == "垂直方向":
        target_candidates = [e for e in all_b_elements if e['direction'].lower() in horizontal_keys]
        target_reference_value = bottom_E
        print(f"执行垂直逻辑: 参考值(bottom_E)={target_reference_value}")

    else:
        print(f"错误: 未知方向 '{direction}'")
        return span

    if not target_candidates:
        print(f"该方向下无候选元素")
        return span
    print("target_candidates",target_candidates)
    # 4. 【核心逻辑】
    # Step 4.1: 尝试寻找基准数据 Data1
    data1_element = None
    for cand in target_candidates:
        cand_val = cand['max_medium_min']
        try:
            if np.allclose(cand_val, target_reference_value, rtol=1e-5, atol=1e-5):
                data1_element = cand
                break
        except:
            if cand_val == target_reference_value:
                data1_element = cand
                break

    # Step 4.2: 兜底逻辑 (如果找不到 Data1)
    if data1_element is None:
        print(f"警告: 未在候选列表中找到与参考值 {target_reference_value} 匹配的 Data1")
        print(">>> 触发兜底逻辑: 直接选取当前方向物理长度最大的元素作为 Span")

        if target_candidates:
            # 直接按物理长度降序排序
            target_candidates.sort(key=lambda x: get_arrow_length(x), reverse=True)

            best_candidate = target_candidates[0]
            max_len = get_arrow_length(best_candidate)

            if max_len > 0:
                span = best_candidate['max_medium_min'].copy()
                print(f"兜底成功: 选取最大物理长度={max_len:.2f} 的元素")
                print(f"兜底 Span 值: {span}")
            else:
                print("兜底失败: 候选元素物理长度均为无效值")

        return span

    # Step 4.3: 正常逻辑 (Data1 存在)
    data1_arrow_len = get_arrow_length(data1_element)
    print(f"找到基准数据 Data1: 值={data1_element['max_medium_min']}, 物理长度={data1_arrow_len:.2f}")

    # 筛选出 物理长度 > Data1物理长度 的所有数据
    span_candidates = []
    for cand in target_candidates:
        cand_len = get_arrow_length(cand)
        if cand_len > data1_arrow_len:
            span_candidates.append({
                'element': cand,
                'length': cand_len
            })
    print(f"筛选出 {span_candidates} ")
    # 排序并取最大
    if span_candidates:
        print(f"找到 {len(span_candidates)} 个物理长度大于 Data1 的潜在 Span 数据")
        span_candidates.sort(key=lambda x: x['length'], reverse=True)
        best_candidate = span_candidates[0]
        span = best_candidate['element']['max_medium_min'].copy()
        print(f"筛选结果: 最大物理长度={best_candidate['length']:.2f} (Data1长度={data1_arrow_len:.2f})")
        print(f"最终确定的 Span 值: {span}")
    else:
        print("未找到物理长度大于 Data1 的数据，返回 Data1 作为 Span")
        span = data1_element['max_medium_min'].copy()
        print(f"Span 值: {span}")

    print(f"=== extract_span_dimensions 执行结束 ===\n")
    return span


def extract_pin_dimensions(pin_boxs, bottom_ocr_data_list, triple_factor, direction,key):
    """
    从bottom视图提取与pin相关的尺寸数据，并根据方向筛选。

    参数:
    pin_boxs: pin角坐标，只有一个框[x1, y1, x2, y2]
    bottom_ocr_data_list: OCR检测数据列表
    triple_factor: 嵌套的视图数据
    direction: 当前处理的方向 ('水平方向' 或 '垂直方向')

    返回:
    bottom_b: 短边方向尺寸数组 [最大, 标准, 最小]
    bottom_L: 长边方向尺寸数组 [最大, 标准, 最小]
    """

    def extract_top_elements(data):
        """递归提取view_name为'top'或'bottom'的元素"""
        top_elements = []

        if isinstance(data, dict):
            if (key == 0):
                if data.get('view_name') == 'bottom':
                    top_elements.append(data)
                for value in data.values():
                    if isinstance(value, (dict, list)):
                        top_elements.extend(extract_top_elements(value))
            else:
                if data.get('view_name') == 'side':
                    top_elements.append(data)
                for value in data.values():
                    if isinstance(value, (dict, list)):
                        top_elements.extend(extract_top_elements(value))
        elif isinstance(data, list):
            for item in data:
                top_elements.extend(extract_top_elements(item))

        return top_elements

    print(f"=== extract_pin_dimensions 开始执行 (Target Direction: {direction}) ===")

    # 初始化输出值
    bottom_b = [0, 0, 0]
    bottom_L = [0, 0, 0]

    # --- 1. 数据准备 ---
    if not bottom_ocr_data_list:
        print("警告: bottom_ocr_data_list为空，返回默认值")
        return bottom_b, bottom_L

    bottom_elements = extract_top_elements(triple_factor)
    if not bottom_elements:
        print("警告: 没有找到bottom元素，返回默认值")
        return bottom_b, bottom_L

    # 分类
    bottom_with_arrow = [
        e for e in bottom_elements
        if e.get('arrow_pairs') is not None and len(e.get('arrow_pairs')) > 0
    ]
    bottom_without_arrow = [
        e for e in bottom_elements
        if e.get('arrow_pairs') is None or len(e.get('arrow_pairs')) == 0
    ]

    # --- 2. OCR 与 Structure 匹配 (生成 all_b_elements) ---
    all_b_elements = []
    position_tolerance = 5.0

    for ocr_data in bottom_ocr_data_list:
        ocr_location = ocr_data.get('location')
        max_medium_min = ocr_data.get('max_medium_min', [])

        if ocr_location is None or len(ocr_location) != 4:
            continue

        if isinstance(max_medium_min, np.ndarray):
            max_medium_min = max_medium_min.tolist()

        matched = False
        matched_element = None

        # 2.1 匹配逻辑 (优先有箭头)
        for el in bottom_with_arrow:
            el_loc = el.get('location', [])
            if len(el_loc) == 4 and all(abs(ocr_location[i] - el_loc[i]) < position_tolerance for i in range(4)):
                matched = True
                matched_element = el
                break

        if not matched:
            for el in bottom_without_arrow:
                el_loc = el.get('location', [])
                if len(el_loc) == 4 and all(abs(ocr_location[i] - el_loc[i]) < position_tolerance for i in range(4)):
                    matched = True
                    matched_element = el
                    break

        # 2.2 构建元素并计算方向
        if matched and matched_element:
            # 动态计算方向 (Geometric Check)
            calculated_direction = matched_element.get('direction', '')
            arrow_pairs_data = matched_element.get('arrow_pairs', None)

            if arrow_pairs_data is not None and len(arrow_pairs_data) >= 4:
                try:
                    ax1, ay1, ax2, ay2 = map(float, arrow_pairs_data[:4])
                    adx, ady = abs(ax2 - ax1), abs(ay2 - ay1)
                    if adx > ady:
                        calculated_direction = 'horizontal'
                    elif ady > adx:
                        calculated_direction = 'vertical'
                except:
                    pass

            b_element = {
                'location': matched_element['location'],
                'direction': str(calculated_direction).lower(),  # 统一转小写
                'arrow_pairs': arrow_pairs_data,
                'max_medium_min': max_medium_min
            }
            all_b_elements.append(b_element)

            # 移除已匹配项防止重复
            if matched_element in bottom_with_arrow:
                bottom_with_arrow.remove(matched_element)
            elif matched_element in bottom_without_arrow:
                bottom_without_arrow.remove(matched_element)

    if not all_b_elements:
        print("警告: 没有找到匹配的B元素，返回默认值")
        return bottom_b, bottom_L

    # --- 3. 方向筛选 (关键修改) ---
    print(f"匹配完成，共 {len(all_b_elements)} 个元素。开始根据方向 '{direction}' 筛选...")

    horizontal_keys = ['horizontal']
    vertical_keys = ['vertical']

    all_b_candidates = []

    if direction == "水平方向":
        all_b_candidates = [e for e in all_b_elements if any(k in e['direction'] for k in horizontal_keys)]
        print(f"筛选水平元素: {len(all_b_candidates)} 个")
    elif direction == "垂直方向":
        all_b_candidates = [e for e in all_b_elements if any(k in e['direction'] for k in vertical_keys)]
        print(f"筛选垂直元素: {len(all_b_candidates)} 个")
    else:
        print(f"方向未知 ({direction})，不做筛选")
        all_b_candidates = all_b_elements

    # 兜底：如果筛选后为空（可能是方向判断错误），回退到使用所有元素
    if not all_b_candidates:
        print("警告: 按方向筛选后列表为空，回退到使用所有匹配元素")
        all_b_candidates = all_b_elements

    # 按标准值排序候选列表
    all_b_candidates.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0)

    print(f"all_b_candidates: {all_b_candidates} ")
    # 检查pin_boxs是否存在
    if pin_boxs is None or len(pin_boxs) == 0:
        print("警告: pin_boxs为空，使用标准值排序方法")
        # 使用排序后第一个元素的max_medium_min作为bottom_b
        if all_b_candidates:
            bottom_b = all_b_candidates[0]['max_medium_min'].copy()
            print(f"bottom_b使用第一个元素: max_medium_min={bottom_b}")

        # bottom_L使用最后一个元素的max_medium_min（如果存在且大于bottom_b），否则使用第二个
        if len(all_b_candidates) >= 2:
            # 判断最后一个元素的标准值是否大于第一个元素
            last_std = all_b_candidates[-1]['max_medium_min'][1] if len(all_b_candidates[-1]['max_medium_min']) > 1 else 0
            first_std = all_b_candidates[0]['max_medium_min'][1] if len(all_b_candidates[0]['max_medium_min']) > 1 else 0

            if last_std > first_std:
                bottom_L = all_b_candidates[-1]['max_medium_min'].copy()
                print(f"bottom_L使用最后一个元素: max_medium_min={bottom_L}")
            else:
                bottom_L = all_b_candidates[1]['max_medium_min'].copy()
                print(f"bottom_L使用第二个元素: max_medium_min={bottom_L}")
        elif all_b_candidates:
            bottom_L = all_b_candidates[0]['max_medium_min'].copy()
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
        if all_b_candidates:
            bottom_b = all_b_candidates[0]['max_medium_min'].copy()
            if len(all_b_candidates) >= 2:
                bottom_L = all_b_candidates[-1]['max_medium_min'].copy() if all_b_candidates[-1]['max_medium_min'][1] > \
                                                                          all_b_candidates[0]['max_medium_min'][1] else \
                all_b_candidates[1]['max_medium_min'].copy()
            else:
                bottom_L = all_b_candidates[0]['max_medium_min'].copy()

        return bottom_b, bottom_L

    # 开始与pin_boxs尺寸进行比对
    print("开始与pin_boxs尺寸进行比对...")
    best_short_match = None
    best_long_match = None
    min_short_diff = float('inf')
    min_long_diff = float('inf')

    # 优先选择有arrow_pairs的元素
    for idx, element in enumerate(all_b_candidates):
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
        for idx, element in enumerate(all_b_candidates):
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
        if all_b_candidates:
            bottom_b = all_b_candidates[0]['max_medium_min'].copy()
            print(f"短边使用标准值排序最小: max_medium_min={bottom_b}")

    # 判断长边是否有匹配
    if best_long_match is not None and min_long_diff <= pin_long_threshold:
        # 如果长边匹配的元素与短边匹配的元素相同，且短边已经匹配，则我们需要找另一个元素
        if best_long_match == best_short_match and short_matched:
            print("长边匹配的元素与短边相同，且短边已匹配，为长边寻找次佳匹配")
            # 在剩余元素中寻找与长边最相似的元素
            second_best_long_match = None
            second_min_long_diff = float('inf')

            for idx, element in enumerate(all_b_candidates):
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
                if len(all_b_candidates) >= 2:
                    # 使用排序后的最后一个元素（最大值）
                    bottom_L = all_b_candidates[-1]['max_medium_min'].copy()
                    long_matched = False
                    print(f"长边使用标准值排序最大: max_medium_min={bottom_L}")
                elif all_b_candidates:
                    bottom_L = all_b_candidates[0]['max_medium_min'].copy()
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
        if len(all_b_candidates) >= 2:
            # 使用排序后的最后一个元素（最大值）
            bottom_L = all_b_candidates[-1]['max_medium_min'].copy()
            long_matched = False
            print(f"长边使用标准值排序最大: max_medium_min={bottom_L}")
        elif all_b_candidates:
            bottom_L = all_b_candidates[0]['max_medium_min'].copy()
            long_matched = False
            print(f"长边只有一个元素可用，使用第一个: max_medium_min={bottom_L}")

    print(f"\n最终结果: bottom_b={bottom_b}, bottom_L={bottom_L}")
    print("=== extract_pin_dimensions 执行结束 ===\n")

    return bottom_b, bottom_L



def extract_pitch_dimension(pin_box, bottom_ocr_data_list, triple_factor, direction,key):
    """
    提取pitch尺寸数据
    1. 融合 OCR 和 TripleFactor 数据
    2. 根据传入的 direction 参数筛选候选元素 (水平/垂直)
    3. 尝试利用 pin_box 计算的间距进行精确匹配
    4. 兜底逻辑：按引线长度排序取第2小
    """

    # --- 内部辅助函数 ---
    def get_sort_value(element):
        """获取用于排序的标准值（中间值）"""
        vals = element.get('max_medium_min', [])
        return vals[1] if len(vals) > 1 else (vals[0] if len(vals) > 0 else 0)

    def get_arrow_length(element):
        """获取用于排序的引线长度（arrow_pairs最后一个值）"""
        pairs = element.get('arrow_pairs')
        if pairs is not None and len(pairs) > 0:
            try:
                return float(pairs[-1])
            except (ValueError, TypeError):
                return float('inf')  # 格式错误放到最后
        return float('inf')  # 没有引线数据的放到最后

    def extract_top_elements(data):
        """递归提取view_name为'top'或'bottom'的元素"""
        top_elements = []

        if isinstance(data, dict):
            if (key == 0):
                if data.get('view_name') == 'bottom':
                    top_elements.append(data)
                for value in data.values():
                    if isinstance(value, (dict, list)):
                        top_elements.extend(extract_top_elements(value))
            else:
                if data.get('view_name') == 'side':
                    top_elements.append(data)
                for value in data.values():
                    if isinstance(value, (dict, list)):
                        top_elements.extend(extract_top_elements(value))
        elif isinstance(data, list):
            for item in data:
                top_elements.extend(extract_top_elements(item))

        return top_elements

    print(f"=== extract_pitch_dimension 开始执行 (Target Direction: {direction}) ===")

    # 初始化输出值
    pitch = [0, 0, 0]

    # 1. 基础数据检查
    if not bottom_ocr_data_list:
        print("警告: bottom_ocr_data_list为空，返回默认值")
        return pitch

    # 提取结构元素
    bottom_elements = extract_top_elements(triple_factor)
    if not bottom_elements:
        print("警告: 没有找到bottom元素，返回默认值")
        return pitch

    # 分类
    bottom_with_arrow = [
        e for e in bottom_elements
        if e.get('arrow_pairs') is not None and len(e.get('arrow_pairs')) > 0
    ]
    bottom_without_arrow = [
        e for e in bottom_elements
        if e.get('arrow_pairs') is None or len(e.get('arrow_pairs')) == 0
    ]

    # 2. 融合 OCR 和 Structural 数据 (生成 all_b_elements)
    all_b_elements = []
    position_tolerance = 2.0

    print(f"开始匹配 OCR ({len(bottom_ocr_data_list)}个) 与 Bottom元素...")
    matched_count = 0

    for ocr_data in bottom_ocr_data_list:
        ocr_location = ocr_data.get('location')
        max_medium_min = ocr_data.get('max_medium_min', [])

        if ocr_location is None or len(ocr_location) != 4:
            continue

        if isinstance(max_medium_min, np.ndarray):
            max_medium_min = max_medium_min.tolist()

        matched = False
        matched_element = None

        # 优先匹配有箭头
        for el in bottom_with_arrow:
            el_loc = el.get('location', [])
            if len(el_loc) == 4 and all(abs(ocr_location[i] - el_loc[i]) < position_tolerance for i in range(4)):
                matched = True
                matched_element = el
                break

        # 其次匹配无箭头
        if not matched:
            for el in bottom_without_arrow:
                el_loc = el.get('location', [])
                if len(el_loc) == 4 and all(abs(ocr_location[i] - el_loc[i]) < position_tolerance for i in range(4)):
                    matched = True
                    matched_element = el
                    break

        if matched and matched_element:
            # --- 方向判定/修正逻辑 ---
            # 默认使用原有的 direction
            calculated_direction = matched_element.get('direction', '')
            arrow_pairs_data = matched_element.get('arrow_pairs', None)

            # 尝试通过几何计算修正方向
            if arrow_pairs_data is not None and len(arrow_pairs_data) >= 4:
                try:
                    ax1, ay1, ax2, ay2 = map(float, arrow_pairs_data[:4])
                    adx, ady = abs(ax2 - ax1), abs(ay2 - ay1)
                    if adx > ady:
                        calculated_direction = 'horizontal'
                    elif ady > adx:
                        calculated_direction = 'vertical'
                except:
                    pass

            b_element = {
                'location': matched_element['location'],
                'direction': str(calculated_direction).lower(),  # 转小写方便后续匹配
                'arrow_pairs': arrow_pairs_data,
                'max_medium_min': max_medium_min
            }
            all_b_elements.append(b_element)
            matched_count += 1

            # 移除已匹配项
            if matched_element in bottom_with_arrow:
                bottom_with_arrow.remove(matched_element)
            elif matched_element in bottom_without_arrow:
                bottom_without_arrow.remove(matched_element)

    if not all_b_elements:
        print("警告: 未找到匹配的融合元素")
        return pitch

    # --- 3. 基于 direction 参数筛选候选元素 (核心修改) ---
    print(f"根据传入方向筛选候选元素: {direction}")
    print(f"all_b_elements:{all_b_elements}")

    horizontal_keys = ['horizontal']
    vertical_keys = ['vertical']

    candidates = []

    if direction == "水平方向":
        # 筛选方向字段中包含水平关键字的元素
        candidates = [e for e in all_b_elements if any(k in e['direction'] for k in horizontal_keys)]
    elif direction == "垂直方向":
        # 筛选方向字段中包含垂直关键字的元素
        candidates = [e for e in all_b_elements if any(k in e['direction'] for k in vertical_keys)]
    else:
        # 如果方向未知，保留所有元素
        print(f"方向 '{direction}' 未知，保留所有元素")
        candidates = all_b_elements

    print(f"candidates:{candidates}")

    # 兜底：如果筛选后为空（可能是方向判断失误），为了不报错，回退到全部元素
    if not candidates:
        print("警告: 按方向筛选后候选列表为空，回退到所有匹配元素")
        candidates = all_b_elements

    # --- 4. 计算 Pin Box 间距 (仅用于数值比对，不用于方向筛选) ---
    target_pin_distance = 0.0

    if pin_box is not None:
        try:
            # 解析 pin_box (兼容 [[x1,y1,x2,y2], [x3,y3,x4,y4]] 或 flatten 格式)
            box1, box2 = None, None
            if isinstance(pin_box, list):
                if len(pin_box) == 2 and isinstance(pin_box[0], (list, np.ndarray)):
                    box1, box2 = pin_box[0], pin_box[1]
                elif len(pin_box) >= 8 and all(isinstance(x, (int, float)) for x in pin_box[:8]):
                    box1, box2 = pin_box[:4], pin_box[4:8]

            if box1 is not None and box2 is not None:
                c1_x, c1_y = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
                c2_x, c2_y = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
                dx, dy = abs(c2_x - c1_x), abs(c2_y - c1_y)

                # 计算两框中心距离
                # 注意：这里我们使用 direction 参数来决定取 dx 还是 dy
                if direction == "水平方向":
                    target_pin_distance = dx
                elif direction == "垂直方向":
                    target_pin_distance = dy
                else:
                    target_pin_distance = max(dx, dy)  # 默认取较大值

                print(f"PinBox 计算出的目标 Pitch ({direction}): {target_pin_distance:.2f}")

        except Exception as e:
            print(f"Pin Box 计算出错: {e}")

    # --- 5. 匹配或排序逻辑 ---

    found_match = False
    similarity_threshold = 0.2

    # A. 尝试通过引线距离匹配 (Match)
    if target_pin_distance > 0:
        best_match = None
        min_diff = float('inf')

        # 仅在有引线的候选中查找
        valid_match_candidates = [
            c for c in candidates
            if c.get('arrow_pairs') is not None and len(c.get('arrow_pairs')) > 0
        ]

        for e in valid_match_candidates:
            try:
                # arrow_pairs 最后一个值通常是长度
                val = float(e['arrow_pairs'][-1])
                diff = abs(val - target_pin_distance)
                if diff < min_diff:
                    min_diff = diff
                    best_match = e
            except:
                continue

        limit = target_pin_distance * similarity_threshold
        if best_match and min_diff <= limit:
            pitch = best_match['max_medium_min'].copy()
            found_match = True
            print(f"PinBox匹配成功! 差异 {min_diff:.2f} <= 阈值 {limit:.2f}. Pitch={pitch}")
        else:
            print(f"PinBox匹配失败 (最小差异 {min_diff:.2f})")

    # B. 排序兜底 (Sort)
    if not found_match:
        print("执行排序逻辑: 按引线长度(arrow_pairs)排序...")

        # 按照引线长度排序 (无引线为 inf)
        candidates.sort(key=get_arrow_length)

        # 过滤掉无引线数据的 (inf)
        valid_candidates = [c for c in candidates if get_arrow_length(c) != float('inf')]

        # 打印调试
        debug_vals = [f"{get_arrow_length(c):.2f}" for c in valid_candidates]
        print(f"有效候选引线长度序列: {debug_vals}")

        if not valid_candidates:
            print("没有有效的带引线数据，退化为按OCR数值排序")
            candidates.sort(key=get_sort_value)
            if candidates:
                pitch = candidates[0]['max_medium_min'].copy()

        elif len(valid_candidates) >= 2:
            # 取第 2 小 (索引 1)
            pitch = valid_candidates[1]['max_medium_min'].copy()
            print(f"取第2小的引线长度对应数值 (L={get_arrow_length(valid_candidates[1]):.2f})")

        else:
            # 只有 1 个，直接取
            pitch = valid_candidates[0]['max_medium_min'].copy()
            print(f"仅有1个有效数据，直接选取 (L={get_arrow_length(valid_candidates[0]):.2f})")

    print(f"最终结果: pitch={pitch}")
    print("=== extract_pitch_dimension 执行结束 ===\n")
    return pitch


def get_integrated_parameter_list(
        L3: List[Any],
        calc_results: Dict[str, List[float]]
) -> List[Dict]:
    """
    结合精确计算结果和OCR数值筛选，生成最终参数列表。

    :param L3: 包含 OCR 数据的大列表
    :param calc_results: 字典，包含精确计算出的 {'D':[], 'E1':[], 'E':[], 'D2':[], 'E2':[]}
    :return: 整合后的参数列表
    """

    # 1. 准备基础数据源
    top_ocr_data = find_list(L3, "top_ocr_data") or []
    bottom_ocr_data = find_list(L3, "bottom_ocr_data") or []
    side_ocr_data = find_list(L3, "side_ocr_data") or []
    detailed_ocr_data = find_list(L3, "detailed_ocr_data") or []

    # 2. 初始化参数字典结构
    # 定义辅助函数快速创建结构
    def create_param(name):
        return {'parameter_name': name, 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}

    # 初始化列表 (顺序保持原逻辑一致)
    # 顺序: 0:D, 1:E1, 2:E, 3:A, 4:A1, 5:e, 6:b, 7:D2, 8:E2,
    #       9:L, 10:GAGE, 11:c, 12:θ, 13:θ1, 14:θ2, 15:θ3, 16:Φ
    params = [
        create_param('D'), create_param('E1'), create_param('E'),
        create_param('A'), create_param('A1'), create_param('e'), create_param('b'),
        create_param('D2'), create_param('E2'), create_param('L'), create_param('GAGE_PLANE'),
        create_param('c'), create_param('θ'), create_param('θ1'), create_param('θ2'),
        create_param('θ3'), create_param('Φ')
    ]

    # 建立名字到索引的映射，方便操作
    p_map = {p['parameter_name']: i for i, p in enumerate(params)}

    # 3. 【第一步】填入精确计算出的值 (D, E, D1, E1, D2, E2)
    # 这些值是你代码中通过 extract_top_D_E 等函数算出来的，最准，直接由外部传入
    # 我们把它们包装成类似 OCR 的格式 {'max_medium_min': [val, val, val], 'source': 'calc'}

    priority_keys = ['D', 'E1', 'E', 'D2', 'E2','L','b','e']

    for key in priority_keys:
        val_list = calc_results.get(key, [])
        # 只有当计算结果有效（不是全0或空）时才填入
        if val_list and any(v != 0 for v in val_list):
            idx = p_map[key]
            # 构造一个模拟数据的对象
            mock_data = {
                'max_medium_min': val_list,
                'Absolutely': f'Calculated_{key}',
                'confidence': 1.0
            }
            params[idx]['maybe_data'].append(mock_data)
            params[idx]['maybe_data_num'] = 1
            params[idx]['OK'] = 1  # 标记为已由精确算法解决

    # 4. 【第二步】对其余参数使用区间筛选 (A, e, b, L, θ, Φ...)
    # 定义阈值 (保留你原代码的阈值)
    ranges = {
        'A': (1.0, 4.5), 'A1': (0, 0.4), 'e': (0.30, 1.3), 'b': (0.13, 0.83),
        'L': (0.4, 0.7), 'GAGE_PLANE': (0.25, 0.25), 'c': (0.10, 0.20),
        'θ': (0, 10), 'θ1': (0, 14), 'θ2': (11, 16), 'θ3': (11, 16), 'Φ': (0.4, 0.8)
    }

    # 定义筛选辅助函数
    def check_and_add(ocr_list, param_name, constraints=None, abs_check=None):
        idx = p_map[param_name]
        # 如果这个参数已经被精确算法填过了(比如D/E)，这里就跳过，防止杂乱数据混入
        if params[idx]['OK'] == 1:
            return

        min_v, max_v = ranges.get(param_name, (0, 0))

        for item in ocr_list:
            mmm = item.get('max_medium_min', [])
            if len(mmm) < 3: continue
            val_max, val_med, val_min = mmm[0], mmm[1], mmm[2]

            # 基础区间判断
            if min_v <= val_min and val_max <= max_v:
                # 额外的绝对标签检查 (Absolutely)
                if abs_check:
                    if item.get('Absolutely') not in abs_check:
                        continue
                # 排除特定的标签 (比如把 pin_diameter 排除出 b 的筛选)
                if param_name == 'b':
                    if item.get('Absolutely') in ['pin_diameter', 'mb_pin_diameter', 'pin_diameter+']:
                        continue

                params[idx]['maybe_data'].append(item)
                params[idx]['maybe_data_num'] += 1

    # --- 开始遍历筛选 ---

    # 4.1 Side View 筛选 (A, A1, e, b, θ2)
    check_and_add(side_ocr_data, 'A')
    check_and_add(side_ocr_data, 'e')
    check_and_add(side_ocr_data, 'b')
    check_and_add(side_ocr_data, 'c')
    # 特殊处理 θ2 (angle)
    for item in side_ocr_data:
        if item.get('Absolutely') == 'angle':
            mmm = item.get('max_medium_min', [])
            # 必须先判断不为 None，再判断长度，最后才能取值比大小
            if mmm is not None and len(mmm) >= 3 and 11 <= mmm[2] and mmm[0] <= 16:
                params[p_map['θ2']]['maybe_data'].append(item)
                params[p_map['θ2']]['maybe_data_num'] += 1

    # 4.2 Detailed View 筛选 (A, L, GAGE, c, angles)
    check_and_add(detailed_ocr_data, 'A1')
    check_and_add(detailed_ocr_data, 'L')
    check_and_add(detailed_ocr_data, 'GAGE_PLANE')
    check_and_add(detailed_ocr_data, 'c')

    # Detailed View 的角度处理
    for item in detailed_ocr_data:
        if item.get('Absolutely') == 'angle':
            mmm = item.get('max_medium_min', [])
            if mmm is None or len(mmm) == 0: continue
            # θ
            if 0 <= mmm[2] and mmm[0] <= 10:
                params[p_map['θ']]['maybe_data'].append(item)
                params[p_map['θ']]['maybe_data_num'] += 1
            # θ1
            if 0 <= mmm[2] and mmm[0] <= 14:
                params[p_map['θ1']]['maybe_data'].append(item)
                params[p_map['θ1']]['maybe_data_num'] += 1
            # θ2
            if 11 < mmm[2] and mmm[0] <= 16:  # 注意你的原代码这里是 <
                params[p_map['θ2']]['maybe_data'].append(item)
                params[p_map['θ2']]['maybe_data_num'] += 1
            # θ3
            if 11 <= mmm[2] and mmm[0] <= 16:
                params[p_map['θ3']]['maybe_data'].append(item)
                params[p_map['θ3']]['maybe_data_num'] += 1

    # 4.3 Bottom/Top View 筛选 (e, b, Φ)
    # 注意：Top/Bottom 的 D, E, D1, E1, D2, E2 已经被精确算法接管，这里只看剩下的
    for ocr_list in [top_ocr_data, bottom_ocr_data]:
        check_and_add(ocr_list, 'e')
        check_and_add(ocr_list, 'b')

    # 特殊处理 Φ (pin_diameter) - 只在 Bottom
    phi_keywords = ['pin_diameter', 'mb_pin_diameter', 'pin_diameter+']
    check_and_add(bottom_ocr_data, 'Φ', abs_check=phi_keywords)

    # 5. 去重逻辑 (保留原逻辑)
    for i in range(len(params)):
        maybe = params[i].get("maybe_data", [])
        seen = set()
        new_maybe = []
        for item in maybe:
            mmm = item.get("max_medium_min")
            if mmm is None: continue
            # 将 numpy array 转换为 tuple 以便 hash
            if isinstance(mmm, np.ndarray):
                key = tuple(mmm.tolist())
            else:
                key = tuple(mmm)

            if key in seen: continue
            seen.add(key)
            new_maybe.append(item)

        params[i]["maybe_data"] = new_maybe
        params[i]["maybe_data_num"] = len(new_maybe)

    return params


###############################################################################

def extract_SOP(package_classes, page_num):
    # 完成图片大小固定、清空建立文件夹等各种操作
    common_pipeline.prepare_workspace(
        DATA,
        DATA_COPY,
        DATA_BOTTOM_CROP,
        ONNX_OUTPUT,
        OPENCV_OUTPUT,
    )
    test_mode = 1  # 0: 正常模式，1: 测试模式
    key = test_mode
    '''
        默认图片型封装
    '''
    letter_or_number = 'number'
    '''
    YOLO检测
    DBnet检测
    SVTR识别
    数据整理
    输出参数
    '''
    # (1)在各个视图中用yolox识别图像元素LOCATION，dbnet识别文本location
    L3 = common_pipeline.get_data_location_by_yolo_dbnet(DATA, package_classes)

    # (2)在yolo和dbnet的标注文本框中去除OTHER类型文本框
    L3 = common_pipeline.remove_other_annotations(L3)

    # (3)为尺寸线寻找尺寸界限
    L3 = common_pipeline.enrich_pairs_with_lines(L3, DATA, key)

    # 处理数据
    L3 = common_pipeline.preprocess_pairs_and_text(L3, key)

    # (4)SVTR识别标注内容
    L3 = common_pipeline.run_svtr_ocr(L3)

    # (5)SVTR后处理数据
    L3 = common_pipeline.normalize_ocr_candidates(L3, key)

    # (6)提取并分离出yolo和dbnet检测出的标注中的序号
    L3 = common_pipeline.extract_pin_serials(L3, package_classes)

    # (7)匹配pairs和data
    L3 = common_pipeline.match_pairs_with_text(L3, key)

    # 处理数据
    L3 = common_pipeline.finalize_pairs(L3)

    '''
        输出QFP参数
        nx,ny
        pitch
        high(A)
        standoff(A1)
        span_x,span_y
        body_x,body_y
        b
        pad_x,pad_y
    '''
    # # 语义对齐
    SOP_parameter_list, nx, ny = find_SOP_parameter(L3)
    # 20250722添加
    # 指定要查找的 page_num
    target_page_num = page_num
    json_file = 'package_baseinfo.json'
    result = []
    # 读取 JSON 文件
    print("开始读取json文件")
    # with open(json_file, 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    with open(json_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content:
            try:
                data = json.loads(content)
                print("解析成功")
                for item in data:
                    if item['page_num'] == target_page_num:
                        result.append(item['pin'])
                        result.append(item['length'])
                        result.append(item['width'])
                        result.append(item['height'])
                        result.append(item['horizontal_pin'])
                        result.append(item['vertical_pin'])
                print("json文件读取完毕")
                print("json:", result)
            except json.JSONDecodeError as e:
                print("JSON 解析失败:", e)
        else:
            print("文件为空")
    # 遍历列表，查找匹配的条目

    if result != []:
        if result[0] != None:
            if result[4] != None and result[5] != None:
                if abs(result[4] * result[5] - result[0]) < 1e-9 and abs(nx * ny - result[4] * result[5]) > 1e-9:
                    nx = result[4]
                    ny = result[5]
                if nx == 0 and result[4] != None:
                    nx = result[4]
                if ny == 0 and result[5] != None:
                    ny = result[5]
    # # 整理获得的参数
    parameter_list = get_SOP_parameter_data(SOP_parameter_list, nx, ny)
    # if result != []:
    #     if result[1] != None:
    #         parameter_list[0][1] = result[1]
    #         parameter_list[0][2] = result[1]
    #         parameter_list[0][3] = result[1]
    #     if result[2] != None:
    #         parameter_list[1][1] = result[2]
    #         parameter_list[1][2] = result[2]
    #         parameter_list[1][3] = result[2]
    #     if result[3] != None:
    #         parameter_list[2][1] = result[3]
    #         parameter_list[2][2] = result[3]
    #         parameter_list[2][3] = result[3]
    try:
        length = float(parameter_list[0][2])
    except:
        print("无法转化为浮点数length", parameter_list[0][2])
    try:
        weight = float(parameter_list[1][2])
    except:
        print("无法转化为浮点数weight", parameter_list[1][2])
    try:
        height = float(parameter_list[2][2])
    except:
        print("无法转化为浮点数height", parameter_list[2][2])
    if result != []:
        if result[1] != None and result[1] != length and (result[1] != weight and result[2] != length):
            parameter_list[0][1] = ''
            parameter_list[0][2] = result[1]
            parameter_list[0][3] = ''
        if result[2] != None and result[2] != weight and (result[1] != weight and result[2] != length):
            parameter_list[1][1] = ''
            parameter_list[1][2] = result[2]
            parameter_list[1][3] = ''
        if result[3] != None and result[3] != height:
            parameter_list[2][1] = ''
            parameter_list[2][2] = result[3]
            parameter_list[2][3] = ''
    print(parameter_list)

    return parameter_list


def find_SOP_parameter(L3):
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

    # (9)输出序号nx,ny和body_x、body_y
    nx, ny = get_serial(top_serial_numbers_data, bottom_serial_numbers_data)
    body_x, body_y = get_body(yolox_pairs_top, top_yolox_pairs_length, yolox_pairs_bottom,
                              bottom_yolox_pairs_length, top_border, bottom_border, top_ocr_data,
                              bottom_ocr_data)
    span_x = get_span(top_ocr_data, bottom_ocr_data, False)
    print("1111111111", span_x)
    # (10)初始化参数列表
    QFP_parameter_list = get_SOP_parameter_list(top_ocr_data, bottom_ocr_data, side_ocr_data, detailed_ocr_data,
                                                body_x, body_y)
    # (11)整理参数列表
    QFP_parameter_list = resort_parameter_list_2(QFP_parameter_list)
    # 输出高

    if len(QFP_parameter_list[4]) > 1:
        high = get_QFP_high(QFP_parameter_list[4]['maybe_data'])
        if len(high) > 0:
            QFP_parameter_list[4]['maybe_data'] = high
            QFP_parameter_list[4]['maybe_data_num'] = len(high)
    # 输出pitch
    if len(QFP_parameter_list[5]['maybe_data']) > 1 or len(QFP_parameter_list[6]['maybe_data']) > 1:
        pitch_x, pitch_y = get_QFP_pitch(QFP_parameter_list[5]['maybe_data'], body_x, body_y, nx, ny)
        if len(pitch_x) > 0:
            QFP_parameter_list[5]['maybe_data'] = pitch_x
            QFP_parameter_list[5]['maybe_data_num'] = len(pitch_x)
        if len(pitch_y) > 0:
            QFP_parameter_list[6]['maybe_data'] = pitch_y
            QFP_parameter_list[6]['maybe_data_num'] = len(pitch_y)
    # 整理参数列表
    QFP_parameter_list = resort_parameter_list_2(QFP_parameter_list)
    # # 补全相同参数的x、y
    # QFP_parameter_list = Completion_QFP_parameter_list(QFP_parameter_list)
    # # 输出参数列表，给出置信度
    # QFP = output_QFP_parameter(QFP_parameter_list, nx, ny)
    return QFP_parameter_list, nx, ny, span_x


def get_SOP_parameter_list(top_ocr_data, bottom_ocr_data, side_ocr_data, detailed_ocr_data, body_x, body_y):
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
    SOP_parameter_list = []
    dic = {'parameter_name': [], 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    dic_D = {'parameter_name': 'D', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    dic_E = {'parameter_name': 'E', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    D_max = 35
    D_min = 8.75
    E_max = D_max
    E_min = D_min

    dic_D1 = {'parameter_name': 'D1', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    dic_E1 = {'parameter_name': 'E1', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    D1_max = 30
    D1_min = 6.9
    E1_max = D1_max
    E1_min = D1_min

    dic_L = {'parameter_name': 'L', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    L_max = 0.75
    L_min = 0.45

    dic_GAGE_PLANE = {'parameter_name': 'GAGE_PLANE', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    GAGE_PLANE_max = 0.25
    GAGE_PLANE_min = 0.25

    dic_c = {'parameter_name': 'c', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    c_max = 0.20
    c_min = 0.09

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

    # dic_D2 = {'parameter_name': 'D2', 'maybe_data': [], 'possible': []}
    # dic_E2 = {'parameter_name': 'E2', 'maybe_data': [], 'possible': []}

    dic_A = {'parameter_name': 'A', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    A_max = 4.5
    A_min = 1.1
    dic_A1 = {'parameter_name': 'A1', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    A1_max = 0.3
    A1_min = 0
    dic_e = {'parameter_name': 'e', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    e_max = 1.3
    e_min = 0.35
    dic_b = {'parameter_name': 'b', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    b_max = 0.83
    b_min = 0.13
    dic_D2 = {'parameter_name': 'D2', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    dic_E2 = {'parameter_name': 'E2', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    D2_max = 7.2
    D2_min = 3.15
    E2_max = 7.2
    E2_min = 3.15
    SOP_parameter_list.append(dic_D)
    SOP_parameter_list.append(dic_E)
    SOP_parameter_list.append(dic_D1)
    SOP_parameter_list.append(dic_E1)
    SOP_parameter_list.append(dic_A)
    SOP_parameter_list.append(dic_A1)
    SOP_parameter_list.append(dic_e)
    SOP_parameter_list.append(dic_b)
    SOP_parameter_list.append(dic_D2)
    SOP_parameter_list.append(dic_E2)
    SOP_parameter_list.append(dic_L)
    SOP_parameter_list.append(dic_GAGE_PLANE)
    SOP_parameter_list.append(dic_c)
    SOP_parameter_list.append(dic_θ)
    SOP_parameter_list.append(dic_θ1)
    SOP_parameter_list.append(dic_θ2)
    SOP_parameter_list.append(dic_θ3)

    for i in range(len(top_ocr_data)):
        if D_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= D_max:
            SOP_parameter_list[0]['maybe_data'].append(top_ocr_data[i])
            SOP_parameter_list[0]['maybe_data_num'] += 1
        if E_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= E_max:
            SOP_parameter_list[1]['maybe_data'].append(top_ocr_data[i])
            SOP_parameter_list[1]['maybe_data_num'] += 1
        if D1_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= D1_max:
            if len(body_x) > 0:
                SOP_parameter_list[2]['maybe_data'] = body_x
            else:
                SOP_parameter_list[2]['maybe_data'].append(top_ocr_data[i])
                SOP_parameter_list[2]['maybe_data_num'] += 1
        if E1_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= E1_max:
            if len(body_y) > 0:
                SOP_parameter_list[3]['maybe_data'] = body_y
            else:
                SOP_parameter_list[3]['maybe_data'].append(top_ocr_data[i])
                SOP_parameter_list[3]['maybe_data_num'] += 1
        if e_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= e_max:
            SOP_parameter_list[6]['maybe_data'].append(top_ocr_data[i])
            SOP_parameter_list[6]['maybe_data_num'] += 1
        if b_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= b_max:
            SOP_parameter_list[7]['maybe_data'].append(top_ocr_data[i])
            SOP_parameter_list[7]['maybe_data_num'] += 1
        if D2_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= D2_max:
            SOP_parameter_list[8]['maybe_data'].append(top_ocr_data[i])
            SOP_parameter_list[8]['maybe_data_num'] += 1
        if E2_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= E2_max:
            SOP_parameter_list[9]['maybe_data'].append(top_ocr_data[i])
            SOP_parameter_list[9]['maybe_data_num'] += 1
    for i in range(len(bottom_ocr_data)):
        if D_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= D_max:
            SOP_parameter_list[0]['maybe_data'].append(bottom_ocr_data[i])
            SOP_parameter_list[0]['maybe_data_num'] += 1
        if E_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= E_max:
            SOP_parameter_list[1]['maybe_data'].append(bottom_ocr_data[i])
            SOP_parameter_list[1]['maybe_data_num'] += 1
        if D1_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= D1_max:
            if len(body_x) > 0:
                SOP_parameter_list[2]['maybe_data'] = body_x
            else:
                SOP_parameter_list[2]['maybe_data'].append(bottom_ocr_data[i])
                SOP_parameter_list[2]['maybe_data_num'] += 1
        if E1_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= E1_max:
            if len(body_y) > 0:
                SOP_parameter_list[3]['maybe_data'] = body_y
            else:
                SOP_parameter_list[3]['maybe_data'].append(bottom_ocr_data[i])
                SOP_parameter_list[3]['maybe_data_num'] += 1
        if e_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= e_max:
            SOP_parameter_list[6]['maybe_data'].append(bottom_ocr_data[i])
            SOP_parameter_list[6]['maybe_data_num'] += 1
        if b_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= b_max:
            SOP_parameter_list[7]['maybe_data'].append(bottom_ocr_data[i])
            SOP_parameter_list[7]['maybe_data_num'] += 1
        if D2_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= D2_max:
            SOP_parameter_list[8]['maybe_data'].append(bottom_ocr_data[i])
            SOP_parameter_list[8]['maybe_data_num'] += 1
        if E2_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= E2_max:
            SOP_parameter_list[9]['maybe_data'].append(bottom_ocr_data[i])
            SOP_parameter_list[9]['maybe_data_num'] += 1
    for i in range(len(side_ocr_data)):
        if A_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= A_max:
            SOP_parameter_list[4]['maybe_data'].append(side_ocr_data[i])
            SOP_parameter_list[4]['maybe_data_num'] += 1
        if A1_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= A1_max:
            SOP_parameter_list[5]['maybe_data'].append(side_ocr_data[i])
            SOP_parameter_list[5]['maybe_data_num'] += 1
        if e_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= e_max:
            SOP_parameter_list[6]['maybe_data'].append(side_ocr_data[i])
            SOP_parameter_list[6]['maybe_data_num'] += 1
        if b_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= b_max:
            SOP_parameter_list[7]['maybe_data'].append(side_ocr_data[i])
            SOP_parameter_list[7]['maybe_data_num'] += 1
        if L_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= L_max:
            SOP_parameter_list[10]['maybe_data'].append(side_ocr_data[i])
            SOP_parameter_list[10]['maybe_data_num'] += 1
        if c_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= c_max:
            SOP_parameter_list[12]['maybe_data'].append(side_ocr_data[i])
            SOP_parameter_list[12]['maybe_data_num'] += 1
        if side_ocr_data[i]['Absolutely'] == 'angle':
            if θ2_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= θ2_max:
                SOP_parameter_list[15]['maybe_data'].append(side_ocr_data[i])
                SOP_parameter_list[15]['maybe_data_num'] += 1
    for i in range(len(detailed_ocr_data)):
        if A_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= A_max:
            SOP_parameter_list[4]['maybe_data'].append(detailed_ocr_data[i])
            SOP_parameter_list[4]['maybe_data_num'] += 1
        if L_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= L_max:
            SOP_parameter_list[10]['maybe_data'].append(detailed_ocr_data[i])
            SOP_parameter_list[10]['maybe_data_num'] += 1
        if GAGE_PLANE_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][
            0] <= GAGE_PLANE_max:
            SOP_parameter_list[11]['maybe_data'].append(detailed_ocr_data[i])
            SOP_parameter_list[11]['maybe_data_num'] += 1
        if c_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= c_max:
            SOP_parameter_list[12]['maybe_data'].append(detailed_ocr_data[i])
            SOP_parameter_list[12]['maybe_data_num'] += 1
        if A1_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= A1_max:
            SOP_parameter_list[5]['maybe_data'].append(detailed_ocr_data[i])
            SOP_parameter_list[5]['maybe_data_num'] += 1
        if detailed_ocr_data[i]['Absolutely'] == 'angle':
            if θ_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][
                0] <= θ_max:
                SOP_parameter_list[13]['maybe_data'].append(detailed_ocr_data[i])
                SOP_parameter_list[13]['maybe_data_num'] += 1
            if θ1_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][
                0] <= θ1_max:
                SOP_parameter_list[14]['maybe_data'].append(detailed_ocr_data[i])
                SOP_parameter_list[14]['maybe_data_num'] += 1

            if θ2_min < detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][
                0] <= θ2_max:
                SOP_parameter_list[15]['maybe_data'].append(detailed_ocr_data[i])
                SOP_parameter_list[15]['maybe_data_num'] += 1
            if θ3_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][
                0] <= θ3_max:
                SOP_parameter_list[16]['maybe_data'].append(detailed_ocr_data[i])
                SOP_parameter_list[16]['maybe_data_num'] += 1

    for i in range(len(SOP_parameter_list)):
        print("***/", SOP_parameter_list[i]['parameter_name'], "/***")

        for j in range(len(SOP_parameter_list[i]['maybe_data'])):
            print(SOP_parameter_list[i]['maybe_data'][j]['max_medium_min'])

    for i in range(len(SOP_parameter_list)):
        print(SOP_parameter_list[i]['maybe_data_num'])

    return SOP_parameter_list


def get_span(top_ocr_data, bottom_ocr_data, find_horizontal):
    '''
    获取 SOP 的 span_x (总宽 D or E) 或 span_y (总长 D or E)
    逻辑：
    1. 从 TOP 视图中寻找：找到指定方向(横向或纵向)的尺寸线且物理长度(pixel distance)最大的数据1
    2. 从 BOTTOM 视图中寻找：找到指定方向(横向或纵向)的尺寸线且物理长度(pixel distance)最大的数据2
    3. 比较数据1和数据2的物理长度，取最大的对应数据作为 span_x 或 span_y
    4. 通过 ocr_data 中的 matched_pairs_location 进行位置关联

    参数:
    - top_ocr_data: 顶部视图OCR数据
    - bottom_ocr_data: 底部视图OCR数据
    - find_horizontal: True表示寻找最大横向尺寸，False表示寻找最大纵向尺寸
    '''
    if find_horizontal:
        print("---开始寻找 Span_x (最大横向尺寸)---")
    else:
        print("---开始寻找 Span_y (最大纵向尺寸)---")

    # 1. 在 Top 视图寻找
    data1, dist1 = find_max_directional_pair_data(top_ocr_data, "Top", find_horizontal)

    # 2. 在 Bottom 视图寻找
    data2, dist2 = find_max_directional_pair_data(bottom_ocr_data, "Bottom", find_horizontal)

    span_result = []

    # 3. 比较并择优
    if data1 is not None and data2 is not None:
        if dist1 >= dist2:
            span_result = [data1]
            print(f"对比结果: 选用 Top 视图数据 (长度 {dist1:.2f} >= {dist2:.2f})")
        else:
            span_result = [data2]
            print(f"对比结果: 选用 Bottom 视图数据 (长度 {dist2:.2f} > {dist1:.2f})")
    elif data1 is not None:
        span_result = [data1]
        print("对比结果: 仅 Top 视图有数据，直接选用")
    elif data2 is not None:
        span_result = [data2]
        print("对比结果: 仅 Bottom 视图有数据，直接选用")
    else:
        print("对比结果: 未找到 Span 数据")

    return span_result


def find_max_directional_pair_data(ocr_data_list, view_name, find_horizontal):
    max_distance = -1
    best_data = None

    for item in ocr_data_list:
        # 获取该 OCR 数据关联的 YOLOX 尺寸线坐标
        # matched_pairs_location 格式通常为 [x1, y1, x2, y2]
        pair_loc = item.get('matched_pairs_location')
        # 必须有关联的尺寸线，且已有提取出的数值
        if pair_loc is not None and len(pair_loc) >= 4 and len(item.get('max_medium_min', [])) == 3:
            x1, y1, x2, y2 = pair_loc[0], pair_loc[1], pair_loc[2], pair_loc[3]
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            # 根据参数判断方向
            if find_horizontal:
                # 判断是否为横向 (宽 > 高)
                if width > height:
                    # 计算物理距离 (对于横向，即宽度)
                    distance = width
                else:
                    continue  # 不符合横向条件，跳过
            else:
                # 寻找纵向 (高 > 宽)
                if height > width:
                    # 计算物理距离 (对于纵向，即高度)
                    distance = height
                else:
                    continue  # 不符合纵向条件，跳过

            if distance > max_distance:
                max_distance = distance
                best_data = item
    if best_data:
        if find_horizontal:
            print(
                f"在 {view_name} 视图中找到最大横向尺寸线，像素长度: {max_distance:.2f}, 值: {best_data['max_medium_min']}")
        else:
            print(
                f"在 {view_name} 视图中找到最大纵向尺寸线，像素长度: {max_distance:.2f}, 值: {best_data['max_medium_min']}")
    else:
        if find_horizontal:
            print(f"在 {view_name} 视图中未找到有效的横向尺寸线")
        else:
            print(f"在 {view_name} 视图中未找到有效的纵向尺寸线")

    return best_data, max_distance
