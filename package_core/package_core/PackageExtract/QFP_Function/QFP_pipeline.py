from ultralytics import YOLO
import os
import cv2
import numpy as np
from typing import Iterable, List, Dict, Any

from package_core.PackageExtract import common_pipeline
from package_core.PackageExtract.function_tool import find_list


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
            if data.get('view_name') == 'top':
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
    if pad is not None and len(pad) > 0:
        try:
            pad_box = pad[0]
            pad_width = abs(float(pad_box[2]) - float(pad_box[0]))  # x2 - x1
            pad_height = abs(float(pad_box[3]) - float(pad_box[1]))  # y2 - y1
            print(f"pad尺寸: 宽度={pad_width:.2f}, 高度={pad_height:.2f}")
        except Exception as e:
            print(f"错误: 计算pad尺寸时出错: {e}")
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
    similarity_threshold = 0.3  # 从10%放宽到20%的误差
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


def extract_bottom_D2_E2(L3, triple_factor, bottom_D, bottom_E):
    bottom_ocr_data = find_list(L3, "top_ocr_data")
    bottom_pad = find_list(L3, "top_pad")
    bottom_dbnet_data = find_list(L3, "bottom_dbnet_data")
    print(f'bottom_ocr_data:{bottom_ocr_data}')
    print(f'bottom_dbnet_data:{bottom_dbnet_data}')
    bottom_D2, bottom_E2 = extract_bottom_dimensions(bottom_D, bottom_E, bottom_pad, bottom_ocr_data, triple_factor)

    # if(bottom_D2[1] > bottom_E2[1]):
    #     bottom_D2, bottom_E2 = bottom_E2, bottom_D2

    return bottom_D2, bottom_E2


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

    # ================= 数据清洗开始 =================
    if top_ocr_data_list:
        cleaned_list = []  # 🟢 必须在这里初始化空列表

        # 定义常见引脚数列表 (QFP 常见引脚)
        std_pins = [44, 48, 52, 64, 80, 100, 120, 128, 144, 160, 176, 208, 240, 256,
                    # 你的日志里出现了 51, 76 这种奇怪的数，可能是引脚索引，也加进去防误判
             50, 51, 75, 76]

        for item in top_ocr_data_list:
            mmm = item.get('max_medium_min')
            if mmm is None or len(mmm) < 2:  # 这里建议改 < 2，防止 mmm 只有 1 个元素报错
                continue

            # 安全获取数值 (兼容 list 和 numpy array)
            try:
                val = float(mmm[1])

                # --- 1. 过滤疑似引脚数 ---
                # 如果数值是整数，且在常见引脚列表里，直接跳过！
                if val.is_integer() and int(val) in std_pins:
                    print(f"⚠️ 过滤疑似引脚数: {val}")
                    continue

                # --- 2. 过滤小杂鱼 ---
                # QFP 的 D/E 尺寸不可能小于 4.0mm
                if val < 4.0:
                    continue

                # 🟢 只有通过了上面两关，才加入清洗列表
                cleaned_list.append(item)

            except:
                # 如果转 float 失败，直接跳过
                continue

        # ⚠️ 核心操作：原地替换列表内容
        top_ocr_data_list[:] = cleaned_list

    print(f"收到 {len(top_ocr_data_list)} 个OCR数据 (已清洗)")

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

    # ======【新增】正方形封装判定（QFP 常见）======
    square_threshold = 0.08  # 允许 8% 误差

    if border_width > 0 and border_height > 0:
        ratio = abs(border_width - border_height) / max(border_width, border_height)

        if ratio < square_threshold:
            print(f"🟦 判定为正方形封装 (W≈H, ratio={ratio:.3f})，Top 视图 D = E")

            # 定义本地的引脚黑名单 (防止外部定义的没传进来)
            local_std_pins = [ 44, 48, 50, 51, 52, 64, 75, 76, 80, 100, 101, 120, 128, 144, 160, 176, 208, 240,
                              256]

            # 在所有 B 元素中，找一个“最像 Span”的尺寸
            def span_score(e):
                try:
                    val = float(e['max_medium_min'][1])
                except:
                    return 1e6

                # 🔒 第一把锁：物理尺寸上限 (尚方宝剑)
                # QFP 封装最大也就 28mm~32mm 左右，绝对不可能超过 35mm
                # 凡是大于 35 的，肯定是引脚数 (50, 75, 100)，直接枪毙！
                if val > 35.0:
                    return 1e6

                    # 🔒 第二把锁：过滤小尺寸
                if val < 5.0:
                    return 1e6

                # 🔒 第三把锁：整数引脚过滤 (针对 26.0, 51.0 这种漏网之鱼)
                if val.is_integer() and int(val) in local_std_pins:
                    return 1e6

                # 🔒 第四把锁：排除 Explicitly 标记为 Pin 的
                if e.get('Absolutely') in ('mb_pin_diameter', 'pin_diameter'):
                    return 1e6

                # 🏆 加分项：有引线 (Arrow Pairs) 的优先
                # 如果这个数据有检测到线长，说明它是尺寸的概率极大
                score = -val  # 基础分：越大越好

                pairs = e.get('arrow_pairs')
                if pairs is not None and len(pairs) > 0:
                    score -= 1000  # 只要有引线，优先度提升 1000 倍！

                return score

            # 排序
            all_b_elements.sort(key=span_score)

            # 检查排序后的第一个元素是否有效 (防止全是 1e6 的垃圾数据)
            best_candidate = all_b_elements[0]
            best_val = float(best_candidate['max_medium_min'][1])

            # 如果最好的结果都被枪毙了 (score >= 1e6)，说明正方形逻辑失败，退回普通逻辑
            if span_score(best_candidate) >= 1e5:
                print("⚠️ 正方形判定虽然通过，但没找到合法的尺寸数据 (都 > 35mm)，跳过正方形强制逻辑")
            else:
                span = best_candidate['max_medium_min'].copy()
                print(f"✅ 选定 Top Span = {span} (Val={best_val})，用于 D 和 E")
                return span, span
    # ======【新增结束】======

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


def extract_top_D_E(L3, triple_factor):
    # 1. 提取所有基础数据
    top_ocr_data = find_list(L3, "top_ocr_data")
    bottom_ocr_data = find_list(L3, "bottom_ocr_data")
    top_border = find_list(L3, "top_border")
    bottom_border = find_list(L3, "bottom_border")

    print("=== 开始提取 D/E 尺寸 (优先 Bottom 视图) ===")

    # 2. 【修改点】首先尝试从 Bottom 视图提取
    # 注意：传入 bottom 数据，且 key=1
    d_vals, e_vals = extract_top_dimensions(bottom_border, bottom_ocr_data, triple_factor, 1)

    # 3. 检查数据完整性，如果数据为0，尝试从 Top 视图补充
    # 判断是否有效：这里假设全0表示无效
    d_invalid = np.all(np.array(d_vals) == 0)
    e_invalid = np.all(np.array(e_vals) == 0)

    if d_invalid or e_invalid:
        print("Bottom 视图数据不完整，尝试从 Top 视图补充...")
        # 提取 Top 数据作为备用
        top_d_temp, top_e_temp = extract_top_dimensions(top_border, top_ocr_data, triple_factor, 0)

        # 如果 D 在 Bottom 没找到，用 Top 的
        if d_invalid:
            d_vals = top_d_temp
            print(f"使用 Top 视图补充 D: {d_vals}")

        # 如果 E 在 Bottom 没找到，用 Top 的
        if e_invalid:
            e_vals = top_e_temp
            print(f"使用 Top 视图补充 E: {e_vals}")

    # 4. 可选：最后的交换逻辑（如果你需要保证 D 是长边，E 是短边，或者反之）
    # if d_vals[1] > e_vals[1]: # 比较标准值
    #     d_vals, e_vals = e_vals, d_vals

    return d_vals, e_vals

##D1E1
def extract_D1_E1_from_ocr(L3, top_D, top_E, bottom_D2, bottom_E2):
    """
    从所有 OCR 数据中提取 D1 / E1 (Body Size)
    已修改：保留原始 Max/Typ/Min 数组结构，防止精度丢失
    """

    # ---------- 1. 边界检查 (保持不变) ----------
    def get_max(arr):
        return np.max(arr) if hasattr(arr, '__iter__') else arr

    def get_min(arr):
        return np.min(arr) if hasattr(arr, '__iter__') else arr

    def is_empty(x):
        return x is None or (hasattr(x, "__len__") and len(x) == 0)

    if is_empty(top_D) or is_empty(top_E) or is_empty(bottom_D2) or is_empty(bottom_E2):
        return [], []

    # 确定 D1/E1 必须存在的“夹缝”区间
    D_limit_max = get_max(top_D)  # 外廓 (Span)
    D2_limit_min = get_min(bottom_D2)  # 内引脚 (Pad/Inner)

    E_limit_max = get_max(top_E)
    E2_limit_min = get_min(bottom_E2)

    print(f"D1 搜索范围: {D2_limit_min} < x < {D_limit_max}")
    print(f"E1 搜索范围: {E2_limit_min} < x < {E_limit_max}")

    candidates = []

    # ---------- 2. 扩大搜索池 & 保留原始数据 (关键修改) ----------
    target_keys = [
        'top_ocr_data', 'bottom_ocr_data', 'side_ocr_data',
        'top_multi_value_2', 'bottom_multi_value_2'
    ]

    for key in target_keys:
        data_list = common_pipeline.find_list(L3, key)  # 确保 common_pipeline 可用，或者去掉前缀
        # 如果找不到 common_pipeline，请把上面这行改成: data_list = find_list(L3, key)
        if not data_list: continue

        for item in data_list:
            val = 0
            raw_mmm = []  # 用于存储 [max, mid, min]

            if isinstance(item, dict):
                # 优先取标准化的 max_medium_min
                if 'max_medium_min' in item and len(item['max_medium_min']) > 0:
                    raw_mmm = item['max_medium_min']
                    val = raw_mmm[0]  # 取最大值来做区间筛选
                elif 'value' in item:
                    val = item['value']
            else:
                val = item

            try:
                val = float(val)
                if val > 0:
                    # ⭐ 修改点：存入字典，保留原始 array 数据
                    candidates.append({'num': val, 'array': raw_mmm})
            except:
                continue

    # 排序 (按数值大小)
    candidates.sort(key=lambda x: x['num'])

    # ---------- 3. 区间筛选 ----------
    # 这里的 v 是字典 {'num':..., 'array':...}
    D1_candidates = [v for v in candidates if D2_limit_min < v['num'] < D_limit_max]
    E1_candidates = [v for v in candidates if E2_limit_min < v['num'] < E_limit_max]

    print(f"符合条件的 D1: {[c['num'] for c in D1_candidates]}")
    print(f"符合条件的 E1: {[c['num'] for c in E1_candidates]}")

    # ---------- 4. 输出格式化 (你的逻辑加在这里) ----------

    def pack_result(cand_list):
        if not cand_list: return []

        # 策略：取最大的一个 (Body Size 通常很大，接近 Span)
        best_candidate = cand_list[-1]

        # 👇👇👇 这里就是你要加的逻辑 👇👇👇
        # 优先使用原始的 3元素数组 (例如 [14.2, 14.0, 13.8])
        arr = best_candidate['array']
        if arr is not None and hasattr(arr, '__len__') and len(arr) == 3:

            print(f"   🌟 成功提取到范围数据: {best_candidate['array']}")
            return best_candidate['array']
        else:
            # 只有万一数组不全时，才用单数填充
            val = best_candidate['num']
            return [val, val, val]
        # 👆👆👆 逻辑结束 👆👆👆

    return pack_result(D1_candidates), pack_result(E1_candidates)


def extract_lead_length_L(L3, top_D, D1_data):
    """
    提取 QFP 的引脚长度 L
    策略：
    1. 在 Detailed/Side 视图中找数值。
    2. 利用 (D - D1)/2 作为几何上限约束，过滤不合理的 L。
    """
    print("\n>>> 开始提取引脚长度 L ...")

    # --- 1. 计算几何上限 (Total Protrusion) ---
    geo_limit_max = 999.0  # 默认无穷大

    # 尝试从 D 和 D1 计算单边伸出长度
    try:
        # 获取标准值 (中间值)
        d_val = top_D[1] if (top_D and len(top_D) > 1) else 0
        d1_val = D1_data[1] if (D1_data and len(D1_data) > 1) else 0

        if d_val > 0 and d1_val > 0 and d_val > d1_val:
            # 物理间隙 = (总跨度 - 封装体) / 2
            # 加上 0.2mm 的容错冗余，防止 OCR 误差导致误杀
            geo_limit_max = ((d_val - d1_val) / 2.0) + 0.2
            print(f"📐 [几何约束] 根据 D={d_val}, D1={d1_val} 计算出 L 的理论上限: < {geo_limit_max:.2f}")
    except Exception as e:
        print(f"⚠️ 无法计算几何约束: {e}")

    candidates = []

    # L 通常出现在 Detailed 或 Side 视图，偶尔在 Bottom
    keys = ['detailed_ocr_data', 'side_ocr_data', 'bottom_ocr_data']

    from package_core.PackageExtract.function_tool import find_list  # 确保引入

    for k in keys:
        data = find_list(L3, k)
        if not data: continue

        for item in data:
            mmm = item.get('max_medium_min')
            if mmm is None or len(mmm) == 0: continue

            # 获取中间值
            val = mmm[1]

            # --- 2. 基础筛选 ---
            # QFP L 通常在 0.30mm ~ 2.0mm 之间 (放宽上限以适应大尺寸)
            if 0.30 <= val <= 2.0:

                # --- 3. 几何约束核心逻辑 ---
                # 如果 OCR 数值比物理伸出空间还大，那它绝对不是 L (可能是 A 或者 e)
                if val > geo_limit_max:
                    # print(f"   ❌ 排除 {val}: 大于理论上限 {geo_limit_max:.2f}")
                    continue

                confidence = 1.0

                # A. 公差奖励：如果有公差 (max != min)，置信度加倍
                if mmm[0] != mmm[2]:
                    confidence += 2.0

                # B. 典型值奖励：0.45, 0.5, 0.6, 0.75, 1.0, 1.2
                if any(abs(val - target) < 0.05 for target in [0.45, 0.50, 0.60, 0.75, 1.0, 1.20]):
                    confidence += 1.0

                # C. 几何贴合奖励：L 通常占伸出长度的 50% ~ 90%
                # 例如：伸出 1.0mm，L 可能是 0.6mm。如果 val 是 0.6，很合理。
                # 如果 val 是 0.1 (太短) 或 0.99 (太满)，可能性较低。
                if geo_limit_max < 100:  # 只有当几何约束有效时
                    ratio = val / (geo_limit_max - 0.2)  # 还原回纯粹的 gap
                    if 0.4 <= ratio <= 0.95:
                        confidence += 1.5

                candidates.append({'val': mmm, 'score': confidence})

    # 排序取置信度最高的
    if candidates:
        candidates.sort(key=lambda x: x['score'], reverse=True)
        best_match = candidates[0]
        print(f"✅ 提取到最佳 L: {best_match['val']} (得分 {best_match['score']})")
        return best_match['val']

    print("⚠️ 未提取到 L，返回默认空值")
    return []

def get_integrated_parameter_list(
        L3: List[Any],
        calc_results: Dict[str, List[float]]
) -> List[Dict]:
    """
    结合精确计算结果和OCR数值筛选，生成最终参数列表。

    :param L3: 包含 OCR 数据的大列表
    :param calc_results: 字典，包含精确计算出的 {'D':[], 'E':[], 'D1':[], 'E1':[], 'D2':[], 'E2':[]}
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
    # 顺序: 0:D, 1:E, 2:D1, 3:E1, 4:A, 5:A1, 6:e, 7:b, 8:D2, 9:E2,
    #       10:L, 11:GAGE, 12:c, 13:θ, 14:θ1, 15:θ2, 16:θ3, 17:Φ
    params = [
        create_param('D'), create_param('E'), create_param('D1'), create_param('E1'),
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

    priority_keys = ['D', 'E', 'D1', 'E1', 'D2', 'E2','e','b','L']

    for key in priority_keys:
        val_list = calc_results.get(key, [])
        # 1. 安全判空：不能直接写 if val_list
        if val_list is None or len(val_list) == 0:
            continue

        # 2. 安全数值检查：防止报错 "truth value of an array is ambiguous"
        # 也就是把 "any(v != 0 ...)" 这一步做得更稳健
        has_valid_value = False

        if isinstance(val_list, np.ndarray):
            # 如果是 numpy 数组，使用 .any() 方法判断是否包含非0值
            if np.any(val_list != 0):
                has_valid_value = True
                # 可选：顺手转成 list，方便后续统一处理
                val_list = val_list.tolist()
        else:
            # 如果是普通 list
            if any(v != 0 for v in val_list):
                has_valid_value = True

        # 3. 只有数据有效才填入
        if has_valid_value:
            idx = p_map[key]
            mock_data = {
                'max_medium_min': val_list,
                'Absolutely': f'Calculated_{key}',
                'confidence': 1.0
            }
            params[idx]['maybe_data'].append(mock_data)
            params[idx]['maybe_data_num'] = 1
            params[idx]['OK'] = 1

    # 4. 【第二步】对其余参数使用区间筛选 (A, e, b, L, θ, Φ...)
    # 定义阈值 (保留你原代码的阈值)
    ranges = {
        'A': (1.0, 4.5), 'A1': (0, 0.4), 'e': (0.30, 1.3), 'b': (0.13, 0.83),
        'L': (0.45, 0.75), 'GAGE_PLANE': (0.25, 0.25), 'c': (0.09, 0.20),
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
    # 🛠️ 修改点：加一个判断条件
    # 如果 detailed_ocr_data 里有数据（说明识别到了细节图），
    # 就【绝对不】在 Side View 里找 A1，防止把引脚厚度 c (0.127) 当成 A1。
    if not detailed_ocr_data:
        check_and_add(side_ocr_data, 'A1')
        print("   Detailed视图为空，降级在 Side 视图中搜索 A1")
    else:
        print(f"   👀 检测到 Detailed 视图 ({len(detailed_ocr_data)}个数据)，跳过 Side A1 搜索 (防误读)")
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

    # 4.2 Detailed View 筛选 (A,A1, L, GAGE, c, angles)
    check_and_add(detailed_ocr_data, 'A')
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


def parse_pin_txt(txt_path):
    """
    解析 QFP_adjacent_pins.txt 文件
    返回: horizontal_pins (list of lists), vertical_pins (list of lists)
    格式: [[x1, y1, x2, y2], ...]
    """
    if not os.path.exists(txt_path):
        print(f"⚠️ Warning: Pin file not found at {txt_path}")
        return [], []

    h_pins = []
    v_pins = []

    # 假设 txt 格式大致是: 每行一个坐标，或者按某种分隔符
    # 根据你之前的代码 common_pipeline.extract_pin_boxes_from_txt，
    # 我假设它返回的是 pin_box, pin_boxh, pin_boxv
    # 这里为了通用性，我重新实现一个简单的解析逻辑，或者你可以直接复用你的 extract_pin_boxes_from_txt

    # 临时模拟你的 extract_pin_boxes_from_txt 的返回结果
    # 在实际整合时，请直接传入 run_f4_pipeline 里读取到的 pin_boxh, pin_boxv 即可
    # 这里仅作逻辑展示
    return [], []


def calculate_geometric_ratios(pin_boxh, pin_boxv):
    """
    根据引脚框计算几何特征：
    1. 平均引脚像素宽度 (Pixel Width) -> 对应 b
    2. 平均引脚像素间距 (Pixel Pitch) -> 对应 e
    3. 间距宽度比 (Pitch/Width Ratio)
    """
    all_widths = []
    all_pitches = []

    # 定义一个内部函数来处理一组引脚
    def process_pins(pins, is_horizontal):
        if not pins or len(pins) < 2: return

        # 1. 排序
        # Horizontal pins: 排成一排，x 递增
        # Vertical pins: 排成一列，y 递增
        sort_idx = 0 if is_horizontal else 1
        # 过滤掉非法的框 (x2<=x1 或 y2<=y1)
        valid_pins = [p for p in pins if p[2] > p[0] and p[3] > p[1]]
        if not valid_pins: return

        sorted_pins = sorted(valid_pins, key=lambda b: b[sort_idx])

        # 2. 计算宽度 (b)
        # Horizontal pin 的宽度是 y方向的长度? 不，QFP通常指引脚本身的宽度
        # 对于横向引脚(左右两侧)，宽度是 y2-y1
        # 对于纵向引脚(上下两侧)，宽度是 x2-x1
        # 修正：根据 QFP 定义，b 是引脚的短边宽度。
        # 如果 pin_boxh 代表上下两排的引脚（竖着长的），那宽度是 x2-x1
        # 如果 pin_boxh 代表左右两排的引脚（横着长的），那宽度是 y2-y1
        # 这里为了稳健，我们取短边作为宽度

        for p in sorted_pins:
            w = abs(p[2] - p[0])
            h = abs(p[3] - p[1])
            pin_w = min(w, h)  # 取短边作为 b
            all_widths.append(pin_w)

        # 3. 计算间距 (e) - 中心距
        for i in range(len(sorted_pins) - 1):
            curr_box = sorted_pins[i]
            next_box = sorted_pins[i + 1]

            # 中心点
            curr_center = (curr_box[sort_idx] + curr_box[sort_idx + 2]) / 2
            next_center = (next_box[sort_idx] + next_box[sort_idx + 2]) / 2

            pitch = abs(next_center - curr_center)
            all_pitches.append(pitch)

    # 处理水平组和竖直组
    # 注意：你需要确认 pin_boxh 里的框是“排成水平行”还是“形状是水平长条”
    # 通常 pin_boxh 指的是“排列方向是水平的” (即上下两边的引脚)
    process_pins(pin_boxh, is_horizontal=True)
    process_pins(pin_boxv, is_horizontal=False)

    if not all_widths or not all_pitches:
        return None, None, None

    # 去除极值求平均
    avg_pixel_b = np.median(all_widths)
    avg_pixel_e = np.median(all_pitches)

    # 核心几何特征：e/b 的比率
    # QFP 通常 e = 0.5, b = 0.2 -> ratio = 2.5
    # 或者 e = 0.65, b = 0.3 -> ratio = 2.16
    # 或者 e = 1.27, b = 0.4 -> ratio = 3.1
    # 无论如何，Ratio 应该 > 1.5
    ratio = avg_pixel_e / avg_pixel_b if avg_pixel_b > 0 else 0

    print(
        f"📐 [几何分析] 像素宽度(b): {avg_pixel_b:.1f}px, 像素间距(e): {avg_pixel_e:.1f}px, 几何比率(e/b): {ratio:.2f}")

    return avg_pixel_b, avg_pixel_e, ratio



def verify_and_extract_e_b(L3, pin_boxh, pin_boxv):
    """
    利用几何比率校验并提取最佳的 e 和 b (V4 智能推导版)
    策略：不盲目注入，而是根据 b 和 几何比率 反推 e，如果推导出的 e 符合标准，才纳入考虑。
    """
    # 1. 获取几何特征
    try:
        px_b, px_e, geom_ratio = calculate_geometric_ratios(pin_boxh, pin_boxv)
    except:
        return [], []

    if geom_ratio <= 0: return [], []

    # 2. 获取 OCR 候选数据
    candidates = []
    keys = ['top_ocr_data', 'bottom_ocr_data', 'side_ocr_data', 'detailed_ocr_data']
    from package_core.PackageExtract.function_tool import find_list

    for k in keys:
        data = find_list(L3, k)
        if data:
            for item in data:
                mmm = item.get('max_medium_min')
                if mmm is not None and len(mmm) >= 2:
                    try:
                        val = float(mmm[1])
                    except:
                        continue
                    # 宽松收集所有可能的尺寸
                    if 0.05 <= val <= 3.0:
                        is_high_conf = (len(mmm) == 3 and mmm[0] != mmm[1])
                        if '±' in item.get('ocr_strings', ''): is_high_conf = True

                        candidates.append({
                            'val': val, 'full_data': item, 'is_high_conf': is_high_conf
                        })

    ocr_vals = sorted(list(set([c['val'] for c in candidates])))
    print(f"🔍 [OCR候选] 原始数值: {ocr_vals}")

    # ==================== 🧠 关键修改: 智能反推 Pitch ====================
    # QFP JEDEC 标准 Pitch
    std_pitches = [0.4, 0.5, 0.65, 0.8, 1.0, 1.27]

    inferred_candidates = []  # 存放推导出来的 e

    # 遍历每一个 OCR 读到的数值，假设它是 b，推导它对应的 e
    for val_b in ocr_vals:
        # 如果这个值太小，可能是 b；如果太大(>1.0)，不太可能是 b
        if val_b > 0.8: continue

        # 根据几何比率反推理论 e
        theoretical_e = val_b * geom_ratio

        # 看看这个理论 e 是否命中某个标准 Pitch (允许 10% 误差)
        for std_e in std_pitches:
            if abs(theoretical_e - std_e) < 0.1:  # 误差 0.1mm 以内
                # print(f"   💡 推导: 若 b={val_b}, 则 e≈{theoretical_e:.3f} -> 命中标准值 {std_e}")

                # 如果这个标准值不在 OCR 列表里，把它作为“推导值”加入
                if std_e not in ocr_vals:
                    # 创建一个虚拟数据包
                    mock_data = {'max_medium_min': [std_e, std_e, std_e]}
                    # 加入候选池，标记来源为 'Inferred'
                    candidates.append({
                        'val': std_e, 'full_data': mock_data,
                        'is_high_conf': False, 'source': 'Inferred'
                    })
                    inferred_candidates.append(std_e)

    # 更新数值列表 (去重)
    ocr_vals_extended = sorted(list(set(ocr_vals + inferred_candidates)))
    if len(inferred_candidates) > 0:
        print(f"✨ [智能补全] 根据几何关系，补全了疑似漏读的 Pitch: {inferred_candidates}")
    # ====================================================================

    print(f"🚀 启用几何比率校验 (几何Ratio={geom_ratio:.2f})...")

    best_score = float('inf')
    best_e = None
    best_b = None

    for val_e in ocr_vals_extended:
        for val_b in ocr_vals_extended:
            if val_e <= val_b: continue
            if val_e < val_b * 1.2: continue  # Pitch 必须明显大于 Width

            ocr_ratio = val_e / val_b
            diff = ocr_ratio - geom_ratio
            score = abs(diff)

            # --- 评分策略 ---
            # 1. 几何误差惩罚
            if 0 < diff < 2.0:
                score *= 0.5
            elif diff >= 2.0:
                score *= 10.0  # 误差太大
            else:
                score *= 5.0  # 反向误差(b太粗)

            # 2. 物理常识约束
            if val_b > 0.55: score += 5.0  # b 太粗，重罚
            if val_b < 0.14: score += 8.0  # b 太细，重罚 (防止选到0.1)

            # 3. 奖励标准 Pitch
            if any(abs(val_e - p) < 0.02 for p in std_pitches):
                score -= 1.0

            # 4. 奖励高置信度 (OCR原生的 b 优于 推导的 e)
            b_cand = next((c for c in candidates if c['val'] == val_b), None)
            if b_cand and b_cand.get('is_high_conf', False): score -= 1.5

            if score < best_score:
                best_score = score
                best_e = val_e
                best_b = val_b

    if best_e and best_b:
        print(f"✅ [校验成功] 最佳匹配: e={best_e}, b={best_b} (OCR Ratio {best_e / best_b:.2f})")

        e_cand = next((c for c in candidates if c['val'] == best_e), None)
        b_cand = next((c for c in candidates if c['val'] == best_b), None)

        e_data = e_cand['full_data']['max_medium_min'] if e_cand else [best_e, best_e, best_e]
        b_data = b_cand['full_data']['max_medium_min'] if b_cand else [best_b, best_b, best_b]

        return e_data, b_data

    return [], []

def extract_e_b_combined(L3, pin_boxh, pin_boxv):
    """
    策略：
    1. 【优先】物理匹配：计算 Pin 的像素宽度(px_b)和间距(px_e)，去匹配尺寸线(Arrow)的长度。
       如果线的长度和 px_b 一样长，那它对应的数字就是 b。
    2. 【兜底】几何推导：如果没找到线，使用之前的几何比率法反推。
    """
    print("\n>>> 开始提取 e 和 b (优先匹配尺寸线)...")

    # --- 1. 获取物理像素基准 (尺子) ---
    try:
        # 调用之前给你的 calculate_geometric_ratios 获取像素值
        # px_b: 引脚平均像素宽度
        # px_e: 引脚平均像素间距
        px_b, px_e, geom_ratio = calculate_geometric_ratios(pin_boxh, pin_boxv)
    except:
        px_b, px_e, geom_ratio = 0, 0, 0

    if px_b <= 0 or px_e <= 0:
        print("⚠️ 无法计算 Pin 像素尺寸，直接跳至几何比率校验...")
        return verify_and_extract_e_b(L3, pin_boxh, pin_boxv)  # 回退到上一版方案

    print(f"📏 [物理基准] 像素宽度(b)≈{px_b:.1f}px, 像素间距(e)≈{px_e:.1f}px")

    # --- 2. 搜集所有带线的数据 ---
    keys = ['top_ocr_data', 'bottom_ocr_data', 'side_ocr_data', 'detailed_ocr_data']

    # 候选池
    candidates_b = []
    candidates_e = []

    from package_core.PackageExtract.function_tool import find_list

    for k in keys:
        data = find_list(L3, k)
        if not data: continue

        for item in data:
            # 必须有数值
            mmm = item.get('max_medium_min')
            if mmm is None or len(mmm) < 2: continue
            val = mmm[1]

            # --- 核心逻辑：检查尺寸线长度 ---
            arrow_pairs = item.get('arrow_pairs')

            matched_by_line = False

            if arrow_pairs is not None and len(arrow_pairs) > 0:
                try:
                    # 获取这根线的像素长度 (最后一个值通常是距离)
                    line_len = float(arrow_pairs[-1])

                    # 容错率 (允许 25% 的误差，因为线可能画得不准)
                    tolerance = 0.25

                    # A. 是 b 吗？(线长 ≈ px_b)
                    diff_b = abs(line_len - px_b)
                    if diff_b < px_b * tolerance:
                        print(f"   🎯 线长匹配成功(b): 数值={val}, 线长={line_len:.1f}px (基准{px_b:.1f})")
                        candidates_b.append({'val': mmm, 'score': 100 - diff_b})  # 差值越小分越高
                        matched_by_line = True

                    # B. 是 e 吗？(线长 ≈ px_e)
                    diff_e = abs(line_len - px_e)
                    if diff_e < px_e * tolerance:
                        print(f"   🎯 线长匹配成功(e): 数值={val}, 线长={line_len:.1f}px (基准{px_e:.1f})")
                        candidates_e.append({'val': mmm, 'score': 100 - diff_e})
                        matched_by_line = True

                except:
                    pass

            # 如果没有线，或者线没匹配上，暂不处理，留给后面的兜底逻辑

    # --- 3. 决策阶段 ---

    final_e = []
    final_b = []

    # 选取 b (优先取线长匹配得分最高的)
    if candidates_b:
        candidates_b.sort(key=lambda x: x['score'], reverse=True)
        final_b = candidates_b[0]['val']
        print(f"✅ [锁定 b] 通过尺寸线锁定 b = {final_b}")

    # 选取 e (优先取线长匹配得分最高的)
    if candidates_e:
        candidates_e.sort(key=lambda x: x['score'], reverse=True)
        final_e = candidates_e[0]['val']
        print(f"✅ [锁定 e] 通过尺寸线锁定 e = {final_e}")

    # --- 4. 兜底逻辑：如果有没找到的，用之前的逻辑补全 ---
    if not final_e or not final_b:
        print("⚠️ 部分参数未通过线长锁定，启用几何比率推导补全...")
        # 调用上一版写的 verify_and_extract_e_b (V4版本)
        fallback_e, fallback_b = verify_and_extract_e_b(L3, pin_boxh, pin_boxv)

        if not final_e: final_e = fallback_e
        if not final_b: final_b = fallback_b

    return final_e, final_b
