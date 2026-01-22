# 外部文件：
import json
from typing import Iterable

from package_core.PackageExtract.function_tool import *
from package_core.PackageExtract.get_pairs_data_present5_test import *
from package_core.PackageExtract import common_pipeline
from package_core.PackageExtract.SOP_Function import SOP_pipeline
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

# 导入统一路径管理
try:
    from package_core.PackageExtract.yolox_onnx_py.model_paths import result_path
except ModuleNotFoundError:
    from pathlib import Path


    def result_path(*parts):
        return str(Path(__file__).resolve().parents[3] / 'Result' / Path(*parts))

# 全局路径 - 使用统一的路径管理函数
DATA = result_path('Package_extract', 'data')
DATA_BOTTOM_CROP = result_path('Package_extract', 'data_bottom_crop')
DATA_COPY = result_path('Package_extract', 'data_copy')
ONNX_OUTPUT = result_path('Package_extract', 'onnx_output')
OPENCV_OUTPUT = result_path('Package_extract', 'opencv_output')
OPENCV_OUTPUT_LINE = result_path('Package_extract', 'opencv_output_yinXian')
YOLO_DATA = result_path('Package_extract', 'yolox_data')


def detect_sop_pin_orientation(image_folder, name):
    """
    使用投影法判断SOP引脚方向
    返回:
    - 'Top_Bottom': 引脚在上下两侧 (需要计算垂直方向的跨度 Span)
    - 'Left_Right': 引脚在左右两侧 (需要计算水平方向的跨度 Span)
    - None: 检测失败
    """
    img_path = os.path.join(image_folder, name)

    if not os.path.exists(img_path):
        print(f"[Error] detect_sop_pin_orientation: 找不到文件 {img_path}")
        return None

    # 1. 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"[Error] detect_sop_pin_orientation: 无法读取图片 {img_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 二值化处理 (Otsu自动阈值)
    # SOP引脚通常是亮的，背景是暗的。如果图片是黑底白引脚，直接用；
    # 如果是白底黑引脚，建议反转: cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. 计算投影
    # 垂直投影 (按列求和) -> 反映 X 轴方向上的变化频率
    v_projection = np.sum(binary, axis=0)
    # 水平投影 (按行求和) -> 反映 Y 轴方向上的变化频率
    h_projection = np.sum(binary, axis=1)

    # 4. 计算投影数据的“波动次数”
    def count_changes(projection_data):
        if len(projection_data) == 0: return 0
        mean_val = np.mean(projection_data)
        # 二值化波形：高于均值为1，低于为0
        binary_signal = (projection_data > mean_val).astype(int)
        # 计算跳变次数 (0->1 或 1->0)
        changes = np.sum(np.abs(np.diff(binary_signal)))
        return changes

    v_changes = count_changes(v_projection)
    h_changes = count_changes(h_projection)

    if v_changes > h_changes:
        print(f"判定结果: 引脚在 [上下方] (Top/Bottom) | V:{v_changes} > H:{h_changes}")
        return "垂直方向"
    else:
        print(f"判定结果: 引脚在 [左右方] (Left/Right) | H:{h_changes} > V:{v_changes}")
        return "水平方向"


import numpy as np


def generate_excel_parameter_list(final_parameter_list):
    """
    从 final_parameter_list 提取数据，并将【所有参数】的第0位和第2位互换
    """

    # --- 1. 定义数据提取辅助函数 (增加了自动互换逻辑) ---
    def get_data(param_name):
        """
        查找参数，如果存在数据，默认执行 [0] 与 [2] 互换
        返回: [Min, Medium, Max] (原本是 Max, Medium, Min)
        """
        # 在 final_parameter_list 中查找 name 匹配的项
        item = next((x for x in final_parameter_list if x['parameter_name'] == param_name), None)

        default_empty = ['', '', '']

        if item and item.get('maybe_data_num', 0) > 0:
            try:
                # 获取第一组数据的 max_medium_min
                raw_val = item['maybe_data'][0]['max_medium_min']

                # 处理 numpy 数组
                if isinstance(raw_val, np.ndarray):
                    raw_val = raw_val.tolist()

                # 确保数据有效且长度足够
                if isinstance(raw_val, list) and len(raw_val) >= 3:
                    # 截取前3位并保留3位小数
                    val = [round(float(x), 3) for x in raw_val[:3]]

                    # ===========【核心修改】===========
                    # 对每一个获取到的参数，都执行互换
                    # 原序: [Max, Mid, Min] -> 互换后: [Min, Mid, Max]
                    val[0], val[2] = val[2], val[0]
                    # =================================

                    return val
            except (ValueError, TypeError, IndexError):
                return default_empty

        return default_empty

    # --- 2. 提取数据 (此时所有数据都已经过互换) ---
    bottom_pitch = get_data('e')  # Pitch
    side_A = get_data('A')  # Package Height
    side_A1 = get_data('A1')  # Standoff
    span = get_data('E')  # Span X
    top_E = get_data('E1')  # Body X
    top_D = get_data('D')  # Body Y
    L = get_data('L')  # Lead Length
    bottom_b = get_data('b')  # Lead width
    side_A3 = get_data('c')  # Lead Thickness
    bottom_E2 = get_data('E2')  # Thermal X
    bottom_D2 = get_data('D2')  # Thermal Y

    # --- 3. 组装最终列表 (不需要再单独写 bottom_pitch 的互换逻辑了) ---
    parameter_list = [
        [''] + bottom_pitch,
        ['', '', '', ''],
        [''] + side_A,
        [''] + side_A1,
        [''] + span,
        [''] + top_E,
        [''] + top_D,
        ['', '', '', ''],
        ['', '', '', ''],
        [''] + L,
        [''] + bottom_b,
        [''] + side_A3,
        ['', '', '', ''],
        [''] + bottom_E2,
        [''] + bottom_D2
    ]

    return parameter_list


def run_f4_pipeline_SOP(
        image_root: str,
        package_class: str,
        key: int = 0,
        test_mode: int = 0,
):
    """串联执行 F4 阶段的主要函数，返回参数列表与中间结果。

        :param image_root: 存放 ``top/bottom/side/detailed`` 视图图片的目录。
        :param package_class: 封装类型，例如 ``"QFP"``、``"BGA"``。
        :param key: 与历史实现一致的流程参数，用于控制 OCR 清洗策略。
        :param test_mode: 传递给 ``find_pairs_length`` 的调试开关。
        :param view_names: 自定义视图顺序；默认为 ``common_pipeline.DEFAULT_VIEWS``。
        :returns: ``dict``，包含 ``L3`` 数据、参数候选列表以及 ``nx``/``ny``。
        """

    # 从 image_root 获取视图名称（支持目录和图片文件）
    if os.path.exists(image_root):
        views_items = []
        for item in os.listdir(image_root):
            item_path = os.path.join(image_root, item)
            if os.path.isfile(item_path) and item.lower().endswith(('.jpg', '.jpeg', '.png')):
                # 去掉文件扩展名作为视图名称
                view_name = os.path.splitext(item)[0]
                views_items.append(view_name)
        views: Iterable[str] = views_items
    else:
        views: Iterable[str] = common_pipeline.DEFAULT_VIEWS
    print("views:", views)
    ## 初始化合并L1L2构建L3
    print("开始测试初始L3集合")
    print(f'图片路径{image_root}')
    L3 = common_pipeline.get_data_location_by_yolo_dbnet(image_root, package_class, view_names=views)

    ## F4.1-F4.4
    print("开始测试F4.1")
    L3 = other_match_dbnet.other_match_boxes_by_overlap(L3)
    ## F4.2
    print("开始测试F4.2")
    L3 = pin_match_dbnet.PINnum_find_matching_boxes(L3)
    print("开始测试F4.3")
    L3 = angle_match_dbnet.angle_find_matching_boxes(L3)
    print("开始测试F4.4")
    L3 = num_match_dbnet.num_match_size_boxes(L3)
    ## F4.45（添加方向字段）
    print("开始测试F4.45")
    L3 = num_direction.add_direction_field_to_yolox_nums(L3)
    ## F4.6
    print("开始测试F4.6")
    L3 = common_pipeline.enrich_pairs_with_lines(L3, image_root, test_mode)
    ## F4.7
    print("开始测试F4.7")
    triple_factor = match_triple_factor.match_arrow_pairs_with_yolox(L3, image_root)
    print("*****triple_factor*****:", triple_factor)
    ## （整理尺寸线与文本，生成初始配对候选）
    L3 = common_pipeline.preprocess_pairs_and_text(L3, key)
    ## F4.5
    # ==================================
    # 5. [Step 1] 先进行 Triple Factor 匹配
    # 此时主要利用 YOLOX 框的位置和箭头进行关联，还没做 OCR
    # print(">> 开始 F4.7 (Match Triple Factor - Location Based)")
    # triple_factor_results = match_triple_factor.match_arrow_pairs_with_yolox(L3, image_root)
    # # 6. [Step 2] 执行定向 OCR (Targeted OCR)
    # # 利用 Triple Factor 里的 YOLOX 大框去跑识别，获取完整的 "0.80x 12=9.60"
    # triple_factor_results = targeted_ocr.run_ocr_on_yolox_locations(triple_factor_results, image_root)
    # # 7. [Step 3] 数据回填 (Overwrite L3)
    # # 用识别好的完整数据，替换掉 L3 里原本可能存在的碎片化 DBNet 数据
    # # 这样后续的 normalize_ocr_candidates 就会处理我们清洗好的数据
    # L3 = targeted_ocr.update_L3_with_yolox_ocr(L3, triple_factor_results)

    L3 = common_pipeline.run_svtr_ocr(L3)
    L3 = common_pipeline.normalize_ocr_candidates(L3, key)

    #######################################
    top_D, top_E = SOP_pipeline.extract_SOP_D_E(L3, triple_factor)
    print(f'top_D:{top_D}')
    print(f'top_E:{top_E}')

    bottom_direction = detect_sop_pin_orientation(image_root, "bottom.jpg")
    span = SOP_pipeline.extract_span(L3, triple_factor, top_D, top_E, bottom_direction)
    print(f'span:{span}')

    D2, E2 = SOP_pipeline.extract_bottom_D2_E2(L3, triple_factor, top_D, top_E)
    print(f'D2:{D2}')
    print(f'E2:{E2}')

    L = SOP_pipeline.extract_SOP_L(L3, triple_factor, bottom_direction, top_D, top_E, span)
    print(f'L:{L}')

    path = 'Result\Package_view\pin\SOP_adjacent_pin.txt'
    pin_box, pin_boxes = SOP_pipeline.extract_pin_boxes_from_txt(path)
    print(f'pin_box:{pin_box}')
    print(f'pin_boxes:{pin_boxes}')

    side_direction = detect_sop_pin_orientation(image_root, "side.jpg")
    bottom_b, bottom_L = SOP_pipeline.extract_bottom_b_L(L3, triple_factor, pin_box, bottom_direction, side_direction)
    print(f'bottom_b:{bottom_b}')
    print(f'bottom_L:{bottom_L}')

    bottom_pitch = SOP_pipeline.extract_bottom_pitch(L3, triple_factor, pin_boxes, bottom_direction, side_direction)
    print(f'bottom_pitch:{bottom_pitch}')

    print("\n>>> 开始整合参数列表 (精确值 + 模糊筛选)...")
    # 将计算结果打包
    calc_results = {
        'D': top_D, 'E1': top_E,
        'E': span, 'D2': D2, 'E2': E2,
        'L': L, 'b': bottom_b,
        'e': bottom_pitch
    }

    # 调用新的整合函数
    final_parameter_list = SOP_pipeline.get_integrated_parameter_list(L3, calc_results)
    parameter_list = generate_excel_parameter_list(final_parameter_list)

    print("最终参数列表：", parameter_list)
    return parameter_list

    # #######################################
    # # '''
    # #     输出QFP参数
    # #     nx,ny
    # #     pitch
    # #     high(A)
    # #     standoff(A1)
    # #     span_x,span_y
    # #     body_x,body_y
    # #     b
    # #     pad_x,pad_y
    # # '''
    # # # # 语义对齐
    # ## F4.8
    # L3 = common_pipeline.extract_pin_serials(L3, package_class)
    # L3 = common_pipeline.match_pairs_with_text(L3, key)
    # ## F4.9
    # L3 = common_pipeline.finalize_pairs(L3)
    # SOP_parameter_list, nx, ny,span_x = SOP_pipeline.find_SOP_parameter(L3)
    #
    # # # 整理获得的参数
    # parameter_list = get_SOP_parameter_data(SOP_parameter_list, nx, ny,span_x)
    #
    # print("开始测试F4.10", parameter_list)
    # # 20250621修改顺序
    # new_parameter_list = []
    # new_parameter_list.append(parameter_list[8])
    # new_parameter_list.append(parameter_list[6])
    # new_parameter_list.append(parameter_list[2])
    # new_parameter_list.append(parameter_list[3])
    # new_parameter_list.append(parameter_list[9])
    # new_parameter_list.append(parameter_list[0])
    # new_parameter_list.append(parameter_list[1])
    # new_parameter_list.append([0, '-', '-', '-'])
    # new_parameter_list.append([0, '-', '-', '-'])
    # new_parameter_list.append(parameter_list[4])
    # new_parameter_list.append(parameter_list[5])
    # new_parameter_list.append(parameter_list[12])
    # new_parameter_list.append([0, '-', '-', '-'])
    # new_parameter_list.append(parameter_list[10])
    # new_parameter_list.append(parameter_list[11])
    #
    # return new_parameter_list


if __name__ == "__main__":
    run_f4_pipeline_SOP(
        image_root=r"D:\20251124\PackageWizard1.1\Result\Package_extract\data",
        package_class="SOP",
        key=0,
        test_mode=0
    )
