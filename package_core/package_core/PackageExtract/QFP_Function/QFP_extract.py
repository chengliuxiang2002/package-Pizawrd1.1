"""封装 F4.6-F4.9 流程的便捷调用入口。"""

from __future__ import annotations
from typing import Iterable
import sys
import os
import numpy as np

from package_core.PackageExtract.QFP_Function import QFP_pipeline
from package_core.PackageExtract.QFP_Function.QFP_pipeline import get_integrated_parameter_list
from package_core.PackageExtract import common_pipeline

from package_core.PackageExtract.BGA_Function.pre_extract import (
    other_match_dbnet,
    pin_match_dbnet,
    angle_match_dbnet,
    num_match_dbnet,
    num_direction,
    match_triple_factor
)
# 获取当前脚本所在目录的绝对路径
current_script_path = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
sys.path.append(root_dir)
os.chdir(root_dir)
# 打印关键信息用于排查
print("当前脚本路径：", current_script_path)
print("计算出的根目录：", root_dir)
print("Python搜索路径：", sys.path)  # 查看root_dir是否已被添加

# from package_core.PackageExtract.BGA_Function import fill_triple_factor

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

    side_A = get_data('A')  # Package Height
    side_A1 = get_data('A1')  # Standoff
    top_E = get_data('E')  # span X
    top_D = get_data('D')  # span Y
    L = get_data('L')  # Lead Length
    bottom_b = get_data('b')  # Lead width
    side_A3 = get_data('c')  # Lead Thickness
    bottom_E2 = get_data('E2')  # Thermal X
    bottom_D2 = get_data('D2')  # Thermal Y
    body_x = get_data('D1')  # Body X (D1)
    body_y = get_data('E1')  # Body Y (E1)

    # --- 3. 组装最终列表 (不需要再单独写 bottom_pitch 的互换逻辑了) ---
    parameter_list = [
        ['', '', '', ''],  # 0 Number of pins along X（未算）
        ['', '', '', ''],  # 1 Number of pins along Y（未算）

        [''] + side_A,  # 2 Package Height (A)
        [''] + side_A1,  # 3 Standoff (A1)

        [''] + top_D,  # 4 Span X (D)
        [''] + top_E,  # 5 Span Y (E)

        [''] + body_x,  # 6 Body X (D1)
        [''] + body_y,  # 7 Body Y (E1)

        ['', '', '', ''],  # 8 Body draft (θ)
        ['', '', '', ''],  # 9 Edge Fillet radius

        [''] + L,  # 10 Lead Length (L)
        [''] + bottom_b,  # 11 Lead width (b)
        [''] + side_A3,  # 12 Lead Thickness (c)

        ['', '', '', ''],  # 13 Lead Radius (r)

        [''] + bottom_D2,  # 14 Thermal X (D2)
        [''] + bottom_E2  # 15 Thermal Y (E2)
    ]

    return parameter_list

def run_f4_pipeline_QFP(
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
    print("*****triple_factor*****", triple_factor)
    # ## （整理尺寸线与文本，生成初始配对候选）
    L3 = common_pipeline.preprocess_pairs_and_text(L3, key)


    L3 = common_pipeline.run_svtr_ocr(L3)
    L3 = common_pipeline.normalize_ocr_candidates(L3, key)





    ######################开始编写QFP的数字提取流程代码##########################


    top_D, top_E = QFP_pipeline.extract_top_D_E(L3, triple_factor)
    print(f'top_D:{top_D}')
    print(f'top_E:{top_E}')

    bottom_D2, bottom_E2 = QFP_pipeline.extract_bottom_D2_E2(L3, triple_factor, top_D, top_E)
    print(f'bottom_D2:{bottom_D2}')
    print(f'bottom_E2:{bottom_E2}')

    body_x, body_y = QFP_pipeline.extract_D1_E1_from_ocr(L3, top_D, top_E, bottom_D2, bottom_E2)
    print(f'D1:{body_x}')
    print(f'E1:{body_y}')

    calc_L = QFP_pipeline.extract_lead_length_L(L3, top_D, body_x)


    # ==================== 🛠️ 新增插入点：利用 Pin 信息提取 e 和 b ====================
    print("\n>>> 开始利用 Pin 信息校验 Pitch (e) 和 Width (b)...")

    # 1. 初始化变量
    calc_e = []
    calc_b = []

    # 2. 构造 Pin 文件路径 (利用你现有的 result_path 函数)
    pin_txt_path = result_path('Package_view', 'pin', 'QFP_adjacent_pins.txt')

    # 3. 读取 Pin 坐标

    if os.path.exists(pin_txt_path):
        try:
            # extract_pin_boxes_from_txt 通常返回三个值: pin_box(全部), pin_boxh(横向), pin_boxv(纵向)
            # 我们只需要 h 和 v
            _, pin_boxh, pin_boxv = common_pipeline.extract_pin_boxes_from_txt(pin_txt_path)
            print(f"  📖 读取 Pin 文件成功: H组{len(pin_boxh)}个, V组{len(pin_boxv)}个")

            # 4. 调用校验函数
            calc_e, calc_b = QFP_pipeline.extract_e_b_combined(L3, pin_boxh, pin_boxv)

            if calc_e is not None and len(calc_e) > 0:
                print(f"  ✅ 几何校验算出 e: {calc_e}")

            if calc_b is not None and len(calc_b) > 0:
                print(f"  ✅ 几何校验算出 b: {calc_b}")

        except Exception as e:
            print(f"  ⚠️ Pin 处理过程出错: {e}")
            # 出错时 calc_e/calc_b 保持为空列表，不影响后续逻辑
    else:
        print(f"  ⚠️ Pin 文件不存在: {pin_txt_path}")
    # ==============================================================================

    print("\n>>> 开始整合参数列表 (精确值 + 模糊筛选)...")

    # 将计算结果打包
    calc_results = {
        'D': top_D, 'E': top_E,
        'D1': body_x, 'E1': body_y,
        'D2': bottom_D2, 'E2': bottom_E2,
        # 🛠️ 把新算出来的 e 和 b 加进去，传给整合函数
        'e': calc_e,
        'b': calc_b,
        'L': calc_L
    }

    # 调用新的整合函数
    final_parameter_list = get_integrated_parameter_list(L3, calc_results)

    parameter_list = generate_excel_parameter_list(final_parameter_list)
    print(parameter_list)
    # ==================== 7. 最终结果打印 ====================


    return parameter_list

    #
    # parameter_list = [[''] + body_x,[''] + body_y,['', '', '', ''],['', '', '', ''],
    #                   [''] + side_A,[''] + side_A1,['', '', '', ''],[''] + top_D,[''] + top_E,
    #                   ['', '', '', ''],[''] + bottom_L,[''] + bottom_b,['', '', '', ''],['', '', '', ''],
    #                   [''] + bottom_D2,[''] + bottom_E2]


if __name__ == "__main__":
    run_f4_pipeline_QFP(
        image_root="D:\Graduate_Project\PackageWizard1.1\Result\Package_extract\data",
        package_class="QFP",
        key=0,
        test_mode=0
    )
