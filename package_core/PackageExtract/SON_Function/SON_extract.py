from typing import Iterable
from package_core.PackageExtract.function_tool import *
from package_core.PackageExtract.get_pairs_data_present5_test import *
from package_core.PackageExtract import common_pipeline
from package_core.PackageExtract.SON_Function import SON_pipeline
from package_core.PackageExtract.BGA_Function.pre_extract import (
    other_match_dbnet,
    pin_match_dbnet,
    angle_match_dbnet,
    num_match_dbnet,
    num_direction,
    match_triple_factor,
    targeted_ocr
)


def safe_swap_1_3(data_list):
    """
    如果列表长度至少为3，则交换索引0和索引2的元素。
    否则什么都不做（防止空列表报错）。
    """
    if len(data_list) >= 3:
        data_list[0], data_list[2] = data_list[2], data_list[0]
    return data_list

def extract_SON(
        image_root: str,
        package_classes: str,
        key: int = 0,
        test_mode: int = 0
):
    """
    执行 SON 封装参数提取流程 (重构版 - 遵循 F4 Pipeline 标准)

    Args:
        image_root: 图片根目录路径
        package_classes: 封装类型 (如 "SON")
        key: OCR 清洗策略参数 (默认 0)
        test_mode: 测试模式开关 (默认 0)
    """
    # 1. 确定视图列表
    if os.path.exists(image_root):
        views_items = []
        for item in os.listdir(image_root):
            item_path = os.path.join(image_root, item)
            if os.path.isfile(item_path) and item.lower().endswith(('.jpg', '.jpeg', '.png')):
                view_name = os.path.splitext(item)[0]
                views_items.append(view_name)
        views: Iterable[str] = views_items
    else:
        views: Iterable[str] = common_pipeline.DEFAULT_VIEWS

    print(f"正在处理视图: {views}")
    print(f"图片路径: {image_root}")

    # =========================================================
    # F4 Pipeline 核心流程
    # =========================================================

    # (1) [F4.0] 初始位置获取
    print("开始测试F4.0")
    L3 = common_pipeline.get_data_location_by_yolo_dbnet(image_root, package_classes, view_names=views)
    # L3 = common_pipeline.remove_other_annotations(L3)
    print("开始测试F4.1")
    L3 = other_match_dbnet.other_match_boxes_by_overlap(L3)
    print("开始测试F4.2")
    L3 = pin_match_dbnet.PINnum_find_matching_boxes(L3)
    print("开始测试F4.3")
    L3 = angle_match_dbnet.angle_find_matching_boxes(L3)
    print("开始测试F4.4")
    L3 = num_match_dbnet.num_match_size_boxes(L3)
    print("开始测试F4.5")
    L3 = num_direction.add_direction_field_to_yolox_nums(L3)
    print("开始测试F4.6")
    L3 = common_pipeline.enrich_pairs_with_lines(L3, image_root, test_mode)
    print("开始测试F4.7")
    triple_factor = match_triple_factor.match_arrow_pairs_with_yolox(L3, image_root)
    print("*****triple_factor*****:", triple_factor)
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

    # ######################开始编写SON的数字提取流程代码##########################
    side_A, side_A3, side_A1 = SON_pipeline.extract_side_A_A1_A3(L3)
    print(f'side_A:{side_A}')
    print(f'side_A3:{side_A3}')
    print(f'side_A1:{side_A1}')

    top_D, top_E = SON_pipeline.extract_top_D_E(L3, triple_factor)
    print(f'top_D:{top_D}')
    print(f'top_E:{top_E}')

    bottom_D2, bottom_E2 = SON_pipeline.extract_bottom_D2_E2(L3, triple_factor, top_D, top_E)
    print(f'bottom_D2:{bottom_D2}')
    print(f'bottom_E2:{bottom_E2}')

    path = 'Result\Package_view\pin\SON_adjacent_pin.txt'
    pin_box, pin_boxes = SON_pipeline.extract_pin_boxes_from_txt(path)
    print(f'pin_box:{pin_box}')
    print(f'pin_boxes:{pin_boxes}')


    bottom_b, bottom_L = SON_pipeline.extract_bottom_b_L(L3, triple_factor, pin_box)
    print(f'bottom_b:{bottom_b}')
    print(f'bottom_L:{bottom_L}')


    bottom_pitch = SON_pipeline.extract_bottom_pitch(L3,triple_factor,pin_boxes)
    print(f'bottom_pitch:{bottom_pitch}')


    all_vars = [
        bottom_pitch, side_A, side_A1, top_D, top_E,
        bottom_L, bottom_b, side_A3, bottom_D2, bottom_E2
    ]

    for var in all_vars:
        safe_swap_1_3(var)

    parameter_list = [
        [''] + bottom_pitch,
        ['', '', '', ''],
        [''] + side_A,
        [''] + side_A1,
        ['', '', '', ''],
        [''] + top_D,
        [''] + top_E,
        ['', '', '', ''],
        [''] + bottom_L,
        [''] + bottom_b,
        [''] + side_A3,
        ['', '', '', ''],
        [''] + bottom_D2,
        [''] + bottom_E2
    ]

    return parameter_list

    # #########################################################################
    #
    # print("开始测试F4.8")
    # L3 = common_pipeline.extract_pin_serials(L3, package_classes)
    # L3 = common_pipeline.match_pairs_with_text(L3, key)
    # print("开始测试F4.9")
    # L3 = common_pipeline.finalize_pairs(L3)
    # SON_parameter_list, nx, ny = SON_pipeline.compute_SON_parameters(L3)
    #
    # parameter_list = get_SON_parameter_data(SON_parameter_list, nx, ny)
    #
    # print("开始测试F4.10",parameter_list)
    # # 20250621修改顺序
    # new_parameter_list = []
    # new_parameter_list.append(parameter_list[8])
    # new_parameter_list.append(parameter_list[6])
    # new_parameter_list.append(parameter_list[2])
    # new_parameter_list.append(parameter_list[3])
    # new_parameter_list.append([0, '-', '-', '-'])
    # new_parameter_list.append(parameter_list[0])
    # new_parameter_list.append(parameter_list[1])
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
    extract_SON(
        image_root=r"D:\20251124\PackageWizard1.1\Result\Package_extract\data",
        package_classes="SON",
        key=0,
        test_mode=0
    )