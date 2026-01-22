# f4_pipeline_logic.py

import numpy as np
from typing import List, Dict, Any, Optional
# 导入 F4.1 - F4.5 所需的函数
from package_core.PackageExtract.BGA_Function.pre_extract.other_match_dbnet import other_match_boxes_by_overlap
from package_core.PackageExtract.BGA_Function.pre_extract.pin_match_dbnet import PINnum_find_matching_boxes
from package_core.PackageExtract.BGA_Function.pre_extract.angle_match_dbnet import angle_find_matching_boxes
from package_core.PackageExtract.BGA_Function.pre_extract.num_match_dbnet import num_match_size_boxes
from package_core.PackageExtract.BGA_Function.pre_extract.merge_box_and_ocr import process_all_variables
from package_core.PackageExtract.BGA_Function.pre_extract.data_process_with_Absolutely import process_recognized_strings


def _structure_raw_boxes(box_list: List[List[int]], category: str) -> List[Dict[str, Any]]:
    """
    一个辅助函数，用于将YOLO直接输出的BGA框列表转换为F4.5所需的字典结构。
    我们假设BGA框不需要与DBNet进行匹配。
    """
    structured_list = []
    if not box_list:
        return []

    for box in box_list:
        structured_list.append({
            'location': box,
            'small_boxes': [],  # 假设BGA框没有内部小框
            'ocr_strings': [],  # 待F4.5填充
            'Absolutely': [category]
        })
    return structured_list


def run_f4_1_to_f4_5_pipeline(
        img_path: str,
        dbnet_data: List[List[int]],
        other_boxes: List[List[int]],
        pin_serial_boxes: List[List[int]],
        angle_pair_boxes: List[List[int]],
        num_boxes: List[List[int]],
        bga_serial_num_boxes: Optional[List[List[int]]] = None,
        bga_serial_letter_boxes: Optional[List[List[int]]] = None,
        # yolox_pairs: List[List[int]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    执行 F4.1 到 F4.5 的完整流水线。

    F4.1[cite: 86]: 匹配 'other' 类别
    F4.2[cite: 94]: 匹配 'PIN_num' 类别
    F4.3[cite: 102]: 匹配 'angle' 类别
    F4.4[cite: 109]: 匹配 'num' 类别
    F4.5[cite: 116]: 对所有匹配框进行OCR识别，并处理字符串

    Args:
        img_path (str): 图像路径
        dbnet_data (List): DBNet 提取的所有文本框
        other_boxes (List): YOLO 提取的 'other' 框
        pin_serial_boxes (List): YOLO 提取的 'PIN_num' 框
        angle_pair_boxes (List): YOLO 提取的 'angle' 框
        num_boxes (List): YOLO 提取的 'num' 框
        bga_serial_num_boxes (Optional[List]): YOLO 提取的 BGA 数字框
        bga_serial_letter_boxes (Optional[List]): YOLO 提取的 BGA 字母框

    Returns:
        Dict: 处理完成的 L3 数据结构，包含所有类别的框、OCR结果和 max_nom_min 值。
    """

    # F4.1: 匹配 'other' 类别 [cite: 86]
    remaining_dbnet, new_other = other_match_boxes_by_overlap(dbnet_data, other_boxes)
    print(f"F4.1: 'other' 匹配完成。剩余 {len(remaining_dbnet)} 个dbnet框。")
    # F4.2: 匹配 'PIN_num' 类别 [cite: 94]
    remaining_dbnet, new_yolox_serial_num = PINnum_find_matching_boxes(remaining_dbnet, pin_serial_boxes)
    print(f"F4.2: 'PIN_num' 匹配完成。剩余 {len(remaining_dbnet)} 个dbnet框。")
    # F4.3: 匹配 'angle' 类别 [cite: 102]
    # 注意：angle_find_matching_boxes 返回3个值
    angle_boxes_dicts, remaining_dbnet, _ = angle_find_matching_boxes(remaining_dbnet, angle_pair_boxes)
    print(f"F4.3: 'angle' 匹配完成。剩余 {len(remaining_dbnet)} 个dbnet框。")
    # F4.4: 匹配 'num' 类别 [cite: 109]
    remaining_dbnet, new_yolox_num = num_match_size_boxes(remaining_dbnet, num_boxes)
    print(f"F4.4: 'num' 匹配完成。剩余 {len(remaining_dbnet)} 个dbnet框。")
    # F4.4x:
    # remaining_dbnet1, new_yolox_num1 = match_arrows_to_dbnet(yolox_pairs, remaining_dbnet, new_yolox_num)
    # --- BGA 数据处理 ---
    # 根据 data_process_with_Absolutely.py 的处理逻辑，BGA框(BGA_serial_num, BGA_serial_letter)
    # 似乎不与dbnet匹配，它们本身就是文本框。
    # 我们将其转换为F4.5所需的字典结构，类别指定为 'PIN_num' 以便
    # process_recognized_strings 能正确处理 BGA 字母和数字。
    new_BGA_serial_num = _structure_raw_boxes(bga_serial_num_boxes, 'PIN_num')
    new_BGA_serial_letter = _structure_raw_boxes(bga_serial_letter_boxes, 'PIN_num')

    # F4.5 (Part 1 - OCR) [cite: 116]
    # 调用 merge_box_and_ocr.py 中的 process_all_variables
    print("F4.5: 正在执行 OCR ...")
    ocr_results = process_all_variables(
        image_path=img_path,
        new_other=new_other,
        new_yolox_serial_num=new_yolox_serial_num,
        angle_boxes_dicts=angle_boxes_dicts,
        new_yolox_num=new_yolox_num,
        BGA_serial_num=new_BGA_serial_num,
        BGA_serial_letter=new_BGA_serial_letter
    )

    # F4.5 (Part 2 - String Processing) [cite: 116]
    # 调用 data_process_with_Absolutely.py 中的 process_recognized_strings
    print("F4.5: 正在处理 OCR 字符串...")
    processed_results = process_recognized_strings(
        new_other=ocr_results['new_other'],
        new_yolox_serial_num=ocr_results['new_yolox_serial_num'],
        angle_boxes_dicts=ocr_results['angle_boxes_dicts'],
        new_yolox_num=ocr_results['new_yolox_num'],
        BGA_serial_num=ocr_results.get('BGA_serial_num'),
        BGA_serial_letter=ocr_results.get('BGA_serial_letter')
    )

    print(f"pre_extract 流水线执行完毕。未匹配的dbnet框: {len(remaining_dbnet)}")

    # processed_results 就是 F4.6-F4.9 所需的 L3 数据
    return processed_results