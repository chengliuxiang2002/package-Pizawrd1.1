import os
import queue
import threading
from typing import Iterable  # [NEW]

import numpy as np

# 导入新的 F4.1-F4.5 流水线
from package_core.PackageExtract.BGA_Function.pre_extract.f4_pipeline_logic import run_f4_1_to_f4_5_pipeline

# 导入 F4.1-F4.5 所需的 Pre-Steps 和工具
from package_core.PackageExtract.common_pipeline import (
    yolo_classify, dbnet_get_text_box, DEFAULT_VIEWS
)

# 导入 F4.6-F4.9流程中的步骤
from package_core.PackageExtract.BGA_Function.BGA_cal_pin import find_pin
import package_core.PackageExtract.get_pairs_data_present5_test as pairs_module
from package_core.PackageExtract.common_pipeline import (
    compute_qfp_parameters,  # (10) F4.9
    enrich_pairs_with_lines,  # (3) F4.7
    # extract_pin_serials,      # (7) F4.8 [SKIPPED, F4.1-F4.5 已完成]
    finalize_pairs,  # (9) F4.8
    # get_data_location_by_yolo_dbnet, # (1) [REPLACED]
    match_pairs_with_text,  # (8) F4.8
    # normalize_ocr_candidates, # (6) F4.7 [SKIPPED, F4.1-F4.5 已完成]
    prepare_workspace,
    preprocess_pairs_and_text,  # (4) F4.7
    # remove_other_annotations, # (2) F4.6 [SKIPPED, F4.1-F4.5 已完成]
    # run_svtr_ocr,             # (5) F4.7 [SKIPPED, F4.1-F4.5 已完成]
    # 导入路径变量
    DATA,
    DATA_BOTTOM_CROP,
    DATA_COPY,
    ONNX_OUTPUT,
    OPENCV_OUTPUT,
    OPENCV_OUTPUT_LINE,
    YOLO_DATA,
)
from package_core.PackageExtract.function_tool import (
    get_QFP_parameter_data,
    alter_QFP_parameter_data
)


def extract_BGA(package_classes):
    """执行 BGA 封装参数提取的主流程。"""

    prepare_workspace(
        DATA,
        DATA_COPY,
        DATA_BOTTOM_CROP,
        ONNX_OUTPUT,
        OPENCV_OUTPUT,
    )
    test_mode = 0
    key = test_mode

    # F4.1-F4.5 流程 ---
    views: Iterable[str] = DEFAULT_VIEWS
    L3 = []  # L3 必须是 list 才能被 F4.6+ 步骤使用

    for view in views:
        print(f"\n--- [F4.1-F4.5] 正在处理视图: {view} ---")
        img_path = os.path.join(DATA, f"{view}.jpg")

        # 定义空数据以备图像不存在时使用
        empty_data_4 = np.empty((0, 4))
        empty_data_10 = np.empty((0, 10))  # BGA 框有时是10列

        if os.path.exists(img_path):
            # 1. "Pre-step": 运行 YOLO 和 DBNet
            yolo_results = yolo_classify(
                img_path=img_path,
                package_classes=package_classes
            )
            (yolox_pairs, yolox_num, yolox_serial_num, pin, other,
             pad, border, angle_pairs, BGA_serial_num, BGA_serial_letter) = yolo_results

            dbnet_data = dbnet_get_text_box(img_path)

            # 2. "F4.1 - F4.5": 运行新的流水线
            # (将 np.ndarray 转换为 list 以匹配 f4_pipeline_logic 的输入类型)
            processed_view_data = run_f4_1_to_f4_5_pipeline(
                img_path=img_path,
                dbnet_data=dbnet_data.tolist() if isinstance(dbnet_data, np.ndarray) else dbnet_data,
                other_boxes=other.tolist() if isinstance(other, np.ndarray) else other,
                pin_serial_boxes=yolox_serial_num.tolist() if isinstance(yolox_serial_num, np.ndarray) else yolox_serial_num,
                angle_pair_boxes=angle_pairs.tolist() if isinstance(angle_pairs, np.ndarray) else angle_pairs,
                num_boxes=yolox_num.tolist() if isinstance(yolox_num, np.ndarray) else yolox_num,
                bga_serial_num_boxes=BGA_serial_num.tolist() if isinstance(BGA_serial_num, np.ndarray) else BGA_serial_num,
                bga_serial_letter_boxes=BGA_serial_letter.tolist() if isinstance(BGA_serial_letter, np.ndarray) else BGA_serial_letter,
                # yolox_pairs = yolox_pairs
            )

            # 3. "Adaptation": 将新旧数据打包成 L3 (list) 格式

            # F4.6+步骤需要这些 *原始* 输入
            L3.append({"list_name": f"{view}_yolox_pairs", "list": yolox_pairs})
            L3.append({"list_name": f"{view}_dbnet_data", "list": dbnet_data})
            L3.append({"list_name": f"{view}_border", "list": border})
            L3.append({"list_name": f"{view}_pin", "list": pin})
            L3.append({"list_name": f"{view}_pad", "list": pad})

            # F4.6+步骤将使用这些 *已处理* 的输入
            # (我们用 F4.1-F4.5 的输出覆盖 L3 中的相应字段)
            L3.append({"list_name": f"{view}_other", "list": processed_view_data.get('new_other', [])})
            L3.append({"list_name": f"{view}_yolox_num", "list": processed_view_data.get('new_yolox_num', [])})
            L3.append({"list_name": f"{view}_yolox_serial_num", "list": processed_view_data.get('new_yolox_serial_num', [])})
            L3.append({"list_name": f"{view}_angle_pairs", "list": processed_view_data.get('angle_boxes_dicts', [])})

            if view == "bottom":
                L3.append({"list_name": "bottom_BGA_serial_letter", "list": processed_view_data.get('BGA_serial_letter', empty_data_10)})
                L3.append({"list_name": "bottom_BGA_serial_num", "list": processed_view_data.get('BGA_serial_num', empty_data_10)})
        else:
            # 图像不存在，填充空数据
            print(f"警告: {img_path} 未找到, 为 {view} 视图填充空数据。")
            for k in ("dbnet_data", "yolox_pairs", "yolox_num", "yolox_serial_num", "pin", "other", "pad", "border", "angle_pairs"):
                L3.append({"list_name": f"{view}_{k}", "list": empty_data_4})
            if view == "bottom":
                L3.append({"list_name": "bottom_BGA_serial_letter", "list": empty_data_10})
                L3.append({"list_name": "bottom_BGA_serial_num", "list": empty_data_10})

    print("--- [F4.1-F4.5] 新流水线执行完毕 ---")
    print("--- [F4.6-F4.9] 开始执行流水线剩余步骤 ---")

    # --- F4.6 - F4.9 流程 ---
    # (2) 剔除 OTHER 类型干扰框
    # L3 = remove_other_annotations(L3) # [SKIPPED] (F4.1 已完成)
    # print("[F4.6] remove_other_annotations (已由F4.1完成) ... SKIPPED")

    # (3) 寻找尺寸线的配对边界 (F4.7 Part 1)
    L3 = enrich_pairs_with_lines(L3, DATA, key)  # [RUN]
    print("[F4.7] 寻找尺寸线的配对边界 ... DONE")

    # (4) 整理尺寸线与文本，生成初步候选 (F4.7 Part 2)
    L3 = preprocess_pairs_and_text(L3, key)  # [RUN]
    print("[F4.7] 整理尺寸线与文本，生成初步候选 ... DONE")

    # (5) 执行 SVTR OCR 识别 (F4.7 Part 3)
    # L3 = run_svtr_ocr(L3) # [SKIPPED] (F4.5 已完成)
    # print("[F4.7] run_svtr_ocr (已由F4.5完成) ... SKIPPED")
    # (6) OCR 后处理，清洗文本候选 (F4.7 Part 4)
    # L3 = normalize_ocr_candidates(L3, key) # [SKIPPED] (F4.5 已完成)
    # print("[F4.7] normalize_ocr_candidates (已由F4.5完成) ... SKIPPED")
    # (7) 提取序号/PIN 结构信息 (F4.8 Part 1)
    # L3 = extract_pin_serials(L3, package_classes) # [SKIPPED] (F4.2/F4.5 已完成)
    # print("[F4.8] extract_pin_serials (已由F4.2/F4.5完成) ... SKIPPED")

    # (8) 匹配尺寸线与文本 (F4.8 Part 2)
    L3 = match_pairs_with_text(L3, key)  # [RUN]
    print("[F4.8] 匹配尺寸线与文本 ... DONE")

    # (9) 整理配对结果 (F4.8 Part 3)
    L3 = finalize_pairs(L3)  # [RUN]
    print("[F4.8] 整理配对结果 ... DONE")

    # (10) 语义对齐，生成参数候选 (F4.9)
    QFP_parameter_list, nx, ny = compute_qfp_parameters(L3)  # [RUN]
    print("[F4.9] 语义对齐，生成参数候选 ... DONE")

    parameter_list = get_QFP_parameter_data(QFP_parameter_list, nx, ny)
    parameter_list = alter_QFP_parameter_data(parameter_list)

    return parameter_list


