import os
import numpy as np
# 引用底层的 OCR 接口
from package_core.PackageExtract.get_pairs_data_present5_test import ocr_data
from package_core.PackageExtract.function_tool import recite_data


def run_ocr_on_yolox_locations(triple_factor_results, image_root):
    """
    核心逻辑：只对 Triple Factor 中的 YOLOX 框（location）进行 OCR。

    Args:
        triple_factor_results: match_triple_factor 的输出列表
        image_root: 图片文件夹路径

    Returns:
        triple_factor_results: 补充了 'ocr_strings' 字段的结果
    """
    print(f">> 开始执行基于 YOLOX 框的定向 OCR (针对完整行数据识别)...")

    # 遍历四个视图的结果
    for view_items in triple_factor_results:
        if not view_items:
            continue

        view_name = view_items[0]['view_name']
        img_path = os.path.join(image_root, f"{view_name}.jpg")

        if not os.path.exists(img_path):
            continue

        print(f"  正在处理视图: {view_name}, 待识别框数量: {len(view_items)}")

        for item in view_items:
            # 策略：不管有没有匹配到箭头，YOLOX 框本身就是高置信度的文本区域
            # 直接使用 'location' (YOLOX大框) 进行识别，而不是 'small_boxes' (DBNet小框)

            target_box = item.get('location')

            # 校验框的格式，确保是 [x1, y1, x2, y2]
            if target_box is None or len(target_box) != 4:
                item['ocr_strings'] = ""
                continue

            try:
                # 调用底层 OCR，注意传入的是列表 [target_box]
                # ocr_data 内部会自动裁剪图片并跑模型
                # 返回结果通常是 list，取第一个
                ocr_results = ocr_data(img_path, [target_box])

                if ocr_results and len(ocr_results) > 0:
                    # 成功识别，拿到完整字符串 "0.80x 12=9.60"
                    item['ocr_strings'] = ocr_results[0]
                else:
                    item['ocr_strings'] = ""

            except Exception as e:
                print(f"  OCR Error on box {target_box}: {e}")
                item['ocr_strings'] = ""

    print("<< 定向 OCR 完成")
    return triple_factor_results


def update_L3_with_yolox_ocr(L3, triple_results):
    """
    关键步骤：将 YOLOX 框 + OCR 结果回写到 L3 的 ocr_data 中。
    这样后续的 extract_pin_serials 和 data_wrangling 就能直接使用这些完整数据，
    完全替代掉原本碎片化的 DBNet 数据。
    """
    views_map = {0: "top", 1: "bottom", 2: "side", 3: "detailed"}

    for i, view_items in enumerate(triple_results):
        if i not in views_map: break
        view_name = views_map[i]

        # 构建符合 L3 标准的 ocr_data 列表
        # 结构: [{'location': np.array, 'ocr_strings': str, ...}, ...]
        new_ocr_data_list = []

        for item in view_items:
            text = item.get('ocr_strings', "")

            # 只有识别出内容的才保留，或者是为了保留位置信息也可以存空字符串
            # 这里建议保留，方便后续流程处理

            # 构造新的数据对象
            ocr_obj = {
                'location': np.array(item['location']),  # 确保转回 numpy，兼容后续计算
                'ocr_strings': text,
                'key_info': [],  # 后续 data_wrangling 会填充这个
                'matched_pairs_location': item.get('arrow_pairs', []),  # 把之前匹配的箭头也带上
                'matched_pairs_yinXian': [],  # 预留
                'Absolutely': item.get('Absolutely', []),
                'max_medium_min': []  # 后续计算
            }
            new_ocr_data_list.append(ocr_obj)

        # 覆盖 L3 中的数据
        # 注意：这里直接覆盖了原有的 svtr 结果
        key_name = f"{view_name}_ocr_data"
        recite_data(L3, key_name, new_ocr_data_list)
        print(f"  [{view_name}] 已更新 L3 数据，共 {len(new_ocr_data_list)} 条")

    return L3