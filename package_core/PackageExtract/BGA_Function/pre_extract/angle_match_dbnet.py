from package_core.PackageExtract.function_tool import find_list,recite_data


def calculate_iou(box1, box2):
    """
    计算两个框的IoU（交并比）
    box1, box2: [x1, y1, x2, y2]
    """
    # 计算交集区域
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # 检查是否有交集
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    # 计算交集面积
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # 计算各自面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集面积
    union_area = area1 + area2 - inter_area
    
    # 计算IoU
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou

def angle_find_matching_boxes(L3):
    """
    在dbnet_data中寻找与angle_pairs中角度外框重合的个体
    
    Args:
        dbnet_data: list of [x1, y1, x2, y2], dbnet识别到的文本框坐标
        angle_pairs: list of [x1, y1, x2, y2], 角度外框坐标
        iou_threshold: IoU阈值，用于判断是否重合
    
    Returns:
        angle_boxes_dicts: 匹配的角度外框字典列表
        new_dbnet_data: 剔除匹配框后的dbnet_data
        new_angle_pairs: 剥除匹配框后的angle_pairs
    """
    iou_threshold = 0.5

    for view in ("top", "bottom", "side", "detailed"):
        dbnet_key = f"{view}_dbnet_data"
        angle_pairs_key = f"{view}_angle_pairs"

        dbnet_data = find_list(L3, dbnet_key)
        angle_pairs = find_list(L3, angle_pairs_key)

        angle_boxes_dicts = []
        matched_dbnet_indices = set()  # 记录已经匹配的dbnet_data索引
        matched_angle_indices = set()  # 记录已经匹配的angle_pairs索引

        # 遍历angle_pairs中的每个角度外框
        for i, angle_box in enumerate(angle_pairs):
            if i in matched_angle_indices:
                continue  # 跳过已匹配的角度框

            best_iou = 0
            best_match_idx = -1

            # 在dbnet_data中寻找最佳匹配
            for j, text_box in enumerate(dbnet_data):
                if j in matched_dbnet_indices:
                    continue  # 跳过已匹配的文本框

                iou = calculate_iou(angle_box, text_box)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_match_idx = j

            # 如果找到匹配的文本框
            if best_match_idx != -1:
                matched_text_box = dbnet_data[best_match_idx]
                matched_dbnet_indices.add(best_match_idx)
                matched_angle_indices.add(i)

                # 创建字典
                dic = {
                    'location': matched_text_box,
                    'ocr_strings': None,  # 这里可以后续填充OCR识别结果
                    'Absolutely': ['angle']  # 根据需求填充绝对信息
                }
                angle_boxes_dicts.append(dic)

        # 构建剔除匹配框后的新数据
        new_dbnet_data = [box for j, box in enumerate(dbnet_data) if j not in matched_dbnet_indices]
        # new_angle_pairs = [box for j, box in enumerate(angle_pairs) if j not in matched_angle_indices]

        # recite_data(L3, dbnet_key, new_dbnet_data)
        # recite_data(L3, angle_pairs_key, new_angle_pairs)  # 修复：使用正确的键名
        recite_data(L3, f"{view}_angle_match_results", angle_boxes_dicts)

    return L3