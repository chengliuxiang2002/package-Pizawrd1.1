import numpy as np

from package_core.PackageExtract.function_tool import find_list, recite_data


def calculate_iou(box1, box2):
    """
    计算两个矩形框的IoU（交并比）
    box1, box2: [x1, y1, x2, y2]
    """
    # 计算交集区域
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # 计算交集面积
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    # 计算各自面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集面积
    union_area = area1 + area2 - inter_area
    
    # 计算IoU
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou

def calculate_overlap_ratio(box1, box2):
    """
    计算box1在box2中的重叠比例
    返回两个值：box1在box2中的比例，box2在box1中的比例
    """
    # 计算交集区域
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # 计算交集面积
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    # 计算各自面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算重叠比例
    ratio1 = inter_area / area1 if area1 > 0 else 0  # box1在box2中的比例
    ratio2 = inter_area / area2 if area2 > 0 else 0  # box2在box1中的比例
    
    return ratio1, ratio2

def is_inside(box_small, box_large, threshold=0.8):
    """
    检查小框是否在大框内部
    threshold: 小框面积有多少比例在大框内才算"内部"
    """
    ratio_in_large, _ = calculate_overlap_ratio(box_small, box_large)
    return ratio_in_large >= threshold

def is_center_inside(box_small, box_large):
    """
    检查小框的中心点是否在大框内部
    """
    center_x = (box_small[0] + box_small[2]) / 2
    center_y = (box_small[1] + box_small[3]) / 2
    
    return (box_large[0] <= center_x <= box_large[2] and 
            box_large[1] <= center_y <= box_large[3])

def is_mostly_inside(box_small, box_large, threshold=0.7):
    """
    检查小框是否大部分在大框内部
    """
    ratio_in_large, _ = calculate_overlap_ratio(box_small, box_large)
    return ratio_in_large >= threshold

def expand_box(box, expand_pixels=5):
    """
    扩展框的边界
    """
    return [
        box[0] - expand_pixels,
        box[1] - expand_pixels,
        box[2] + expand_pixels,
        box[3] + expand_pixels
    ]

def calculate_match_score(dbnet_box, serial_box, expanded_serial_box=None):
    """
    计算小框与大框的匹配分数
    分数越高表示匹配越好
    """
    # 基础IoU
    iou = calculate_iou(dbnet_box, serial_box)
    
    # 重叠比例
    ratio_in_serial, _ = calculate_overlap_ratio(dbnet_box, serial_box)
    
    # 中心点是否在内部
    center_inside = is_center_inside(dbnet_box, serial_box)
    
    # 是否大部分在内部
    mostly_inside = is_mostly_inside(dbnet_box, serial_box)
    
    # 扩展边界匹配
    if expanded_serial_box:
        iou_expanded = calculate_iou(dbnet_box, expanded_serial_box)
        ratio_in_expanded, _ = calculate_overlap_ratio(dbnet_box, expanded_serial_box)
        center_inside_expanded = is_center_inside(dbnet_box, expanded_serial_box)
        mostly_inside_expanded = is_mostly_inside(dbnet_box, expanded_serial_box)
    else:
        iou_expanded = 0
        ratio_in_expanded = 0
        center_inside_expanded = False
        mostly_inside_expanded = False
    
    # 计算综合匹配分数
    score = 0
    
    # IoU权重最高
    score += iou * 0.3
    
    # 内部比例也很重要
    score += ratio_in_serial * 0.3
    
    # 扩展边界匹配
    score += iou_expanded * 0.15
    
    # 中心点匹配
    if center_inside or center_inside_expanded:
        score += 0.1
    
    # 大部分在内部
    if mostly_inside or mostly_inside_expanded:
        score += 0.15
    
    return score

def PINnum_find_matching_boxes(L3):
    """
    在dbnet_data中寻找与yolox_serial_num外框重合的个体，使用更严格的匹配策略
    
    Args:
        dbnet_data: list of [x1, y1, x2, y2], dbnet识别到的文本框
        yolox_serial_num: list of [x1, y1, x2, y2], YOLOX识别的PIN序号外框
        iou_threshold: IoU阈值，大于该值认为重合
        inside_threshold: 小框在大框内的面积比例阈值
        use_center_check: 是否使用中心点检查
        use_expanded_check: 是否使用扩展边界检查
        expand_pixels: 扩展边界像素数
        min_match_score: 最小匹配分数阈值
    
    Returns:
        new_dbnet_data: 剔除已匹配框后的dbnet_data
        new_yolox_serial_num: 匹配的PIN序号外框信息列表，按指定字典结构
    """
    iou_threshold = 0.2
    inside_threshold = 0.7
    use_center_check = True
    use_expanded_check = True
    expand_pixels = 10
    min_match_score = 0.3

    for view in ("top", "bottom", "side", "detailed"):
        dbnet_key = f"{view}_dbnet_data"
        yolox_serial_num_key = f"{view}_yolox_serial_num"

        dbnet_data = find_list(L3, dbnet_key)
        yolox_serial_num = find_list(L3, yolox_serial_num_key)

        # 初始化结果
        new_yolox_serial_num = []
        matched_indices = set()

        print(f"=== 开始匹配 ===")
        print(f"yolox_serial_num 数量: {len(yolox_serial_num)}")
        print(f"dbnet_data 数量: {len(dbnet_data)}")

        # 为每个大框创建匹配列表
        for i, serial_box in enumerate(yolox_serial_num):
            matched_info = {
                'location': serial_box,
                'small_boxes': [],
                'ocr_strings': None,
                'Absolutely': ['PIN_num']
            }
            new_yolox_serial_num.append(matched_info)

        # 第一步：为每个小框找到最优匹配的大框
        print(f"\n=== 第一步：为每个小框寻找最优匹配 ===")
        for dbnet_idx, dbnet_box in enumerate(dbnet_data):
            if dbnet_idx in matched_indices:
                continue

            best_match_score = -1
            best_match_large_idx = -1
            match_details = {}

            # 遍历所有大框，计算匹配分数
            for large_idx, serial_box in enumerate(yolox_serial_num):
                # 计算扩展边界
                expanded_serial_box = expand_box(serial_box, expand_pixels) if use_expanded_check else serial_box

                # 计算匹配分数
                score = calculate_match_score(dbnet_box, serial_box, expanded_serial_box)

                # 计算其他匹配指标
                iou = calculate_iou(dbnet_box, serial_box)
                ratio_in_serial, _ = calculate_overlap_ratio(dbnet_box, serial_box)
                center_inside = is_center_inside(dbnet_box, serial_box)
                mostly_inside = is_mostly_inside(dbnet_box, serial_box)

                match_details[large_idx] = {
                    'score': score,
                    'iou': iou,
                    'ratio_in_serial': ratio_in_serial,
                    'center_inside': center_inside,
                    'mostly_inside': mostly_inside
                }

                # 更新最优匹配
                if score > best_match_score:
                    best_match_score = score
                    best_match_large_idx = large_idx

            # 检查是否满足匹配条件
            if best_match_large_idx >= 0:
                best_match_details = match_details[best_match_large_idx]

                # 检查是否满足任一匹配条件（更严格的条件）
                match_found = False
                match_reason = ""

                # 条件1: IoU足够高
                if best_match_details['iou'] >= iou_threshold:
                    match_found = True
                    match_reason = f"IoU匹配: IoU={best_match_details['iou']:.4f}"
                # 条件2: 大部分在大框内部
                elif best_match_details['mostly_inside']:
                    match_found = True
                    match_reason = f"大部分内部匹配: 在大框内比例={best_match_details['ratio_in_serial']:.4f}"
                # 条件3: 中心点在大框内部且匹配分数足够高
                elif best_match_details['center_inside'] and use_center_check and best_match_score >= min_match_score:
                    match_found = True
                    match_reason = f"中心点匹配: 中心点在大框内, 分数={best_match_score:.4f}"
                # 条件4: 综合匹配分数足够高
                elif best_match_score >= min_match_score:
                    match_found = True
                    match_reason = f"综合匹配: 分数={best_match_score:.4f}"

                if match_found:
                    matched_indices.add(dbnet_idx)
                    new_yolox_serial_num[best_match_large_idx]['small_boxes'].append(dbnet_box)
                    print(f"小框 {dbnet_idx} -> 大框 {best_match_large_idx+1}: {match_reason}")
                else:
                    print(f"小框 {dbnet_idx} 未匹配任何大框: 最高分数={best_match_score:.4f}, 最高IoU={best_match_details['iou']:.4f}, 内部比例={best_match_details['ratio_in_serial']:.4f}")
            else:
                print(f"小框 {dbnet_idx} 未找到任何匹配的大框")

        # 第二步：汇总匹配结果
        print(f"\n=== 第二步：匹配结果汇总 ===")
        total_matched = 0
        for i, matched_info in enumerate(new_yolox_serial_num):
            count = len(matched_info['small_boxes'])
            total_matched += count
            print(f"大框 {i+1} 匹配到 {count} 个小框")
            for j, small_box in enumerate(matched_info['small_boxes']):
                iou = calculate_iou(small_box, matched_info['location'])
                ratio, _ = calculate_overlap_ratio(small_box, matched_info['location'])
                print(f"  小框 {j+1}: IoU={iou:.4f}, 内部比例={ratio:.4f}")

        # 构建剔除已匹配框后的dbnet_data
        new_dbnet_data = [box for idx, box in enumerate(dbnet_data) if idx not in matched_indices]

        print(f"\n=== 最终匹配结果 ===")
        print(f"总共匹配了 {len(matched_indices)} 个文本框")
        print(f"剩余 {len(new_dbnet_data)} 个未匹配文本框")

        # 打印未匹配的文本框
        if new_dbnet_data:
            print(f"\n未匹配的文本框:")
            for idx, box in enumerate(new_dbnet_data):
                print(f"  文本框: {box}")

                # 分析为什么这个文本框没有被匹配
                for i, serial_box in enumerate(yolox_serial_num):
                    iou = calculate_iou(box, serial_box)
                    ratio_in_serial, _ = calculate_overlap_ratio(box, serial_box)
                    center_inside = is_center_inside(box, serial_box)

                    print(f"    与 PIN 外框 {i+1}: IoU={iou:.4f}, 重叠比例={ratio_in_serial:.4f}, 中心点在内={center_inside}")

        # recite_data(L3, dbnet_key, new_dbnet_data)
        # recite_data(L3, yolox_serial_num_key, new_yolox_serial_num)
        recite_data(L3, f"{view}_pin_match_results", new_yolox_serial_num)

    return L3

# 辅助函数：检查两个框是否有任何重叠
def has_overlap(box1, box2):
    """检查两个框是否有任何重叠"""
    return not (box1[2] <= box2[0] or box1[0] >= box2[2] or 
                box1[3] <= box2[1] or box1[1] >= box2[3])

# 辅助函数：计算重叠面积
def calculate_overlap_area(box1, box2):
    """计算两个框的重叠面积"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0
    
    return (x2_inter - x1_inter) * (y2_inter - y1_inter)