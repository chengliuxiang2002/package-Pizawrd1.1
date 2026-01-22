from package_core.PackageExtract.function_tool import find_list, recite_data

import numpy as np
def num_match_size_boxes(L3):
    """
    在dbnet_data中寻找被yolox_num中尺寸外框包含的文本框

    Args:
        dbnet_data: list of [x1, y1, x2, y2] - 文本框坐标
        yolox_num: list of [x1, y1, x2, y2] - 尺寸外框坐标
        overlap_threshold: float - 重叠阈值，小框多大比例在大框内才被视为匹配

    Returns:
        new_dbnet_data: 剔除匹配框后的剩余文本框
        new_yolox_num: 包含小框位置信息的尺寸外框列表
    """
    overlap_threshold = 0.5
    all_results = []  # 收集所有结果
    for view in ("top", "bottom", "side", "detailed"):
        dbnet_key = f"{view}_dbnet_data_back"
        yolox_num_key = f"{view}_yolox_num"
        yolox_pairs_key = f"{view}_yolox_pairs"
        dbnet_data = find_list(L3, dbnet_key)
        yolox_num = find_list(L3, yolox_num_key)
        yolox_pairs = find_list(L3, yolox_pairs_key)
        print(f'view:{view}')
        print(f'yolox_num处理前{yolox_num}')
        def calculate_iou(box1, box2):
            """计算两个框的交并比"""
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2

            # 计算交集
            x_left = max(x1_1, x1_2)
            y_top = max(y1_1, y1_2)
            x_right = min(x2_1, x2_2)
            y_bottom = min(y2_1, y2_2)

            if x_right < x_left or y_bottom < y_top:
                return 0.0

            # 交集面积
            intersection_area = (x_right - x_left) * (y_bottom - y_top)

            # 并集面积
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = area1 + area2 - intersection_area

            # 交并比
            iou = intersection_area / union_area if union_area > 0 else 0

            return iou

        def calculate_overlap_ratio(box1, box2):
            """
            计算两个框的重叠比例
            返回两个值：box1在box2中的重叠比例，box2在box1中的重叠比例
            """
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2

            # 计算交集
            x_left = max(x1_1, x1_2)
            y_top = max(y1_1, y1_2)
            x_right = min(x2_1, x2_2)
            y_bottom = min(y2_1, y2_2)

            if x_right < x_left or y_bottom < y_top:
                return 0.0, 0.0

            # 交集面积
            intersection_area = (x_right - x_left) * (y_bottom - y_top)

            # 两个框各自的面积
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

            # 重叠比例
            ratio1_to_2 = intersection_area / area1 if area1 > 0 else 0
            ratio2_to_1 = intersection_area / area2 if area2 > 0 else 0

            return ratio1_to_2, ratio2_to_1

        new_yolox_num = []
        matched_indices = set()  # 记录已匹配的dbnet_data索引

        # 遍历所有尺寸外框（yolox_num）
        for large_box in yolox_num:
            matched_small_boxes = []
            ocr_strings = []

            # 在dbnet_data中寻找与当前大框有足够重叠的框
            for idx, dbnet_box in enumerate(dbnet_data):
                if idx in matched_indices:
                    continue

                # 计算两个方向的重叠比例
                dbnet_in_yolo, yolo_in_dbnet = calculate_overlap_ratio(dbnet_box, large_box)

                # 计算IoU
                iou = calculate_iou(dbnet_box, large_box)

                # 如果满足以下任一条件，则认为匹配：
                # 1. dbnet框大部分在yolo框内 (dbnet_in_yolo >= threshold)
                # 2. yolo框大部分在dbnet框内 (yolo_in_dbnet >= threshold)
                # 3. 两者有显著重叠 (iou >= threshold/2)
                if (dbnet_in_yolo >= overlap_threshold or
                        yolo_in_dbnet >= overlap_threshold or
                        iou >= overlap_threshold / 2):
                    matched_small_boxes.append(dbnet_box)
                    ocr_strings.append("")
                    matched_indices.add(idx)

            # 创建大框字典
            box_dict = {
                'location': large_box,
                'small_boxes': matched_small_boxes,
                'ocr_strings': ocr_strings,
                'Absolutely': ['num']
            }
            new_yolox_num.append(box_dict)

        # 从dbnet_data中剔除已匹配的文本框
        new_dbnet_data = [box for idx, box in enumerate(dbnet_data) if idx not in matched_indices]
        new_dbnet_5, new_yolox_num_supply = match_arrows_to_dbnet(yolox_pairs, new_dbnet_data, new_yolox_num)
        new_yolox_num_supply = precise_merge_overlapping_boxes(new_yolox_num_supply)
        # 修改这里：提取每个字典中的 'location' 值，组成 [n, 4] 的列表
        location_list = []
        for item in new_yolox_num_supply:
            if isinstance(item, dict) and 'location' in item:
                location_list.append(item['location'])
            elif isinstance(item, (list, tuple)) and len(item) >= 4:
                # 如果已经是坐标列表格式，直接添加
                location_list.append(item)

        # 存储提取后的 location 列表
        recite_data(L3, dbnet_key, new_dbnet_5)
        location_list = np.array(location_list).round(3).tolist()
        # recite_data(L3, f"{view}_num_match_results", location_list)
        recite_data(L3, f"{view}_yolox_num", location_list)
        
        print(f'view:{view}')
        print(f'location_list处理后{location_list}')
        
        # yolox_num_key = f"{view}_yolox_num"
        
        # 收集结果用于打印
        view_result = find_list(L3, f"{view}_yolox_num")
        all_results.append({
            'view': view,
            'data': view_result,
            'count': len(view_result)
        })

    # 在函数最后统一打印
    print("\n=== 所有视图匹配结果 ===")
    for result in all_results:
        print(f"{result['view']}: 找到 {result['count']} 个匹配结果")
        if result['data']:
            print(f"  数据: {result['data']}")
        else:
            print(f"  数据: 空")

    return L3
    #     recite_data(L3, f"{view}_num_match_results", new_yolox_num_supply)

    #     new_yolox_num_key = f"{view}_num_match_results"
    #     new_yolox_num = find_list(L3, new_yolox_num_key)
    #     print(new_yolox_num)
    # return L3


def match_arrows_to_dbnet(yolox_pairs, dbnet_data, yolox_num, distance_threshold=40, overlap_threshold=0.1, high_overlap_threshold=0.3):
    """
    将未匹配的yolox_pairs箭头对与剩余的dbnet数据进行匹配

    Args:
        yolox_pairs: list of [x1, y1, x2, y2, direction] - 箭头对坐标和方向(0:外向, 1:内向)
        dbnet_data: list of [x1, y1, x2, y2] - 剩余的文本框坐标
        yolox_num: list of dict - 已有的尺寸外框数据
        distance_threshold: float - 距离阈值(像素)
        overlap_threshold: float - 重叠阈值(用于外向箭头)
        high_overlap_threshold: float - 高重叠阈值，用于判断是否几乎重合

    Returns:
        new_dbnet_data: 剔除匹配框后的剩余文本框
        new_yolox_num: 补充数据后的尺寸外框列表
    """

    def calculate_box_distance(box1, box2):
        """计算两个框之间的最小距离"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # 计算两个框在x轴和y轴上的投影
        left = x2_1 < x1_2
        right = x1_1 > x2_2
        bottom = y2_1 < y1_2
        top = y1_1 > y2_2

        if not (left or right or bottom or top):
            # 两个框有重叠，距离为0
            return 0

        # 计算两个框在x轴和y轴上的距离
        dx = max(x1_2 - x2_1, x1_1 - x2_2, 0)
        dy = max(y1_2 - y2_1, y1_1 - y2_2, 0)

        return (dx ** 2 + dy ** 2) ** 0.5

    def calculate_overlap(box_small, box_large):
        """计算小框在大框内的重叠比例"""
        x1_s, y1_s, x2_s, y2_s = box_small
        x1_l, y1_l, x2_l, y2_l = box_large

        # 计算交集
        x_left = max(x1_s, x1_l)
        y_top = max(y1_s, y1_l)
        x_right = min(x2_s, x2_l)
        y_bottom = min(y2_s, y2_l)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        small_box_area = (x2_s - x1_s) * (y2_s - y1_s)

        return intersection_area / small_box_area if small_box_area > 0 else 0

    def get_box_direction(box):
        """判断框的方向：水平或竖直"""
        width = box[2] - box[0]
        height = box[3] - box[1]

        # 如果宽度大于高度，则为水平方向；否则为竖直方向
        return "horizontal" if width > height else "vertical"

    def is_num_contained_by_arrow(arrow_box, num_box, arrow_direction_type):
        """判断yolo_num是否被箭头包含（在箭头方向上被包含）"""
        x1_a, y1_a, x2_a, y2_a = arrow_box
        x1_n, y1_n, x2_n, y2_n = num_box

        if arrow_direction_type == "horizontal":
            # 对于水平箭头，检查yolo_num的x坐标是否在箭头x1,x2内
            # 并且yolo_num在箭头的上下附近（不要求完全在y范围内）
            return (x1_a <= x1_n and x2_n <= x2_a)
        else:  # vertical
            # 对于竖直箭头，检查yolo_num的y坐标是否在箭头y1,y2内
            # 并且yolo_num在箭头的左右附近（不要求完全在x范围内）
            return (y1_a <= y1_n and y2_n <= y2_a)

    def calculate_arrow_length(arrow_box, direction_type):
        """计算箭头长度"""
        if direction_type == "horizontal":
            return arrow_box[2] - arrow_box[0]  # x2 - x1
        else:
            return arrow_box[3] - arrow_box[1]  # y2 - y1

    # 第一步：采用全局最优匹配策略找出未匹配到yolox_num的箭头对
    unmatched_pairs = []
    matched_yolox_indices = set()
    diff_direction_pairs = []  # 记录异方向匹配的箭头对

    # 收集所有可能的匹配对
    all_possible_matches = []

    for pair_idx, pair in enumerate(yolox_pairs):
        pair_box = pair[:4]
        direction = pair[4]  # 0:外向, 1:内向
        pair_direction_type = get_box_direction(pair_box)
        arrow_length = calculate_arrow_length(pair_box, pair_direction_type)

        for yolox_idx, num_box in enumerate(yolox_num):
            num_location = num_box['location']
            num_direction = get_box_direction(num_location)

            # 计算距离
            distance = calculate_box_distance(pair_box, num_location)

            # 判断是否匹配
            is_matched = False
            same_direction = (pair_direction_type == num_direction)

            # 特殊处理：对于长度大于200的外向箭头
            if direction == 0 and arrow_length > 200:
                # 检查yolo_num是否在箭头方向上被包含
                if is_num_contained_by_arrow(pair_box, num_location, pair_direction_type):
                    is_matched = True
                    score = 1000  # 给予很高的分数确保匹配
                    print(f"长外向箭头包含匹配: 箭头 {pair_box} 包含 yolox_num[{yolox_idx}] {num_location}")
                # elif distance <= distance_threshold:
                #     is_matched = True
                #     score = -distance
                #     if same_direction:
                #         score += 10
            else:
                # 原有逻辑：对于其他箭头（内向箭头或短外向箭头）
                if distance <= distance_threshold:
                    is_matched = True
                    score = -distance
                    if same_direction:
                        score += 10

            if is_matched:
                all_possible_matches.append({
                    'pair_idx': pair_idx,
                    'yolox_idx': yolox_idx,
                    'pair_box': pair_box,
                    'direction': direction,
                    'num_location': num_location,
                    'score': score,
                    'distance': distance,
                    'same_direction': same_direction,
                    'arrow_length': arrow_length
                })

    # 按匹配分数降序排序（分数高的优先匹配）
    all_possible_matches.sort(key=lambda x: x['score'], reverse=True)

    # 全局最优匹配
    matched_pairs = set()
    matched_yolox = set()

    for match in all_possible_matches:
        pair_idx = match['pair_idx']
        yolox_idx = match['yolox_idx']

        # 如果箭头对和yolo_num框都未被匹配，则匹配
        if pair_idx not in matched_pairs and yolox_idx not in matched_yolox:
            matched_pairs.add(pair_idx)
            matched_yolox.add(yolox_idx)

            # 如果是异方向匹配，记录下来
            if not match['same_direction']:
                diff_direction_pairs.append((match['pair_box'], match['direction']))
                print(f"异方向匹配: 箭头对 {match['pair_box']} 与 yolox_num 框 {match['num_location']}")

            print(f"全局匹配: 箭头对 {match['pair_box']} 与 yolox_num[{yolox_idx}] {match['num_location']}, 距离: {match['distance']}, 同方向: {match['same_direction']}, 箭头长度: {match['arrow_length']}")

    # 记录未匹配的箭头对
    for idx, pair in enumerate(yolox_pairs):
        if idx not in matched_pairs:
            unmatched_pairs.append((pair[:4], pair[4]))

    print(f"总共找到 {len(unmatched_pairs)} 个未匹配的箭头对")
    print(f"异方向匹配的箭头对数量: {len(diff_direction_pairs)}")

    # 第二步：用未匹配的箭头对去匹配dbnet_data（保持原有逻辑不变）
    new_yolox_num = yolox_num.copy()
    matched_indices = set()

    # 先处理异方向匹配的箭头对
    for pair_box, direction in diff_direction_pairs:
        best_match_idx = -1
        best_overlap = 0

        # 寻找与当前箭头对有高重叠的dbnet_data框
        for idx, dbnet_box in enumerate(dbnet_data):
            if idx in matched_indices:
                continue

            overlap = calculate_overlap(pair_box, dbnet_box)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match_idx = idx

        # 如果找到高重叠的dbnet_data框，则匹配
        if best_match_idx != -1 and best_overlap >= high_overlap_threshold:
            matched_box = dbnet_data[best_match_idx]
            matched_indices.add(best_match_idx)

            # 创建新的尺寸外框字典
            new_num_dict = {
                'location': matched_box,
                'small_boxes': [],
                'ocr_strings': [],
                'Absolutely': ['num']
            }
            new_yolox_num.append(new_num_dict)
            print(f"异方向箭头对 {pair_box} 与dbnet框 {matched_box} 有高重叠 {best_overlap}，优先匹配")

    # 然后处理其他未匹配的箭头对
    for pair_box, direction in unmatched_pairs:
        best_match_idx = -1
        best_score = float('inf')

        for idx, dbnet_box in enumerate(dbnet_data):
            if idx in matched_indices:
                continue

            # 计算重叠比例和距离
            overlap = calculate_overlap(pair_box, dbnet_box)
            distance = calculate_box_distance(pair_box, dbnet_box)

            # 如果有重叠，优先考虑重叠比例
            if overlap > 0:
                score = -overlap
            else:
                score = distance

            if score < best_score:
                best_score = score
                best_match_idx = idx

        # 检查是否满足匹配条件
        if best_match_idx != -1:
            if best_score < 0 or (best_score >= 0 and best_score <= distance_threshold):
                matched_box = dbnet_data[best_match_idx]
                matched_indices.add(best_match_idx)

                new_num_dict = {
                    'location': matched_box,
                    'small_boxes': [],
                    'ocr_strings': [],
                    'Absolutely': ['num']
                }
                new_yolox_num.append(new_num_dict)
                if best_score < 0:
                    print(f"为箭头对 {pair_box} 匹配到dbnet框 {matched_box}, 重叠比例: {-best_score}")
                else:
                    print(f"为箭头对 {pair_box} 匹配到dbnet框 {matched_box}, 距离: {best_score}")

    # 从dbnet_data中剔除已匹配的文本框
    new_dbnet_data = [box for idx, box in enumerate(dbnet_data) if idx not in matched_indices]

    print(f"最终yolox_num数量: {len(new_yolox_num)}, 剩余dbnet_data数量: {len(new_dbnet_data)}")

    return new_dbnet_data, new_yolox_num


def precise_merge_overlapping_boxes(elements, min_overlap_ratio=0.6):
    """
    改进的合并算法：先处理竖向包含的情况，再处理水平重叠的情况
    """

    def should_merge_simple(box1, box2):
        """判断两个框是否应该合并（基于重叠比例）"""
        # 计算交集面积
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return False

        inter_area = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # 较小的框的面积
        min_area = min(area1, area2)

        # 如果较小的框被覆盖的比例超过阈值，就合并
        overlap_ratio = inter_area / min_area
        return overlap_ratio > min_overlap_ratio

    def is_vertical_containment(big_box, small_boxes, min_contained=2):
        """
        判断一个大框是否竖向包含多个小框
        返回：是否属于竖向包含的情况，以及应该保留的小框索引列表
        """
        contained_boxes = []

        for i, small_box in enumerate(small_boxes):
            # 判断小框是否完全在大框内部（允许一定的边界误差）
            # 竖向包含：小框的顶部和底部都在大框内部
            vertical_contain = (small_box[1] >= big_box[1] - 5 and
                                small_box[3] <= big_box[3] + 5)

            # 水平方向有一定重叠
            horizontal_overlap = max(0, min(big_box[2], small_box[2]) - max(big_box[0], small_box[0]))
            width_small = small_box[2] - small_box[0]

            if vertical_contain and horizontal_overlap > width_small * 0.5:
                contained_boxes.append(i)

        return len(contained_boxes) >= min_contained, contained_boxes

    # 第一步：检测并处理竖向包含的情况
    n = len(elements)
    boxes = [elem['location'] for elem in elements]

    # 标记需要排除的大框和需要保留的小框
    exclude_indices = set()
    keep_indices = set(range(n))

    for i in range(n):
        if i in exclude_indices:
            continue

        # 获取其他所有框
        other_boxes = [boxes[j] for j in range(n) if j != i and j not in exclude_indices]
        other_indices = [j for j in range(n) if j != i and j not in exclude_indices]

        # 检查框i是否是一个竖向包含多个小框的大框
        is_containing, contained_indices = is_vertical_containment(
            boxes[i], other_boxes
        )

        if is_containing:
            # 将大框标记为排除
            exclude_indices.add(i)
            # 将被包含的小框从排除列表中移除（确保它们被保留）
            for idx in [other_indices[j] for j in contained_indices]:
                if idx in exclude_indices:
                    exclude_indices.remove(idx)

    # 更新需要处理的元素列表（排除大框）
    elements_to_process = []
    original_to_new_index = {}
    new_index = 0

    for i in range(n):
        if i not in exclude_indices:
            elements_to_process.append(elements[i].copy())
            original_to_new_index[i] = new_index
            new_index += 1

    # 第二步：对剩余的元素进行常规合并
    n_remaining = len(elements_to_process)
    merged = [False] * n_remaining
    result = []

    for i in range(n_remaining):
        if merged[i]:
            continue

        # 开始一个新的组
        group = [i]
        merged[i] = True

        # 尝试合并其他框
        for j in range(i + 1, n_remaining):
            if merged[j]:
                continue

            if should_merge_simple(elements_to_process[i]['location'],
                                   elements_to_process[j]['location']):
                group.append(j)
                merged[j] = True

        # 合并这个组
        if len(group) == 1:
            result.append(elements_to_process[i].copy())
        else:
            # 计算最小外接矩形
            boxes_group = [elements_to_process[idx]['location'] for idx in group]
            x1 = min(box[0] for box in boxes_group)
            y1 = min(box[1] for box in boxes_group)
            x2 = max(box[2] for box in boxes_group)
            y2 = max(box[3] for box in boxes_group)

            # 合并信息
            merged_info = {
                'small_boxes': [],
                'ocr_strings': [],
                'Absolutely': []
            }

            for idx in group:
                elem = elements_to_process[idx]
                merged_info['small_boxes'].extend(elem['small_boxes'])
                merged_info['ocr_strings'].extend(elem['ocr_strings'])
                merged_info['Absolutely'].extend(elem['Absolutely'])

            result.append({
                'location': [x1, y1, x2, y2],
                'small_boxes': merged_info['small_boxes'],
                'ocr_strings': list(set(merged_info['ocr_strings'])),
                'Absolutely': list(set(merged_info['Absolutely']))
            })

    return result