from package_core.PackageExtract.function_tool import find_list,recite_data



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

    for view in ("top", "bottom", "side", "detailed"):
        dbnet_key = f"{view}_dbnet_data"
        yolox_num_key = f"{view}_yolox_num"
        yolox_pairs_key = f"{view}_yolox_pairs"
        dbnet_data = find_list(L3, dbnet_key)
        yolox_num = find_list(L3, yolox_num_key)
        yolox_pairs = find_list(L3, yolox_pairs_key)
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
        # recite_data(L3, dbnet_key, new_dbnet_data)
        # recite_data(L3, yolox_num_key, new_yolox_num)  # 修复：使用正确的键名
        recite_data(L3, f"{view}_num_match_results", new_yolox_num_supply)

    return L3



def match_arrows_to_dbnet(yolox_pairs, dbnet_data, yolox_num, distance_threshold=40, overlap_threshold=0.1, high_overlap_threshold=0.3):
    """
    将未匹配的yolox_pairs箭头对与剩余的dbnet数据进行匹配
    
    Args:
        yolox_pairs: list of [x1, y1, x2, y2, direction] - 箭头对坐标和方向(1:内向, 0:外向)
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
        
        return (dx**2 + dy**2)**0.5
    
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
    
    # 第一步：采用全局最优匹配策略找出未匹配到yolox_num的箭头对
    unmatched_pairs = []
    matched_yolox_indices = set()
    diff_direction_pairs = []  # 记录异方向匹配的箭头对
    
    # 收集所有可能的匹配对
    all_possible_matches = []
    
    for pair_idx, pair in enumerate(yolox_pairs):
        pair_box = pair[:4]
        direction = pair[4]
        pair_direction = get_box_direction(pair_box)
        
        for yolox_idx, num_box in enumerate(yolox_num):
            num_location = num_box['location']
            num_direction = get_box_direction(num_location)
            
            # 计算距离
            distance = calculate_box_distance(pair_box, num_location)
            
            if distance <= distance_threshold:
                # 检查方向是否一致
                same_direction = (pair_direction == num_direction)
                
                # 计算匹配分数（距离越小分数越高，同方向加分）
                score = -distance  # 距离越小分数越高
                if same_direction:
                    score += 10  # 同方向匹配加分
                
                all_possible_matches.append({
                    'pair_idx': pair_idx,
                    'yolox_idx': yolox_idx,
                    'pair_box': pair_box,
                    'direction': direction,
                    'num_location': num_location,
                    'score': score,
                    'distance': distance,
                    'same_direction': same_direction
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
            
            print(f"全局匹配: 箭头对 {match['pair_box']} 与 yolox_num[{yolox_idx}] {match['num_location']}, 距离: {match['distance']}, 同方向: {match['same_direction']}")
    
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