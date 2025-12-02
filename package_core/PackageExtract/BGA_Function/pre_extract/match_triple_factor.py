import numpy as np
import copy
import json
import os
from scipy.spatial.distance import cdist
from package_core.PackageExtract.function_tool import find_list, recite_data

def match_arrow_pairs_with_yolox(L3, image_root):
    """
    将pairs_length中的箭头框与yolox_num_direction中的位置进行匹配

    参数:
    pairs_length: numpy数组，维度(n,13)，每行包含:
                 [pairs_x1, pairs_y1, pairs_x2, pairs_y2,
                  line1_x1, line1_y1, line1_x2, line1_y2,
                  line2_x1, line2_y1, line2_x2, line2_y2,
                  distance]
    yolox_num_direction: list of dict，每个字典包含:
                'location': [x1, y1, x2, y2] 或类似格式,
                'small_boxes': [],
                'ocr_strings': [],
                'Absolutely': ['num'],
                'direction': str
    返回:
    list: 新的数据结构列表，每个元素包含原始yolox_num_direction信息和匹配的arrow_pairs
    """
    all_views_results = []
    print("开始执行循环")  # 先打印这行
    for view in ("top", "bottom", "side", "detailed"):
        pairs_length = find_list(L3, f"{view}_yolox_pairs_length")
        yolox_nums_direction = find_list(L3, f"{view}_yolox_nums_direction")
        img_path = os.path.join(image_root, f"{view}.jpg")
        print(f'{view}方向准备3合一')
        print(f'箭头数据:{pairs_length}')
        print(f'尺寸数据:{yolox_nums_direction}')
        # 创建yolox_num_direction的副本，避免修改原始数据
        matched_yolox = [item.copy() for item in yolox_nums_direction]

        # 初始化所有yolox_num的arrow_pairs字段为None
        for item in matched_yolox:
            item['arrow_pairs'] = None

        # 如果pairs_length为空，直接返回
        if len(pairs_length) == 0:
            continue

        # 提取pairs_length中的箭头框中心点
        arrow_centers = []
        for pair in pairs_length:
            x1, y1, x2, y2 = pair[0:4]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            arrow_centers.append([center_x, center_y])

        arrow_centers = np.array(arrow_centers)

        # 提取yolox_num_direction中的位置中心点
        yolox_centers = []
        for item in matched_yolox:
            loc = item['location']
            if len(loc) == 4:  # [x1, y1, x2, y2]格式
                x1, y1, x2, y2 = loc
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
            else:
                center_x, center_y = loc[0], loc[1]
            yolox_centers.append([center_x, center_y])

        yolox_centers = np.array(yolox_centers)

        if yolox_centers.size == 0 or arrow_centers.size == 0:
            pass  # 跳过匹配，直接进入后续的处理步骤
        else:
            # 计算所有点对之间的距离矩阵
            distance_matrix = cdist(yolox_centers, arrow_centers)

            # 创建标记数组，记录哪些yolox和arrow已经被匹配
            yolox_matched = [False] * len(matched_yolox)
            arrow_matched = [False] * len(pairs_length)

            # 按照距离从小到大进行匹配
            while True:
                # 找到最小的未匹配距离
                min_distance = float('inf')
                min_i, min_j = -1, -1

                for i in range(len(matched_yolox)):
                    if yolox_matched[i]:
                        continue
                    for j in range(len(pairs_length)):
                        if arrow_matched[j]:
                            continue
                        if distance_matrix[i, j] < min_distance:
                            min_distance = distance_matrix[i, j]
                            min_i, min_j = i, j

                # 如果没有找到可匹配的对，退出循环
                if min_i == -1 or min_j == -1:
                    break

                # 标记为已匹配
                yolox_matched[min_i] = True
                arrow_matched[min_j] = True

                # 将匹配的arrow_pairs添加到yolox_num中
                matched_yolox[min_i]['arrow_pairs'] = pairs_length[min_j]

        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # 为每个元素添加 view_name 和 parameters 字段，parameters 先为空
        processed_list = []
        for item in matched_yolox:
            new_item = {
                'view_name': img_name,
                'parameters': None,  # parameters 先为空
                **item  # 将原来的所有字段展开到外层
            }
            processed_list.append(new_item)

        all_views_results.append(processed_list)

    return all_views_results