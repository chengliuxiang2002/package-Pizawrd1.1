import numpy as np
import copy
import json
import os
from scipy.spatial.distance import cdist
from package_core.PackageExtract.function_tool import find_list, recite_data


def visualize_matched_pairs(matched_yolox, image_path, output_path=None):
    """
    可视化匹配结果

    参数:
    matched_yolox: 匹配后的yolox_num_direction列表
    image_path: 背景图像路径
    output_path: 输出图像路径
    """
    import cv2
    import os
    import numpy as np

    # 读取背景图像
    if not os.path.exists(image_path):
        print(f"错误: 图像路径不存在: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法加载图像: {image_path}")
        return

    # 创建图像副本
    img = image.copy()

    # 定义颜色
    yolox_color = (0, 255, 100)  # 黄色 - yolox框
    arrow_color = (0, 0, 255)  # 红色 - 箭头框
    line_color = (0, 255, 100)  # 绿色 - 连接线
    line1_color = (0, 165, 255)  # 蓝色 - 引线1
    line2_color = (0, 165, 255)  # 橙色 - 引线2
    text_color = (0, 0, 200)  # 黑色 - 文本
    max_medium_min_color = (255, 0, 0)  # 蓝色 - max_medium_min文本
    key_info_color = (255, 0, 255)  # 紫色 - key_info文本
    ocr_color = (0, 100, 255)  # 橙色 - OCR文本

    # 遍历所有匹配的yolox_num
    for i, item in enumerate(matched_yolox):
        # 提取yolox位置
        loc = item['location']
        if len(loc) == 4:
            x1, y1, x2, y2 = [int(coord) for coord in loc]
        else:
            # 假设是其他格式
            x1, y1 = int(loc[0]), int(loc[1])
            x2, y2 = x1 + 10, y1 + 10

        # 绘制yolox框
        cv2.rectangle(img, (x1, y1), (x2, y2), yolox_color, 2)

        # 添加方向文本
        direction = item.get('direction', 'unknown')
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.putText(img, f"Dir: {direction}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, yolox_color, 1, cv2.LINE_AA)

        # # 添加view_name文本
        # view_name = item.get('view_name', '')
        # if view_name:
        #     cv2.putText(img, f"View: {view_name}", (x1, y1 - 25),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        # 获取parameters中的信息 - 添加None检查
        parameters = item.get('parameters', {})
        if parameters is None:
            parameters = {}

        # 添加max_medium_min文本
        max_medium_min = parameters.get('max_medium_min', [])
        if max_medium_min is not None and len(max_medium_min) == 3:
            # 在yolox框下方显示max_medium_min值
            max_val, medium_val, min_val = max_medium_min
            text_y = y2 + 15
            # cv2.putText(img, f"Max: {max_val:.3f}", (x1, text_y),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, max_medium_min_color, 1, cv2.LINE_AA)
            cv2.putText(img, f"Med: {medium_val:.3f}", (x1, text_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, max_medium_min_color, 1, cv2.LINE_AA)
            # cv2.putText(img, f"Min: {min_val:.3f}", (x1, text_y + 30),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, max_medium_min_color, 1, cv2.LINE_AA)

        # # 添加key_info文本
        # key_info = parameters.get('key_info', [])
        # if key_info is not None and key_info:
        #     # 在max_medium_min下方显示key_info
        #     key_info_y = y2 + 60
        #     for j, (key, value) in enumerate(key_info):
        #         cv2.putText(img, f"{key}: {value}", (x1, key_info_y + j * 15),
        #                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, key_info_color, 1, cv2.LINE_AA)

        # # 添加ocr_strings文本
        # ocr_strings = parameters.get('ocr_strings', '')
        # if ocr_strings is not None and ocr_strings:
        #     # 在key_info下方显示ocr_strings
        #     ocr_y = y2 + 60 + (len(key_info) if key_info else 0) * 15 + 15
        #     cv2.putText(img, f"OCR: {ocr_strings}", (x1, ocr_y),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.4, ocr_color, 1, cv2.LINE_AA)

        # # 添加Absolutely信息
        # absolutely = item.get('Absolutely', [])
        # if absolutely is not None and absolutely:
        #     # 计算ocr_strings的位置
        #     ocr_y_offset = 0
        #     if ocr_strings is not None and ocr_strings:
        #         ocr_y_offset = 15
        #     key_info_count = len(key_info) if key_info else 0
        #     abs_y = y2 + 60 + key_info_count * 15 + 15 + ocr_y_offset
        #     cv2.putText(img, f"Abs: {absolutely}", (x1, abs_y),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        # # 绘制small_boxes
        # small_boxes = item.get('small_boxes', [])
        # if small_boxes is not None:
        #     for j, small_box in enumerate(small_boxes):
        #         if len(small_box) == 4:
        #             sx1, sy1, sx2, sy2 = [int(coord) for coord in small_box]
        #             cv2.rectangle(img, (sx1, sy1), (sx2, sy2), (255, 255, 0), 1)

        #             # 显示small_box的索引
        #             cv2.putText(img, str(j), (sx1, sy1 - 5),
        #                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1, cv2.LINE_AA)

        # 如果有匹配的arrow_pairs，绘制箭头框和连接线
        arrow_pairs = item.get('arrow_pairs')
        if arrow_pairs is not None:
            arrow_data = arrow_pairs

            # 检查arrow_data是否有足够的元素
            if len(arrow_data) >= 13:
                # 提取箭头框坐标
                arrow_x1, arrow_y1, arrow_x2, arrow_y2 = [int(coord) for coord in arrow_data[0:4]]

                # 绘制箭头框
                cv2.rectangle(img, (arrow_x1, arrow_y1), (arrow_x2, arrow_y2), arrow_color, 2)

                # 绘制连接线（从yolox中心到箭头框中心）
                arrow_center_x = (arrow_x1 + arrow_x2) // 2
                arrow_center_y = (arrow_y1 + arrow_y2) // 2
                cv2.line(img, (center_x, center_y), (arrow_center_x, arrow_center_y), line_color, 1)

                # 绘制引线1
                line1_x1, line1_y1, line1_x2, line1_y2 = [int(coord) for coord in arrow_data[4:8]]
                cv2.line(img, (line1_x1, line1_y1), (line1_x2, line1_y2), line1_color, 2)

                # 绘制引线2
                line2_x1, line2_y1, line2_x2, line2_y2 = [int(coord) for coord in arrow_data[8:12]]
                cv2.line(img, (line2_x1, line2_y1), (line2_x2, line2_y2), line2_color, 2)

                # 在引线1和引线2上添加端点标记
                cv2.circle(img, (line1_x1, line1_y1), 3, line1_color, -1)
                cv2.circle(img, (line1_x2, line1_y2), 3, line1_color, -1)
                cv2.circle(img, (line2_x1, line2_y1), 3, line2_color, -1)
                cv2.circle(img, (line2_x2, line2_y2), 3, line2_color, -1)

                # 在箭头框中心显示距离值
                distance = arrow_data[12]
                cv2.putText(img, f"{distance:.2f}", (arrow_center_x, arrow_center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

                # 添加匹配索引
                cv2.putText(img, str(i), (arrow_x1, arrow_y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, arrow_color, 1, cv2.LINE_AA)

    # 添加图例
    legend_y = 30
    cv2.putText(img, "Yellow: YOLOX Box", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, yolox_color, 1, cv2.LINE_AA)
    cv2.putText(img, "Red: Arrow Box", (10, legend_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, arrow_color, 1, cv2.LINE_AA)
    cv2.putText(img, "Green: Connection Line", (10, legend_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1, cv2.LINE_AA)
    cv2.putText(img, "Blue: Line 1", (10, legend_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, line1_color, 1, cv2.LINE_AA)
    cv2.putText(img, "Orange: Line 2", (10, legend_y + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, line2_color, 1, cv2.LINE_AA)
    cv2.putText(img, "Blue Text: max_medium_min", (10, legend_y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, max_medium_min_color, 1, cv2.LINE_AA)
    cv2.putText(img, "Purple Text: key_info", (10, legend_y + 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, key_info_color, 1, cv2.LINE_AA)
    cv2.putText(img, "Orange Text: OCR strings", (10, legend_y + 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ocr_color, 1, cv2.LINE_AA)

    # 保存或显示图像
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"匹配结果可视化已保存到: {output_path}")
    else:
        # 自动生成输出路径 - 使用原图名+"_matched.png"
        base_name = os.path.basename(image_path)
        name_without_ext = os.path.splitext(base_name)[0]

        # 获取原图目录
        original_dir = os.path.dirname(image_path)

        # 生成输出路径
        default_output_path = os.path.join(original_dir, f"{name_without_ext}_matched.png")

        cv2.imwrite(default_output_path, img)
        print(f"匹配结果可视化已保存到: {default_output_path}")

    return img


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
            
        img_name_without_ext = os.path.splitext(img_name)[0]
        # 2. 拼接根目录和处理后的图片名（自动处理路径分隔符）
        full_path = os.path.join(image_root, f"{img_name_without_ext}.jpg") 
        # print(f'processed_list:{processed_list}')   
        visualize_matched_pairs(processed_list,full_path)
        all_views_results.append(processed_list)

    return all_views_results