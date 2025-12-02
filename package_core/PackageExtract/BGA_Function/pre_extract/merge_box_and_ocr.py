from typing import List, Dict, Any
from package_core.PackageExtract.get_pairs_data_present5_test import ocr_data
def is_horizontal_overlap(box1: List[int], box2: List[int], 
                         overlap_ratio: float = 0.3, 
                         max_y_diff: float = 10) -> bool:
    """
    判断两个框是否水平方向排列且有水平方向重合
    
    Args:
        box1: 第一个框坐标 [x1, y1, x2, y2]
        box2: 第二个框坐标 [x1, y1, x2, y2]
        overlap_ratio: 水平重叠比例阈值
        max_y_diff: 最大y坐标中心点差异阈值
    
    Returns:
        bool: 是否水平排列且有水平方向重合
    """
    # 提取坐标
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 1. 计算y坐标中心点，判断是否在同一水平线上
    y_center1 = (y1_1 + y2_1) / 2
    y_center2 = (y1_2 + y2_2) / 2
    y_diff = abs(y_center1 - y_center2)
    
    # 2. 计算水平方向的重合度
    x_overlap = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
    x_union = max(x2_1, x2_2) - min(x1_1, x1_2)
    horizontal_overlap_ratio = x_overlap / x_union if x_union > 0 else 0
    
    # 3. 判断条件：y坐标相近且水平方向有重合
    return y_diff <= max_y_diff and horizontal_overlap_ratio > overlap_ratio

def merge_horizontal_boxes(boxes: List[List[int]]) -> List[int]:
    """
    合并水平方向排列的多个小框为一个大框
    
    Args:
        boxes: 小框坐标列表
    
    Returns:
        List[int]: 合并后的大框坐标 [x1, y1, x2, y2]
    """
    if not boxes:
        return []
    
    # 找到所有框的边界
    all_x1 = [box[0] for box in boxes]
    all_y1 = [box[1] for box in boxes]
    all_x2 = [box[2] for box in boxes]
    all_y2 = [box[3] for box in boxes]
    
    # 计算合并后的大框坐标
    merged_box = [
        min(all_x1),  # x1
        min(all_y1),  # y1
        max(all_x2),  # x2
        max(all_y2)   # y2
    ]
    
    return merged_box

def find_horizontal_groups(boxes: List[List[int]]) -> List[List[List[int]]]:
    """
    找出水平方向排列且有水平方向重合的框组
    
    Args:
        boxes: 所有小框坐标
    
    Returns:
        List: 水平框组列表，每个组内是应该合并的框
    """
    if len(boxes) <= 1:
        return []
    
    # 按x坐标排序
    sorted_boxes = sorted(boxes, key=lambda box: box[0])
    
    groups = []
    current_group = [sorted_boxes[0]]
    
    for i in range(1, len(sorted_boxes)):
        current_box = sorted_boxes[i]
        last_box_in_group = current_group[-1]
        
        # 判断当前框是否与组内最后一个框水平方向有重合
        if is_horizontal_overlap(last_box_in_group, current_box):
            current_group.append(current_box)
        else:
            # 当前组结束，开始新组（只保留有2个以上框的组）
            if len(current_group) > 1:
                groups.append(current_group)
            current_group = [current_box]
    
    # 处理最后一组
    if len(current_group) > 1:
        groups.append(current_group)
    
    return groups

def process_ocr_recognition(image_path: str, data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    处理OCR识别的完整函数
    
    Args:
        image_path: 图像路径
        data_list: 包含位置信息的数据列表
    
    Returns:
        List[Dict]: 处理后的数据列表，包含识别结果
    """
    processed_data = []
    
    for item in data_list:
        # 深拷贝原始数据
        processed_item = item.copy()
        
        # 情况A：只有一个框（无small_boxes或small_boxes为空）
        if not item.get('small_boxes'):
            # 直接将location坐标送入OCR函数
            ocr_result = ocr_data(image_path, [item['location']])
            processed_item['ocr_strings'] = ocr_result[0] if ocr_result else ""
        
        # 情况B：有小框
        else:
            small_boxes = item['small_boxes']
            
            # 找出水平排列且有水平重合的框组
            horizontal_groups = find_horizontal_groups(small_boxes)
            
            # 获取所有需要单独识别的小框（不在水平组中的框）
            all_boxes_in_groups = [box for group in horizontal_groups for box in group]
            single_boxes = [box for box in small_boxes if box not in all_boxes_in_groups]
            
            recognition_results = []
            
            # 处理水平组：合并后识别
            for group in horizontal_groups:
                if len(group) > 1:
                    merged_box = merge_horizontal_boxes(group)
                    ocr_result = ocr_data(image_path, [merged_box])
                    recognition_results.append(ocr_result[0] if ocr_result else "")
                else:
                    # 单个框直接识别（理论上不会出现这种情况）
                    ocr_result = ocr_data(image_path, group)
                    recognition_results.append(ocr_result[0] if ocr_result else "")
            
            # 处理单独的小框：逐个识别
            for single_box in single_boxes:
                ocr_result = ocr_data(image_path, [single_box])
                recognition_results.append(ocr_result[0] if ocr_result else "")
            
            processed_item['ocr_strings'] = recognition_results
        
        processed_data.append(processed_item)
    
    return processed_data

def process_all_variables(image_path: str, 
                         new_other: List[Dict],
                         new_yolox_serial_num: List[Dict], 
                         angle_boxes_dicts: List[Dict],
                         new_yolox_num: List[Dict],
                         BGA_serial_num: List[Dict] = None,
                         BGA_serial_letter: List[Dict] = None) -> Dict[str, List[Dict]]:
    # """
    # 处理所有变量的主函数
    
    # Args:
    #     image_path: 图像路径
    #     new_other, new_yolox_serial_num, angle_boxes_dicts, new_yolox_num: 各个变量数据
    
    # Returns:
    #     Dict: 包含所有处理结果的字典
    # """
    # # 处理各个变量
    # processed_new_other = process_ocr_recognition(image_path, new_other)
    # processed_new_yolox_serial_num = process_ocr_recognition(image_path, new_yolox_serial_num)
    # processed_angle_boxes_dicts = process_ocr_recognition(image_path, angle_boxes_dicts)
    # processed_new_yolox_num = process_ocr_recognition(image_path, new_yolox_num)
    
    # return {
    #     'new_other': processed_new_other,
    #     'new_yolox_serial_num': processed_new_yolox_serial_num,
    #     'angle_boxes_dicts': processed_angle_boxes_dicts,
    #     'new_yolox_num': processed_new_yolox_num
    # }
    """
    处理所有变量的主函数，包括BGA特有类型
    
    Args:
        image_path: 图像路径
        new_other, new_yolox_serial_num, angle_boxes_dicts, new_yolox_num: 各个变量数据
        BGA_serial_num: BGA序列号数字数据 (BGA特有)
        BGA_serial_letter: BGA序列号字母数据 (BGA特有)
    
    Returns:
        Dict: 包含所有处理结果的字典
    """
    # 处理各个变量
    processed_new_other = process_ocr_recognition(image_path, new_other)
    processed_new_yolox_serial_num = process_ocr_recognition(image_path, new_yolox_serial_num)
    processed_angle_boxes_dicts = process_ocr_recognition(image_path, angle_boxes_dicts)
    processed_new_yolox_num = process_ocr_recognition(image_path, new_yolox_num)
    
    # 初始化结果字典
    results = {
        'new_other': processed_new_other,
        'new_yolox_serial_num': processed_new_yolox_serial_num,
        'angle_boxes_dicts': processed_angle_boxes_dicts,
        'new_yolox_num': processed_new_yolox_num
    }
    
    # 处理BGA特有类型
    if BGA_serial_num is not None:
        results['BGA_serial_num'] = process_ocr_recognition(image_path, BGA_serial_num)
    
    if BGA_serial_letter is not None:
        results['BGA_serial_letter'] = process_ocr_recognition(image_path, BGA_serial_letter)
    
    return results