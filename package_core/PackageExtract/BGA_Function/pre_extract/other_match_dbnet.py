import numpy as np

from package_core.PackageExtract.function_tool import find_list,recite_data


def other_match_boxes_by_overlap(L3):
    """
    使用重叠面积比例进行匹配
    
    Args:
        dbnet_data: 小框列表
        other: 大框列表
        overlap_ratio: 重叠面积占小框面积的比例阈值
    """
    overlap_ratio = 0.5
    def calculate_overlap_ratio(small_box, large_box):
        """计算小框与大框的重叠面积占小框面积的比例"""
        # 计算交集区域
        x1 = max(small_box[0], large_box[0])
        y1 = max(small_box[1], large_box[1])
        x2 = min(small_box[2], large_box[2])
        y2 = min(small_box[3], large_box[3])
        
        # 计算交集面积
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # 计算小框面积
        small_area = (small_box[2] - small_box[0]) * (small_box[3] - small_box[1])
        
        return intersection / small_area if small_area > 0 else 0
    for view in ("top", "bottom", "side", "detailed"):
        dbnet_key = f"{view}_dbnet_data"
        other_key = f"{view}_other"

        dbnet_data = find_list(L3, dbnet_key)
        other_data = find_list(L3, other_key)
        print(f'dbnet数据:{dbnet_data}')
        print(f'other数据:{other_data}')
        dbnet_data = dbnet_data.tolist() if isinstance(dbnet_data, np.ndarray) else dbnet_data
        new_dbnet_data = dbnet_data.copy()
        new_other = []

        for large_box in other_data:
            matched_small_boxes = []

            for small_box in dbnet_data:
                ratio = calculate_overlap_ratio(small_box, large_box)
                if ratio >= overlap_ratio:
                    matched_small_boxes.append(small_box)

            if matched_small_boxes:
                for small_box in matched_small_boxes:
                    if small_box in new_dbnet_data:
                        new_dbnet_data.remove(small_box)

                result_dict = {
                    'location': large_box,
                    'small_boxes': matched_small_boxes,
                    'ocr_strings': [],
                    'Absolutely': ['other']
                }
                new_other.append(result_dict)
        new_dbnet_data = np.array(new_dbnet_data)

        recite_data(L3, dbnet_key, new_dbnet_data)
        recite_data(L3, other_key, new_other)
    return L3