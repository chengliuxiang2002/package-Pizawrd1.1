from package_core.PackageExtract.function_tool import find_list, recite_data
import copy
import numpy as np

def add_direction_field_to_yolox_nums(L3):
    """
    根据border的对角线为每个yolox_num添加方向字段

    参数:
    L3: list of dict, 包含各种检测框信息的字典列表

    返回:
    list: 添加了方向信息到对应视图中的L3列表
    """
    # 遍历所有视图
    for view in ("top", "bottom", "side", "detailed"):
        print(f'{view}方向步骤')
        yolox_num_key = f"{view}_yolox_num"
        border_key = f"{view}_border"

        yolox_nums = find_list(L3, yolox_num_key)
        border = find_list(L3, border_key)
        print(f'调试{yolox_nums}')
        # 确保有数据需要处理
        # 处理numpy数组的情况
        if isinstance(yolox_nums, np.ndarray):
            if yolox_nums.size == 0:
                print("1")
                continue
        elif not yolox_nums:
            print("2")
            continue
        print("*************************&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print("===== 开始打印border =====")
        print(f'调试{view}方向border')
        
        yolox_nums_with_direction = []
        if isinstance(border, np.ndarray):
            if border.size == 0 or len(border) == 0:
                for item in yolox_nums:
                    # 处理元素的"位置信息"：如果是numpy数组则转为列表（保持可序列化），否则保留原始类型
                    if isinstance(item, np.ndarray):
                        location = item.tolist()  # numpy数组转列表
                    else:
                        location = item  # 非数组类型（如列表）直接保留
                    
                    # 为当前元素创建包含"位置"和"方向"的字典
                    item_with_direction = {
                        'location': location,  # 原始位置数据（处理后兼容序列化）
                        'direction': ''  # 方向字段，初始化为空字符串（可后续赋值）
                    }
                    
                    # 添加到新列表中
                    yolox_nums_with_direction.append(item_with_direction)
                direction_key = f"{view}_yolox_nums_direction"
                print(f'方向数据:{yolox_nums_with_direction}')
                recite_data(L3, direction_key, yolox_nums_with_direction)
                continue
        elif not border or len(border) == 0:
            for item in yolox_nums:
                # 处理元素的"位置信息"：如果是numpy数组则转为列表（保持可序列化），否则保留原始类型
                if isinstance(item, np.ndarray):
                    location = item.tolist()  # numpy数组转列表
                else:
                    location = item  # 非数组类型（如列表）直接保留
                
                # 为当前元素创建包含"位置"和"方向"的字典
                item_with_direction = {
                    'location': location,  # 原始位置数据（处理后兼容序列化）
                    'direction': ''  # 方向字段，初始化为空字符串（可后续赋值）
                }
                
                # 添加到新列表中
                yolox_nums_with_direction.append(item_with_direction)
            direction_key = f"{view}_yolox_nums_direction"
            print(f'方向数据:{yolox_nums_with_direction}')
            recite_data(L3, direction_key, yolox_nums_with_direction)
            continue

        # 提取border坐标
        if isinstance(border, np.ndarray):
            x1, y1, x2, y2 = border[0]
        else:
            x1, y1, x2, y2 = border[0]

        # 计算border的中心点
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 计算两条对角线的斜率和截距
        # 对角线1: 从(x1,y1)到(x2,y2)
        slope1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
        intercept1 = y1 - slope1 * x1 if slope1 != float('inf') else None

        # 对角线2: 从(x1,y2)到(x2,y1)
        slope2 = (y1 - y2) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
        intercept2 = y2 - slope2 * x1 if slope2 != float('inf') else None

        # 创建带方向信息的yolox_nums副本（保持原始数据结构不变）
        if isinstance(yolox_nums, np.ndarray):
            # 如果是numpy数组，创建包含方向信息的新结构
            yolox_nums_with_direction = []
            for item in yolox_nums:
                yolox_nums_with_direction.append({
                    'location': item.tolist() if isinstance(item, np.ndarray) else item,
                    'direction': ''  # 初始化为空字符串
                })
        else:
            # 如果是字典列表，创建带方向信息的副本
            yolox_nums_with_direction = copy.deepcopy(yolox_nums)
            # 确保每个字典都有direction字段
            for item in yolox_nums_with_direction:
                if 'direction' not in item:
                    item['direction'] = ''

        # 为每个yolox_num添加方向字段
        for yolox_num in yolox_nums_with_direction:
            # 提取yolox_num的位置信息
            loc = yolox_num['location']
            if isinstance(loc, np.ndarray):
                loc = loc.tolist()
                
            if len(loc) == 4:  # [x1, y1, x2, y2]格式
                obj_center_x = (loc[0] + loc[2]) / 2
                obj_center_y = (loc[1] + loc[3]) / 2
            else:
                # 假设是其他格式，使用前两个值作为中心点
                obj_center_x = loc[0]
                obj_center_y = loc[1]

            # 判断对象相对于两条对角线的位置
            if slope1 != float('inf') and slope2 != float('inf'):
                # 计算对象中心点到两条对角线的有向距离
                # 对于对角线1
                dist1 = (slope1 * obj_center_x - obj_center_y + intercept1) / ((slope1 ** 2 + 1) ** 0.5)

                # 对于对角线2
                dist2 = (slope2 * obj_center_x - obj_center_y + intercept2) / ((slope2 ** 2 + 1) ** 0.5)

                # 根据有向距离判断区域
                if dist1 > 0 and dist2 > 0:
                    direction = 'up'
                elif dist1 > 0 and dist2 < 0:
                    direction = 'right'
                elif dist1 < 0 and dist2 > 0:
                    direction = 'left'
                else:  # dist1 < 0 and dist2 < 0
                    direction = 'down'
            else:
                # 处理垂直或水平border的特殊情况
                if x1 == x2:  # 垂直border
                    if obj_center_x < center_x:
                        direction = 'right'
                    else:
                        direction = 'left'
                else:  # 水平border
                    if obj_center_y < center_y:
                        direction = 'up'
                    else:
                        direction = 'down'

            # 添加方向字段到yolox_num字典
            yolox_num['direction'] = direction

        # 将带有方向信息的数据添加到L3中，按照规范命名为[视图名]_yolox_nums_direction
        direction_key = f"{view}_yolox_nums_direction"
        print(f'方向数据:{yolox_nums_with_direction}')
        recite_data(L3, direction_key, yolox_nums_with_direction)

    return L3