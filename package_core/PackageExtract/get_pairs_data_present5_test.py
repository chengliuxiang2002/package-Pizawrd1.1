from __future__ import annotations
import math
from math import sqrt
import cv2
import numpy as np
import shutil
import time
import os
import operator
import re
from random import randint
import copy
from concurrent.futures import ThreadPoolExecutor
from package_core.PackageExtract.yolox_onnx_py.onnx_QFP_pairs_data_location2 import begain_output_pairs_data_location
# from yolox_onnx_py.onnx_output_other_location import begain_output_other_location
from package_core.PackageExtract.yolox_onnx_py.onnx_output_serial_number_letter_location import begain_output_serial_number_letter_location
from package_core.PackageExtract.yolox_onnx_py.onnx_output_side_body_standoff_location import begain_output_side_body_standoff_location
from package_core.PackageExtract.yolox_onnx_py.onnx_output_pin_yinXian_find_pitch import begain_output_pin_location
from package_core.PackageExtract.yolox_onnx_py.onnx_output_top_body_location import begain_output_top_body_location
from package_core.PackageExtract.yolox_onnx_py.onnx_output_bottom_body_location import begain_output_bottom_body_location

from package_core.PackageExtract.onnx_use import Run_onnx_det
from package_core.PackageExtract.onnx_use import Run_onnx
from pathlib import Path

log_path = Path("test_data.txt")
# 把 print 的输出复制一份到文件
def tee_print(*args, sep=" ", end="\n"):
    msg = sep.join(map(str, args)) + end
    print(*args, sep=sep, end=end)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(msg)

# 导入统一路径管理
try:
    from package_core.PackageExtract.yolox_onnx_py.model_paths import result_path
except ModuleNotFoundError:
    from pathlib import Path
    def result_path(*parts):
        return str(Path(__file__).resolve().parents[2] / 'Result' / Path(*parts))

# 全局路径 - 使用统一的路径管理函数
DATA = result_path('Package_extract', 'data')
DATA_BOTTOM_CROP = result_path('Package_extract', 'data_bottom_crop')
DATA_COPY = result_path('Package_extract', 'data_copy')
ONNX_OUTPUT = result_path('Package_extract', 'onnx_output')
OPENCV_OUTPUT = result_path('Package_extract', 'opencv_output')
OPENCV_OUTPUT_LINE = result_path('Package_extract', 'opencv_output_yinXian')
YOLO_DATA = result_path('Package_extract', 'yolox_data')

def delete_other(other, data):
    '''
    other:np.(,4)[x1,y1,x2,y2]
    data:np.(,4)[x1,y1,x2,y2]
    '''
    # 将other和光洁度框线缩小防止误删
    ratio = 0.5
    ratio = ratio * 0.5
    for i in range(len(other)):
        other[i][0] = other[i][0] + ratio * abs(other[i][0] - other[i][2])
        other[i][1] = other[i][1] + ratio * abs(other[i][1] - other[i][3])
        other[i][2] = other[i][2] - ratio * abs(other[i][0] - other[i][2])
        other[i][3] = other[i][3] - ratio * abs(other[i][1] - other[i][3])

    # 当外框重叠且重叠部分在两方中至少一方占面积比较大，即可筛除
    data_count = np.zeros((len(data)))  # 1=将要被筛出
    ratio = 0.5
    for i in range(len(data)):
        for j in range(len(other)):
            if not (data[i][0] > other[j][2] or data[i][2] < other[j][0]):  # 两矩形在x坐标上的长有重叠
                if not (data[i][1] > other[j][3] or data[i][3] < other[j][1]):  # 两矩形在y坐标上的高有重叠
                    l = data[i][2] - data[i][0] + other[j][2] - other[j][0] - max(abs(data[i][0] - other[j][2]),
                                                                                  abs(data[i][2] - other[j][0]))
                    w = data[i][3] - data[i][1] + other[j][3] - other[j][1] - max(abs(data[i][1] - other[j][3]),
                                                                                  abs(data[i][3] - other[j][1]))
                    if l * w / (data[i][2] - data[i][0]) * (data[i][3] - data[i][1]) > ratio or l * w / (
                            other[j][2] - other[j][0]) * (other[j][3] - other[j][1]) > ratio:
                        data_count[i] = 1

    new_data = np.zeros((0, 4))
    for i in range(len(data_count)):
        if data_count[i] == 0:
            new_data = np.r_[new_data, [data[i]]]
    return new_data

##############################
import numpy as np

def remove_duplicate_pairs(pairs_length):
    """
    去除使用相同引线组合的重复箭头对，只保留引线到箭头对框距离最近的那个
    
    参数:
    pairs_length: np.二维数组 (n, 13) [pairs_x1_y1_x2_y2, 引线1_x1_y1_x2_y2, 引线2_x1_y1_x2_y2, 两引线距离]
    
    返回:
    去重后的 pairs_length
    """
    print("开始检查重复引线...")
    
    # 创建一个字典来存储引线组合和对应的箭头对信息
    line_pairs_dict = {}
    
    for i in range(len(pairs_length)):
        # 提取两条引线的坐标，作为唯一标识
        line1_key = tuple(pairs_length[i][4:8].astype(int))  # 引线1的坐标
        line2_key = tuple(pairs_length[i][8:12].astype(int)) # 引线2的坐标
        
        # 创建引线组合的唯一标识符（不考虑顺序）
        line_pair_key = tuple(sorted([line1_key, line2_key]))
        
        # 计算引线到箭头对框的距离
        arrow_box = [pairs_length[i][0], pairs_length[i][1], pairs_length[i][2], pairs_length[i][3]]
        
        # 对于每条引线，计算到箭头对框的最小距离
        line1_distance = calculate_line_to_box_distance(
            [pairs_length[i][4], pairs_length[i][5], pairs_length[i][6], pairs_length[i][7]], 
            arrow_box
        )
        
        line2_distance = calculate_line_to_box_distance(
            [pairs_length[i][8], pairs_length[i][9], pairs_length[i][10], pairs_length[i][11]], 
            arrow_box
        )
        
        # 使用两条引线到箭头对框的平均距离作为总距离
        distance = (line1_distance + line2_distance) / 2
        
        # 如果这个引线组合已经存在，比较距离
        if line_pair_key in line_pairs_dict:
            existing_distance, existing_index = line_pairs_dict[line_pair_key]
            if distance < existing_distance:
                # 当前箭头对距离更近，更新字典
                line_pairs_dict[line_pair_key] = (distance, i)
                print(f"发现重复引线组合，更新为距离更近的箭头对，距离: {distance:.2f}")
            else:
                print(f"发现重复引线组合，保留原有箭头对，距离: {existing_distance:.2f}")
        else:
            # 新的引线组合，添加到字典
            line_pairs_dict[line_pair_key] = (distance, i)
    
    # 构建去重后的结果
    if len(line_pairs_dict) < len(pairs_length):
        print(f"去重前: {len(pairs_length)} 个箭头对，去重后: {len(line_pairs_dict)} 个箭头对")
        
        unique_indices = [item[1] for item in line_pairs_dict.values()]
        pairs_length = pairs_length[unique_indices]
    else:
        print("没有发现重复的引线组合")
    
    return pairs_length


def calculate_line_to_box_distance(line, box):
    """
    计算直线到矩形框的最小距离
    line: [x1, y1, x2, y2] 直线端点坐标
    box: [x1, y1, x2, y2] 矩形框左上角和右下角坐标
    返回: 直线到矩形框的最小距离
    """
    # 提取直线端点
    line_p1 = (line[0], line[1])
    line_p2 = (line[2], line[3])
    
    # 提取矩形框的四个角点
    box_p1 = (box[0], box[1])  # 左上角
    box_p2 = (box[2], box[1])  # 右上角
    box_p3 = (box[2], box[3])  # 右下角
    box_p4 = (box[0], box[3])  # 左下角
    
    # 计算直线到矩形框四条边的最小距离
    distances = []
    
    # 直线到矩形框上边的距离
    distances.append(line_segment_distance(line_p1, line_p2, box_p1, box_p2))
    # 直线到矩形框右边的距离
    distances.append(line_segment_distance(line_p1, line_p2, box_p2, box_p3))
    # 直线到矩形框下边的距离
    distances.append(line_segment_distance(line_p1, line_p2, box_p3, box_p4))
    # 直线到矩形框左边的距离
    distances.append(line_segment_distance(line_p1, line_p2, box_p4, box_p1))
    
    return min(distances)


def line_segment_distance(p1, p2, p3, p4):
    """
    计算两条线段之间的最短距离
    p1, p2: 第一条线段的两个端点
    p3, p4: 第二条线段的两个端点
    """
    # 计算两条线段端点之间的距离
    d1 = point_to_line_segment_distance(p1, p3, p4)
    d2 = point_to_line_segment_distance(p2, p3, p4)
    d3 = point_to_line_segment_distance(p3, p1, p2)
    d4 = point_to_line_segment_distance(p4, p1, p2)
    
    return min(d1, d2, d3, d4)


def point_to_line_segment_distance(point, line_p1, line_p2):
    """
    计算点到线段的最短距离
    point: (x, y) 点坐标
    line_p1, line_p2: (x, y) 线段的两个端点
    """
    x, y = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    
    # 线段长度的平方
    l2 = (x2 - x1)**2 + (y2 - y1)**2
    
    if l2 == 0:  # 线段退化为点
        return ((x - x1)**2 + (y - y1)**2)**0.5
    
    # 计算投影比例 t
    t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / l2))
    
    # 计算投影点坐标
    projection_x = x1 + t * (x2 - x1)
    projection_y = y1 + t * (y2 - y1)
    
    # 返回点到投影点的距离
    return ((x - projection_x)**2 + (y - projection_y)**2)**0.5
##############################
def remove_contained_arrows(pairs_length, angle_threshold=15, position_threshold=20, overlap_threshold=0.7):
    """
    去除被包含在长箭头内的短箭头，要求方向一致且位置对齐
    
    参数:
    pairs_length: np.array, 形状为(n, 13)的二维数组
    angle_threshold: 角度阈值(度)，默认15度内认为方向一致
    position_threshold: 位置阈值(像素)，默认20像素内认为位置对齐
    overlap_threshold: 重叠阈值，默认0.7表示70%以上重叠则认为被包含
    """
    n = pairs_length.shape[0]
    if n == 0:
        return pairs_length
    
    # 提取箭头坐标 (前4列)
    arrows = pairs_length[:, :4]
    
    # 计算每个箭头的长度和角度
    arrow_lengths = np.sqrt((arrows[:, 2] - arrows[:, 0])**2 + 
                           (arrows[:, 3] - arrows[:, 1])**2)
    arrow_angles = np.degrees(np.arctan2(arrows[:, 3] - arrows[:, 1], 
                                        arrows[:, 2] - arrows[:, 0]))
    
    # 计算每个箭头的中心点
    centers_x = (arrows[:, 0] + arrows[:, 2]) / 2
    centers_y = (arrows[:, 1] + arrows[:, 3]) / 2
    
    # 按箭头长度降序排序
    sorted_indices = np.argsort(arrow_lengths)[::-1]
    
    # 标记要保留的箭头
    keep_mask = np.ones(n, dtype=bool)
    
    for i in range(n):
        if not keep_mask[sorted_indices[i]]:
            continue
            
        long_arrow_idx = sorted_indices[i]
        long_arrow = arrows[long_arrow_idx]
        long_angle = arrow_angles[long_arrow_idx]
        long_center_x = centers_x[long_arrow_idx]
        long_center_y = centers_y[long_arrow_idx]
        
        for j in range(i + 1, n):
            short_arrow_idx = sorted_indices[j]
            
            if not keep_mask[short_arrow_idx]:
                continue
                
            short_arrow = arrows[short_arrow_idx]
            short_angle = arrow_angles[short_arrow_idx]
            short_center_x = centers_x[short_arrow_idx]
            short_center_y = centers_y[short_arrow_idx]
            
            # 检查角度是否一致（考虑180度对称性）
            angle_diff = abs(short_angle - long_angle)
            angle_diff = min(angle_diff, 180 - angle_diff)  # 处理180度对称
            
            # 检查位置是否对齐
            position_aligned = are_arrows_aligned(
                long_arrow, short_arrow, long_angle, position_threshold)
            
            # 只有方向一致、位置对齐且被包含时才去除
            if (angle_diff <= angle_threshold and 
                position_aligned and 
                is_contained(short_arrow, long_arrow, overlap_threshold)):
                keep_mask[short_arrow_idx] = False
    
    return pairs_length[keep_mask]

def are_arrows_aligned(long_arrow, short_arrow, long_angle, position_threshold):
    """
    检查两个箭头是否位置对齐
    
    对于竖直箭头，检查x坐标是否接近
    对于水平箭头，检查y坐标是否接近
    对于斜向箭头，检查垂直距离是否接近
    """
    # 判断箭头方向
    is_vertical = abs(long_angle) > 45 and abs(long_angle) < 135
    is_horizontal = abs(long_angle) <= 45 or abs(long_angle) >= 135
    
    if is_vertical:
        # 对于竖直箭头，比较x坐标
        long_center_x = (long_arrow[0] + long_arrow[2]) / 2
        short_center_x = (short_arrow[0] + short_arrow[2]) / 2
        return abs(long_center_x - short_center_x) <= position_threshold
    
    elif is_horizontal:
        # 对于水平箭头，比较y坐标
        long_center_y = (long_arrow[1] + long_arrow[3]) / 2
        short_center_y = (short_arrow[1] + short_arrow[3]) / 2
        return abs(long_center_y - short_center_y) <= position_threshold
    
    else:
        # 对于斜向箭头，计算到长箭头直线的垂直距离
        # 使用点到直线的距离公式
        dx = long_arrow[2] - long_arrow[0]
        dy = long_arrow[3] - long_arrow[1]
        length = np.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            return False
            
        # 计算短箭头中心到长箭头直线的距离
        short_center_x = (short_arrow[0] + short_arrow[2]) / 2
        short_center_y = (short_arrow[1] + short_arrow[3]) / 2
        
        # 点到直线的距离公式
        distance = abs(dy*short_center_x - dx*short_center_y + 
                      long_arrow[2]*long_arrow[1] - long_arrow[3]*long_arrow[0]) / length
        
        return distance <= position_threshold

def is_contained(short_arrow, long_arrow, overlap_threshold=0.7):
    """
    判断短箭头是否被长箭头包含
    """
    # 计算短箭头的长度
    short_len = np.sqrt((short_arrow[2] - short_arrow[0])**2 + 
                       (short_arrow[3] - short_arrow[1])**2)
    
    # 计算短箭头两个端点到长箭头线段的距离
    dist1 = point_to_line_distance(short_arrow[0], short_arrow[1], 
                                  long_arrow[0], long_arrow[1], 
                                  long_arrow[2], long_arrow[3])
    dist2 = point_to_line_distance(short_arrow[2], short_arrow[3], 
                                  long_arrow[0], long_arrow[1], 
                                  long_arrow[2], long_arrow[3])
    
    # 计算短箭头中心点到长箭头线段的距离
    center_x = (short_arrow[0] + short_arrow[2]) / 2
    center_y = (short_arrow[1] + short_arrow[3]) / 2
    dist_center = point_to_line_distance(center_x, center_y,
                                        long_arrow[0], long_arrow[1],
                                        long_arrow[2], long_arrow[3])
    
    # 计算短箭头在长箭头方向上的投影长度
    projection_ratio = calculate_projection_ratio(short_arrow, long_arrow)
    
    # 判断条件：两个端点距离近，中心点距离近，且在长箭头方向上有足够长的投影
    max_distance = short_len * 0.2  # 允许的最大距离为短箭头长度的20%
    
    if (dist1 <= max_distance and dist2 <= max_distance and 
        dist_center <= max_distance and projection_ratio >= overlap_threshold):
        return True
    
    return False

def point_to_line_distance(px, py, x1, y1, x2, y2):
    """
    计算点到线段的距离
    """
    line_len_sq = (x2 - x1)**2 + (y2 - y1)**2
    
    if line_len_sq == 0:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq))
    
    projection_x = x1 + t * (x2 - x1)
    projection_y = y1 + t * (y2 - y1)
    
    return np.sqrt((px - projection_x)**2 + (py - projection_y)**2)

def calculate_projection_ratio(short_arrow, long_arrow):
    """
    计算短箭头在长箭头方向上的投影比例
    """
    long_dx = long_arrow[2] - long_arrow[0]
    long_dy = long_arrow[3] - long_arrow[1]
    long_length = np.sqrt(long_dx**2 + long_dy**2)
    
    if long_length == 0:
        return 0
    
    ux = long_dx / long_length
    uy = long_dy / long_length
    
    short_dx = short_arrow[2] - short_arrow[0]
    short_dy = short_arrow[3] - short_arrow[1]
    
    projection_length = abs(short_dx * ux + short_dy * uy)
    short_length = np.sqrt(short_dx**2 + short_dy**2)
    
    if short_length == 0:
        return 0
    
    return projection_length / short_length

##############################
# def find_pairs_length(img_path, pairs, test_mode):
#     '''
#     功能：检测标尺线附近成对的引线
#     pairs np.二维数组[x1,y1,x2,y2,0 = outside 1 = inside]
#     img_path str
#     '''
#     print("***/开始引线和标尺线的匹配/***")
#     # 1.根据pinmap所在位置推测出大概十字线坐标
#     pin_map_limation = get_np_array_in_txt(f'{YOLO_DATA}/pin_map_limation.txt')
#     w, h = get_img_info(img_path)
#     a = np.array(([1, 1, 1, 1]))
#     if (pin_map_limation != a).all():
#         heng = (pin_map_limation[0][3] + pin_map_limation[0][1]) * 0.5 / 2
#         shu = (pin_map_limation[0][2] + pin_map_limation[0][0]) * 0.5 / 2
#     if img_path == f'{YOLO_DATA}/side.jpg' or img_path == f'{YOLO_DATA}/top.jpg':
#         heng = h / 2
#         shu = w / 2

#     ver_lines_heng, ver_lines_shu = find_all_lines(img_path, test_mode)

#     pairs_length = np.zeros((0, 13))  # 存储pairs以及所表示的距离
#     pairs_length_middle = np.zeros(13)

#     # 横向直线的坐标排列np.二维数组[x1,y1,x2,y2]改为x1<x2 y1<y2
#     new_ver_lines = np.zeros((0, 4))
#     for i in range(len(ver_lines_heng)):
#         if ver_lines_heng[i][0] > ver_lines_heng[i][2]:
#             c = ver_lines_heng[i][0]
#             ver_lines_heng[i][0] = ver_lines_heng[i][2]
#             ver_lines_heng[i][2] = c
#         if ver_lines_heng[i][1] > ver_lines_heng[i][3]:
#             c = ver_lines_heng[i][1]
#             ver_lines_heng[i][1] = ver_lines_heng[i][3]
#             ver_lines_heng[i][3] = c
#     # 滤除较短的直线
#     min_length = 10  # 最短直线长
#     for i in range(len(ver_lines_heng)):
#         if max(abs(ver_lines_heng[i][2] - ver_lines_heng[i][0]),
#                abs(ver_lines_heng[i][3] - ver_lines_heng[i][1])) > min_length:
#             new_ver_lines = np.r_[new_ver_lines, [ver_lines_heng[i]]]
#     ver_lines_heng = new_ver_lines
#     min_length = 20  # 最短直线长
#     new_ver_lines = np.zeros((0, 4))
#     ver_lines_shu = np.array(ver_lines_shu)
#     for i in range(len(ver_lines_shu)):
#         if max(abs(ver_lines_shu[i][2] - ver_lines_shu[i][0]),
#                abs(ver_lines_shu[i][3] - ver_lines_shu[i][1])) > min_length:
#             new_ver_lines = np.r_[new_ver_lines, [ver_lines_shu[i]]]
#     ver_lines_shu = new_ver_lines
    
#     # 新增：计算引线与箭头的匹配评分函数 - 简化版本，主要考虑距离
#     def calculate_match_score_simple(arrow_pair, line, heng, is_left=True):
#         """
#         简化版匹配评分，主要考虑距离
#         分数越高表示匹配越好
#         """
#         score = 0
        
#         # 1. 检查引线是否在箭头范围内或附近
#         arrow_y_center = (arrow_pair[1] + arrow_pair[3]) / 2
#         line_y_center = (line[1] + line[3]) / 2
#         line_y_range = line[3] - line[1]
#         arrow_y_range = arrow_pair[3] - arrow_pair[1]
        
#         # 计算y方向重叠
#         y_overlap_start = max(arrow_pair[1], line[1])
#         y_overlap_end = min(arrow_pair[3], line[3])
#         y_overlap = max(0, y_overlap_end - y_overlap_start)
        
#         # 如果完全没有重叠，检查距离是否在可接受范围内
#         if y_overlap == 0:
#             # 计算最近距离
#             y_distance = min(abs(arrow_pair[1] - line[3]), abs(arrow_pair[3] - line[1]))
#             if y_distance > 3 * arrow_y_range:  # 距离超过3倍箭头高度
#                 return -1000  # 完全不匹配
        
#         # 2. 水平距离评分 - 这是最重要的因素
#         if is_left:
#             x_distance = abs(line[0] - arrow_pair[0])
#         else:
#             x_distance = abs(line[0] - arrow_pair[2])
        
#         # 距离越小分数越高，使用指数衰减函数
#         # 距离在0-50像素内，分数为100-0
#         if x_distance <= 50:
#             score = 100 * (1 - x_distance / 50)
#         else:
#             score = max(0, 10 * (1 - x_distance / 100))  # 距离大于50时分数迅速下降
        
#         # 3. 重叠奖励
#         if y_overlap > 0:
#             overlap_ratio = y_overlap / min(arrow_y_range, line_y_range)
#             score += 20 * overlap_ratio
        
#         # 4. Y方向中心对齐奖励
#         center_diff = abs(arrow_y_center - line_y_center)
#         if center_diff < 30:
#             score += 10 * (1 - center_diff / 30)
        
#         return score
    
#     ratio = 0.4
#     ra = 2
#     # save_box_to_txt(ver_lines_heng, "D:\\BaiduNetdiskDownload\\post0\\0910\\QFN\\QFN_4_test\\ver_lines_heng.txt")
#     # save_box_to_txt(ver_lines_shu, "D:\\BaiduNetdiskDownload\\post0\\0910\\QFN\\QFN_4_test\\ver_lines_shu.txt")
#     print("开始视图**")
    
#     for i in range(len(pairs)):
#         print("一组pairs开始*")
#         print(f'开始{pairs[i]}判断')
#         print(f'横{heng}')
#         print(f'竖{shu}')
        
#         if pairs[i][4] == 0:  # 外向标尺线
#             print("外向")
#             ratio = 0.3  # 增加ratio，让搜索范围更大
            
#             if (pairs[i][2] - pairs[i][0]) > (pairs[i][3] - pairs[i][1]):  # 横向标尺线
#                 print("横向")
#                 left_straight = np.zeros((0, 6))  # 增加一列存储评分
#                 right_straight = np.zeros((0, 6))
#                 middle = np.zeros(6)
                
#                 for j in range(len(ver_lines_shu)):
#                     # 简化：主要根据水平距离匹配
#                     left_distance = abs(ver_lines_shu[j][0] - pairs[i][0])
#                     right_distance = abs(ver_lines_shu[j][0] - pairs[i][2])
                    
#                     # 左引线匹配条件
#                     if left_distance < 50:  # 距离小于50像素
#                         # 检查Y方向是否有重叠或接近
#                         y_overlap_start = max(pairs[i][1], ver_lines_shu[j][1])
#                         y_overlap_end = min(pairs[i][3], ver_lines_shu[j][3])
#                         y_overlap = max(0, y_overlap_end - y_overlap_start)
                        
#                         if y_overlap > 0 or min(abs(pairs[i][1] - ver_lines_shu[j][3]), 
#                                                 abs(pairs[i][3] - ver_lines_shu[j][1])) < 20:
#                             # 计算简化评分
#                             left_score = calculate_match_score_simple(pairs[i], ver_lines_shu[j], heng, is_left=True)
                            
#                             if left_score > 0:
#                                 middle[0:4] = ver_lines_shu[j]
#                                 middle[4] = left_distance  # 存储距离
#                                 middle[5] = left_score  # 存储评分
#                                 left_straight = np.r_[left_straight, [middle]]
#                                 print(f"外向横pairs找到左竖线，距离: {left_distance}, 评分: {left_score}")
                    
#                     # 右引线匹配条件
#                     if right_distance < 50:  # 距离小于50像素
#                         # 检查Y方向是否有重叠或接近
#                         y_overlap_start = max(pairs[i][1], ver_lines_shu[j][1])
#                         y_overlap_end = min(pairs[i][3], ver_lines_shu[j][3])
#                         y_overlap = max(0, y_overlap_end - y_overlap_start)
                        
#                         if y_overlap > 0 or min(abs(pairs[i][1] - ver_lines_shu[j][3]), 
#                                                 abs(pairs[i][3] - ver_lines_shu[j][1])) < 20:
#                             # 计算简化评分
#                             right_score = calculate_match_score_simple(pairs[i], ver_lines_shu[j], heng, is_left=False)
                            
#                             if right_score > 0:
#                                 middle[0:4] = ver_lines_shu[j]
#                                 middle[4] = right_distance  # 存储距离
#                                 middle[5] = right_score  # 存储评分
#                                 right_straight = np.r_[right_straight, [middle]]
#                                 print(f"外向横pairs找到右竖线，距离: {right_distance}, 评分: {right_score}")
                
#                 # 按距离从小到大排序（主要排序依据），距离相同再按评分排序
#                 if len(left_straight) > 0:
#                     # 先按距离排序，距离小的在前
#                     left_straight = left_straight[np.argsort(left_straight[:, 4])]
#                     # 如果前几个距离相同，再按评分排序
#                     # 简单实现：取距离最小的几个，再按评分排序
#                     min_distance = left_straight[0, 4]
#                     same_distance_indices = np.where(left_straight[:, 4] <= min_distance + 1)[0]
#                     if len(same_distance_indices) > 1:
#                         # 对距离相同的部分按评分排序
#                         same_distance_lines = left_straight[same_distance_indices]
#                         same_distance_lines = same_distance_lines[np.argsort(-same_distance_lines[:, 5])]
#                         left_straight[same_distance_indices] = same_distance_lines
                
#                 if len(right_straight) > 0:
#                     # 先按距离排序，距离小的在前
#                     right_straight = right_straight[np.argsort(right_straight[:, 4])]
#                     # 如果前几个距离相同，再按评分排序
#                     min_distance = right_straight[0, 4]
#                     same_distance_indices = np.where(right_straight[:, 4] <= min_distance + 1)[0]
#                     if len(same_distance_indices) > 1:
#                         # 对距离相同的部分按评分排序
#                         same_distance_lines = right_straight[same_distance_indices]
#                         same_distance_lines = same_distance_lines[np.argsort(-same_distance_lines[:, 5])]
#                         right_straight[same_distance_indices] = same_distance_lines
                
#                 # 根据找到的引线数量处理
#                 if len(left_straight) > 0 and len(right_straight) > 0:
#                     # 找到两条引线，直接使用距离最小的
#                     pairs_length_middle[0:4] = pairs[i, 0:4]
#                     pairs_length_middle[4:8] = left_straight[0, 0:4]
#                     pairs_length_middle[8:12] = right_straight[0, 0:4]
#                     pairs_length_middle[12] = abs(left_straight[0, 0] - right_straight[0, 0])
#                     pairs_length = np.r_[pairs_length, [pairs_length_middle]]
#                     print(f"找到两条引线，左距离: {left_straight[0, 4]}, 右距离: {right_straight[0, 4]}")
#                 elif len(left_straight) > 0 or len(right_straight) > 0:
#                     # 只找到一条引线，生成另一条
#                     pairs_length_middle[0:4] = pairs[i, 0:4]
                    
#                     # 如果找到左侧引线
#                     if len(left_straight) > 0:
#                         pairs_length_middle[4:8] = left_straight[0, 0:4]
#                         left_line = left_straight[0, 0:4]
                        
#                         # 计算左侧引线与箭头对的距离
#                         left_distance = left_line[0] - pairs[i][0]
                        
#                         # 生成右侧引线（与左侧对称）
#                         right_line = [
#                             pairs[i][2] - abs(left_distance),  # 保持与左侧对称的距离
#                             left_line[1],  # y1 - 保持与左侧引线相同的高度
#                             pairs[i][2] - abs(left_distance),  # x2 - 保持与左侧对称的距离
#                             left_line[3]   # y2 - 保持与左侧引线相同的高度
#                         ]
#                         pairs_length_middle[8:12] = right_line
#                         print("根据左侧引线生成右侧引线")
                    
#                     # 如果找到右侧引线
#                     elif len(right_straight) > 0:
#                         pairs_length_middle[8:12] = right_straight[0, 0:4]
#                         right_line = right_straight[0, 0:4]
                        
#                         # 计算右侧引线与箭头对的距离
#                         right_distance = right_line[0] - pairs[i][2]
                        
#                         # 生成左侧引线（与右侧对称）
#                         left_line = [
#                             pairs[i][0] + abs(right_distance),  # 保持与右侧对称的距离
#                             right_line[1],  # y1 - 保持与右侧引线相同的高度
#                             pairs[i][0] + abs(right_distance),  # x2 - 保持与右侧对称的距离
#                             right_line[3]   # y2 - 保持与右侧引线相同的高度
#                         ]
#                         pairs_length_middle[4:8] = left_line
#                         print("根据右侧引线生成左侧引线")
                    
#                     # 计算距离
#                     pairs_length_middle[12] = abs(pairs_length_middle[4] - pairs_length_middle[8])
#                     pairs_length = np.r_[pairs_length, [pairs_length_middle]]
#                 else:
#                     # 没有找到任何引线，不保存这个pairs
#                     print("没有找到任何引线，跳过这个pairs")
            
#             # 竖向标尺线的处理类似
#             if (pairs[i][2] - pairs[i][0]) < (pairs[i][3] - pairs[i][1]):  # 竖向标尺线
#                 print("竖向")
#                 up_straight = np.zeros((0, 5))
#                 down_straight = np.zeros((0, 5))
#                 middle = np.zeros((5))
                
#                 for j in range(len(ver_lines_heng)):
#                     # 上端附近的直线
#                     if pairs[i][1] - ratio * (pairs[i][3] - pairs[i][1]) < ver_lines_heng[j][1] < pairs[i][1] + ratio * (
#                             pairs[i][3] - pairs[i][1]):
#                         if (not (pairs[i][0] > ver_lines_heng[j][2] or pairs[i][2] < ver_lines_heng[j][0])) or min(
#                                 abs(pairs[i][0] - ver_lines_heng[j][2]),
#                                 abs(pairs[i][2] - ver_lines_heng[j][0])) < ra * (
#                                 pairs[i][2] - pairs[i][0]):
#                             if (pairs[i][0] > shu and ver_lines_heng[j][0] < pairs[i][0]) or (
#                                     pairs[i][0] < shu and ver_lines_heng[j][2] > pairs[i][2]):
#                                 middle[0:4] = ver_lines_heng[j]
#                                 middle[4] = abs(pairs[i][1] - ver_lines_heng[j][1])
#                                 up_straight = np.r_[up_straight, [middle]]
#                                 print("外向竖pairs找到上横线")

#                     # 下端附近的直线
#                     if pairs[i][3] - ratio * (pairs[i][3] - pairs[i][1]) < ver_lines_heng[j][1] < pairs[i][3] + ratio * (
#                             pairs[i][3] - pairs[i][1]):
#                         if (not (pairs[i][0] > ver_lines_heng[j][2] or pairs[i][2] < ver_lines_heng[j][0])) or min(
#                                 abs(pairs[i][0] - ver_lines_heng[j][2]),
#                                 abs(pairs[i][2] - ver_lines_heng[j][0])) < ra * (
#                                 pairs[i][2] - pairs[i][0]):
#                             if (pairs[i][0] > shu and ver_lines_heng[j][0] < pairs[i][0]) or (
#                                     pairs[i][0] < shu and ver_lines_heng[j][2] > pairs[i][2]):
#                                 middle[0:4] = ver_lines_heng[j]
#                                 middle[4] = abs(pairs[i][0] - ver_lines_heng[j][0])
#                                 down_straight = np.r_[down_straight, [middle]]
#                                 print("外向竖pairs找到下横线")
                
#                 up_straight = up_straight[np.argsort(up_straight[:, 4])]
#                 down_straight = down_straight[np.argsort(down_straight[:, 4])]
                
#                 # 修改：根据找到的引线数量处理
#                 if len(up_straight) > 0 and len(down_straight) > 0:
#                     pairs_length_middle[0:4] = pairs[i, 0:4]
#                     pairs_length_middle[4:8] = up_straight[0, 0:4]
#                     pairs_length_middle[8:12] = down_straight[0, 0:4]
#                     pairs_length_middle[12] = abs(up_straight[0, 1] - down_straight[0, 1])
#                     pairs_length = np.r_[pairs_length, [pairs_length_middle]]
#                     print("找到两条引线，直接使用")
#                 elif len(up_straight) > 0 or len(down_straight) > 0:
#                     pairs_length_middle[0:4] = pairs[i, 0:4]
                    
#                     if len(up_straight) > 0:
#                         pairs_length_middle[4:8] = up_straight[0, 0:4]
#                         up_line = up_straight[0, 0:4]
                        
#                         up_distance = up_line[1] - pairs[i][1]
                        
#                         # 生成下侧引线（与上侧对称）
#                         down_line = [
#                             up_line[0],
#                             pairs[i][3] - abs(up_distance),  # 保持与上侧对称的距离
#                             up_line[2],
#                             pairs[i][3] - abs(up_distance)
#                         ]
#                         pairs_length_middle[8:12] = down_line
#                         print("根据上侧引线生成下侧引线")
                    
#                     elif len(down_straight) > 0:
#                         pairs_length_middle[8:12] = down_straight[0, 0:4]
#                         down_line = down_straight[0, 0:4]
                        
#                         down_distance = down_line[1] - pairs[i][3]
                        
#                         # 生成上侧引线（与下侧对称）
#                         up_line = [
#                             down_line[0],
#                             pairs[i][1] + abs(down_distance),  # 保持与下侧对称的距离
#                             down_line[2],
#                             pairs[i][1] + abs(down_distance)
#                         ]
#                         pairs_length_middle[4:8] = up_line
#                         print("根据下侧引线生成上侧引线")
                    
#                     pairs_length_middle[12] = abs(pairs_length_middle[5] - pairs_length_middle[9])
#                     pairs_length = np.r_[pairs_length, [pairs_length_middle]]
#                 else:
#                     print("没有找到任何引线，跳过这个pairs")
        
#         # 内向标尺线的处理（保持原有逻辑）
#         if pairs[i][4] == 1:  # 内向标尺线
#             print("内向")
#             ratio = 0.5
#             ratio_inside = 0.15
            
#             if (pairs[i][2] - pairs[i][0]) > (pairs[i][3] - pairs[i][1]):  # 横向标尺线
#                 print("横向")
#                 left_straight = np.zeros((0, 5))
#                 right_straight = np.zeros((0, 5))
#                 middle = np.zeros((5))
                
#                 for j in range(len(ver_lines_shu)):
#                     # 左端附近的直线
#                     if pairs[i][0] - ratio_inside * (pairs[i][2] - pairs[i][0]) < ver_lines_shu[j][0] < pairs[i][0] + ratio * (
#                             pairs[i][2] - pairs[i][0]):
#                         if (not (pairs[i][1] > ver_lines_shu[j][3] or pairs[i][3] < ver_lines_shu[j][1])) or min(
#                                 abs(pairs[i][1] - ver_lines_shu[j][3]), abs(pairs[i][3] - ver_lines_shu[j][1])) < ra * (
#                                 pairs[i][3] - pairs[i][1]):
#                             if (pairs[i][1] < heng and ver_lines_shu[j][3] > pairs[i][3]) or (
#                                     pairs[i][1] > heng and ver_lines_shu[j][1] < pairs[i][1]):
#                                 middle[0:4] = ver_lines_shu[j]
#                                 middle[4] = abs(pairs[i][0] - ver_lines_shu[j][0])
#                                 left_straight = np.r_[left_straight, [middle]]
#                                 print("内向横pairs找到左竖线")

#                     # 右端附近的直线
#                     if pairs[i][2] - ratio * (pairs[i][2] - pairs[i][0]) < ver_lines_shu[j][0] < pairs[i][2] + ratio_inside * (pairs[i][2] - pairs[i][0]):
#                         if (not (pairs[i][1] > ver_lines_shu[j][3] or pairs[i][3] < ver_lines_shu[j][1])) or min(
#                                 abs(pairs[i][1] - ver_lines_shu[j][3]), abs(pairs[i][3] - ver_lines_shu[j][1])) < ra * (
#                                 pairs[i][3] - pairs[i][1]):
#                             if (pairs[i][1] < heng and ver_lines_shu[j][3] > pairs[i][3]) or (
#                                     pairs[i][1] > heng and ver_lines_shu[j][1] < pairs[i][1]):
#                                 middle[0:4] = ver_lines_shu[j]
#                                 middle[4] = abs(pairs[i][0] - ver_lines_shu[j][0])
#                                 right_straight = np.r_[right_straight, [middle]]
#                                 print("内向横pairs找到右竖线")
                
#                 left_straight = left_straight[np.argsort(left_straight[:, 4])]
#                 right_straight = right_straight[np.argsort(right_straight[:, 4])]
                
#                 # 修改：根据找到的引线数量处理
#                 if len(left_straight) > 0 and len(right_straight) > 0:
#                     pairs_length_middle[0:4] = pairs[i, 0:4]
#                     pairs_length_middle[4:8] = left_straight[0, 0:4]
#                     pairs_length_middle[8:12] = right_straight[0, 0:4]
#                     pairs_length_middle[12] = abs(left_straight[0, 0] - right_straight[0, 0])
#                     pairs_length = np.r_[pairs_length, [pairs_length_middle]]
#                     print("找到两条引线，直接使用")
#                 elif len(left_straight) > 0 or len(right_straight) > 0:
#                     pairs_length_middle[0:4] = pairs[i, 0:4]
                    
#                     if len(left_straight) > 0:
#                         pairs_length_middle[4:8] = left_straight[0, 0:4]
#                         left_line = left_straight[0, 0:4]
                        
#                         left_distance = left_line[0] - pairs[i][0]
                        
#                         right_line = [
#                             pairs[i][2] - abs(left_distance),
#                             left_line[1],
#                             pairs[i][2] - abs(left_distance),
#                             left_line[3]
#                         ]
#                         pairs_length_middle[8:12] = right_line
#                         print("根据左侧引线生成右侧引线（内向）")
                    
#                     elif len(right_straight) > 0:
#                         pairs_length_middle[8:12] = right_straight[0, 0:4]
#                         right_line = right_straight[0, 0:4]
                        
#                         right_distance = right_line[0] - pairs[i][2]
                        
#                         left_line = [
#                             pairs[i][0] + abs(right_distance),
#                             right_line[1],
#                             pairs[i][0] + abs(right_distance),
#                             right_line[3]
#                         ]
#                         pairs_length_middle[4:8] = left_line
#                         print("根据右侧引线生成左侧引线（内向）")
                    
#                     pairs_length_middle[12] = abs(pairs_length_middle[4] - pairs_length_middle[8])
#                     pairs_length = np.r_[pairs_length, [pairs_length_middle]]
#                 else:
#                     print("没有找到任何引线，跳过这个pairs（内向）")
            
#             if (pairs[i][2] - pairs[i][0]) < (pairs[i][3] - pairs[i][1]):  # 竖向标尺线
#                 print("竖向")
#                 up_straight = np.zeros((0, 5))
#                 down_straight = np.zeros((0, 5))
#                 middle = np.zeros((5))
                
#                 for j in range(len(ver_lines_heng)):
#                     # 上端附近的直线
#                     if pairs[i][1] + ratio_inside * (pairs[i][3] - pairs[i][1]) < ver_lines_heng[j][1] < pairs[i][1] + ratio * (
#                             pairs[i][3] - pairs[i][1]):
#                         if (not (pairs[i][0] > ver_lines_heng[j][2] or pairs[i][2] < ver_lines_heng[j][0])) or min(
#                                 abs(pairs[i][0] - ver_lines_heng[j][2]),
#                                 abs(pairs[i][2] - ver_lines_heng[j][0])) < ra * (
#                                 pairs[i][2] - pairs[i][0]):
#                             if (pairs[i][0] > shu and ver_lines_heng[j][0] < pairs[i][0]) or (
#                                     pairs[i][0] < shu and ver_lines_heng[j][2] > pairs[i][2]):
#                                 middle[0:4] = ver_lines_heng[j]
#                                 middle[4] = abs(pairs[i][1] - ver_lines_heng[j][1])
#                                 up_straight = np.r_[up_straight, [middle]]
#                                 print("内向竖pairs找到上横线")

#                     # 下端附近的直线
#                     if pairs[i][3] - ratio * (pairs[i][3] - pairs[i][1]) < ver_lines_heng[j][1] < pairs[i][3] - ratio_inside * (
#                             pairs[i][3] - pairs[i][1]):
#                         if (not (pairs[i][0] > ver_lines_heng[j][2] or pairs[i][2] < ver_lines_heng[j][0])) or min(
#                                 abs(pairs[i][0] - ver_lines_heng[j][2]),
#                                 abs(pairs[i][2] - ver_lines_heng[j][0])) < ra * (
#                                 pairs[i][2] - pairs[i][0]):
#                             if (pairs[i][0] > shu and ver_lines_heng[j][0] < pairs[i][0]) or (
#                                     pairs[i][0] < shu and ver_lines_heng[j][2] > pairs[i][2]):
#                                 middle[0:4] = ver_lines_heng[j]
#                                 middle[4] = abs(pairs[i][0] - ver_lines_heng[j][0])
#                                 down_straight = np.r_[down_straight, [middle]]
#                                 print("内向竖pairs找到下横线")
                
#                 up_straight = up_straight[np.argsort(up_straight[:, 4])]
#                 down_straight = down_straight[np.argsort(down_straight[:, 4])]
                
#                 # 修改：根据找到的引线数量处理
#                 if len(up_straight) > 0 and len(down_straight) > 0:
#                     pairs_length_middle[0:4] = pairs[i, 0:4]
#                     pairs_length_middle[4:8] = up_straight[0, 0:4]
#                     pairs_length_middle[8:12] = down_straight[0, 0:4]
#                     pairs_length_middle[12] = abs(up_straight[0, 1] - down_straight[0, 1])
#                     pairs_length = np.r_[pairs_length, [pairs_length_middle]]
#                     print("找到两条引线，直接使用")
#                 elif len(up_straight) > 0 or len(down_straight) > 0:
#                     pairs_length_middle[0:4] = pairs[i, 0:4]
                    
#                     if len(up_straight) > 0:
#                         pairs_length_middle[4:8] = up_straight[0, 0:4]
#                         up_line = up_straight[0, 0:4]
                        
#                         up_distance = up_line[1] - pairs[i][1]
                        
#                         down_line = [
#                             up_line[0],
#                             pairs[i][3] - abs(up_distance),
#                             up_line[2],
#                             pairs[i][3] - abs(up_distance)
#                         ]
#                         pairs_length_middle[8:12] = down_line
#                         print("根据上侧引线生成下侧引线（内向）")
                    
#                     elif len(down_straight) > 0:
#                         pairs_length_middle[8:12] = down_straight[0, 0:4]
#                         down_line = down_straight[0, 0:4]
                        
#                         down_distance = down_line[1] - pairs[i][3]
                        
#                         up_line = [
#                             down_line[0],
#                             pairs[i][1] + abs(down_distance),
#                             down_line[2],
#                             pairs[i][1] + abs(down_distance)
#                         ]
#                         pairs_length_middle[4:8] = up_line
#                         print("根据下侧引线生成上侧引线（内向）")
                    
#                     pairs_length_middle[12] = abs(pairs_length_middle[5] - pairs_length_middle[9])
#                     pairs_length = np.r_[pairs_length, [pairs_length_middle]]
#                 else:
#                     print("没有找到任何引线，跳过这个pairs（内向）")
        
#         print("一组pairs结束*")
    
#     # 可视化部分保持不变
#     if test_mode == 1:
#         drawn_img = cv2.imread(img_path)
#         for i in range(len(pairs_length)):
#             x1 = int(pairs_length[i][4])
#             x2 = int(pairs_length[i][6])
#             y1 = int(pairs_length[i][5])
#             y2 = int(pairs_length[i][7])
#             drawn_img = cv2.line(drawn_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#             x1 = int(pairs_length[i][8])
#             x2 = int(pairs_length[i][10])
#             y1 = int(pairs_length[i][9])
#             y2 = int(pairs_length[i][11])
#             drawn_img = cv2.line(drawn_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#         for i in range(len(pairs_length)):
#             ptLeftTop = (int(pairs_length[i][0]), int(pairs_length[i][1]))
#             ptRightBottom = (int(pairs_length[i][2]), int(pairs_length[i][3]))
#             point_color = (0, 0, 255)
#             thickness = 2
#             lineType = 8
#             cv2.rectangle(drawn_img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
#         cv2.namedWindow("LSD", 0)
#         cv2.imshow("LSD", drawn_img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         print("视图结束**")
    
#     try:
#         drawn_img = cv2.imread(img_path)
#         for i in range(len(pairs_length)):
#             x1 = int(pairs_length[i][4])
#             x2 = int(pairs_length[i][6])
#             y1 = int(pairs_length[i][5])
#             y2 = int(pairs_length[i][7])
#             drawn_img = cv2.line(drawn_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#             x1 = int(pairs_length[i][8])
#             x2 = int(pairs_length[i][10])
#             y1 = int(pairs_length[i][9])
#             y2 = int(pairs_length[i][11])
#             drawn_img = cv2.line(drawn_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#         for i in range(len(pairs_length)):
#             ptLeftTop = (int(pairs_length[i][0]), int(pairs_length[i][1]))
#             ptRightBottom = (int(pairs_length[i][2]), int(pairs_length[i][3]))
#             point_color = (0, 0, 255)
#             thickness = 2
#             lineType = 8
#             cv2.rectangle(drawn_img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
#         path = fr'{OPENCV_OUTPUT_LINE}/' + img_path[-7:]
#         cv2.imwrite(path, drawn_img)
#         print("保存引线+标尺线组合成功:", path)
#     except:
#         print("保存引线+标尺线组合失败")
    
#     print("***/结束引线和标尺线的匹配/***")
#     pairs_length = remove_duplicate_pairs(pairs_length)
#     pairs_length = remove_contained_arrows(pairs_length)
#     return pairs_length

#####################优化######################
def find_pairs_length(img_path, pairs, test_mode):
    '''
    功能：检测标尺线附近成对的引线
    pairs np.二维数组[x1,y1,x2,y2,0 = outside 1 = inside]
    img_path str
    '''
    print("***/开始引线和标尺线的匹配/***")
    # 1.根据pinmap所在位置推测出大概十字线坐标
    pin_map_limation = get_np_array_in_txt(f'{YOLO_DATA}/pin_map_limation.txt')
    w, h = get_img_info(img_path)
    a = np.array(([1, 1, 1, 1]))
    if (pin_map_limation != a).all():
        heng = (pin_map_limation[0][3] + pin_map_limation[0][1]) * 0.5 / 2
        shu = (pin_map_limation[0][2] + pin_map_limation[0][0]) * 0.5 / 2
    if img_path == f'{YOLO_DATA}/side.jpg' or img_path == f'{YOLO_DATA}/top.jpg':
        heng = h / 2
        shu = w / 2

    ver_lines_heng, ver_lines_shu = find_all_lines(img_path, test_mode)

    pairs_length = np.zeros((0, 13))  # 存储pairs以及所表示的距离
    pairs_length_middle = np.zeros(13)

    # 横向直线的坐标排列np.二维数组[x1,y1,x2,y2]改为x1<x2 y1<y2
    new_ver_lines = np.zeros((0, 4))
    for i in range(len(ver_lines_heng)):
        if ver_lines_heng[i][0] > ver_lines_heng[i][2]:
            c = ver_lines_heng[i][0]
            ver_lines_heng[i][0] = ver_lines_heng[i][2]
            ver_lines_heng[i][2] = c
        if ver_lines_heng[i][1] > ver_lines_heng[i][3]:
            c = ver_lines_heng[i][1]
            ver_lines_heng[i][1] = ver_lines_heng[i][3]
            ver_lines_heng[i][3] = c
    # 滤除较短的直线
    min_length = 10  # 最短直线长
    for i in range(len(ver_lines_heng)):
        if max(abs(ver_lines_heng[i][2] - ver_lines_heng[i][0]),
               abs(ver_lines_heng[i][3] - ver_lines_heng[i][1])) > min_length:
            new_ver_lines = np.r_[new_ver_lines, [ver_lines_heng[i]]]
    ver_lines_heng = new_ver_lines
    min_length = 20  # 最短直线长
    new_ver_lines = np.zeros((0, 4))
    ver_lines_shu = np.array(ver_lines_shu)
    for i in range(len(ver_lines_shu)):
        if max(abs(ver_lines_shu[i][2] - ver_lines_shu[i][0]),
               abs(ver_lines_shu[i][3] - ver_lines_shu[i][1])) > min_length:
            new_ver_lines = np.r_[new_ver_lines, [ver_lines_shu[i]]]
    ver_lines_shu = new_ver_lines
    
    # 新增：计算引线与箭头的匹配评分函数 - 简化版本，主要考虑距离
    def calculate_match_score_simple(arrow_pair, line, heng, is_left=True, is_top=None):
        """
        简化版匹配评分，支持水平和竖直方向
        分数越高表示匹配越好
        
        参数:
        - arrow_pair: 箭头对 [x1,y1,x2,y2,type]
        - line: 引线 [x1,y1,x2,y2]
        - heng: 水平中心线
        - is_left: 是否为左/上侧引线（水平方向）
        - is_top: 是否为上侧引线（竖直方向）
        """
        score = 0
        
        # 判断是水平箭头还是竖直箭头
        is_horizontal = (arrow_pair[2] - arrow_pair[0]) > (arrow_pair[3] - arrow_pair[1])
        
        if is_horizontal:
            # 水平箭头处理
            arrow_y_center = (arrow_pair[1] + arrow_pair[3]) / 2
            line_y_center = (line[1] + line[3]) / 2
            line_y_range = line[3] - line[1]
            arrow_y_range = arrow_pair[3] - arrow_pair[1]
            
            # 计算y方向重叠
            y_overlap_start = max(arrow_pair[1], line[1])
            y_overlap_end = min(arrow_pair[3], line[3])
            y_overlap = max(0, y_overlap_end - y_overlap_start)
            
            # 如果完全没有重叠，检查距离是否在可接受范围内
            if y_overlap == 0:
                # 计算最近距离
                y_distance = min(abs(arrow_pair[1] - line[3]), abs(arrow_pair[3] - line[1]))
                if y_distance > 3 * arrow_y_range:  # 距离超过3倍箭头高度
                    return -1000  # 完全不匹配
            
            # 2. 水平距离评分 - 这是最重要的因素
            if is_left:
                x_distance = abs(line[0] - arrow_pair[0])
            else:
                x_distance = abs(line[0] - arrow_pair[2])
            
            # 距离越小分数越高
            if x_distance <= 50:
                score = 100 * (1 - x_distance / 50)
            else:
                score = max(0, 10 * (1 - x_distance / 100))
            
            # 3. 重叠奖励
            if y_overlap > 0:
                overlap_ratio = y_overlap / min(arrow_y_range, line_y_range)
                score += 20 * overlap_ratio
            
            # 4. Y方向中心对齐奖励
            center_diff = abs(arrow_y_center - line_y_center)
            if center_diff < 30:
                score += 10 * (1 - center_diff / 30)
        
        else:
            # 竖直箭头处理
            arrow_x_center = (arrow_pair[0] + arrow_pair[2]) / 2
            line_x_center = (line[0] + line[2]) / 2
            line_x_range = abs(line[2] - line[0])
            arrow_x_range = abs(arrow_pair[2] - arrow_pair[0])
            
            # 计算x方向重叠
            x_overlap_start = max(arrow_pair[0], line[0])
            x_overlap_end = min(arrow_pair[2], line[2])
            x_overlap = max(0, x_overlap_end - x_overlap_start)
            
            # 如果完全没有重叠，检查距离是否在可接受范围内
            if x_overlap == 0:
                # 计算最近距离
                x_distance = min(abs(arrow_pair[0] - line[2]), abs(arrow_pair[2] - line[0]))
                if x_distance > 3 * arrow_x_range:  # 距离超过3倍箭头宽度
                    return -1000  # 完全不匹配
            
            # 2. 竖直距离评分
            if is_top:
                y_distance = abs(line[1] - arrow_pair[1])
            else:
                y_distance = abs(line[1] - arrow_pair[3])
            
            # 距离越小分数越高
            if y_distance <= 50:
                score = 100 * (1 - y_distance / 50)
            else:
                score = max(0, 10 * (1 - y_distance / 100))
            
            # 3. 重叠奖励
            if x_overlap > 0:
                overlap_ratio = x_overlap / min(arrow_x_range, line_x_range)
                score += 20 * overlap_ratio
            
            # 4. X方向中心对齐奖励
            center_diff = abs(arrow_x_center - line_x_center)
            if center_diff < 30:
                score += 10 * (1 - center_diff / 30)
        
        return score
    
    ratio = 0.4
    ra = 2
    # save_box_to_txt(ver_lines_heng, "D:\\BaiduNetdiskDownload\\post0\\0910\\data\\\\ver_lines_heng.txt")
    # save_box_to_txt(ver_lines_shu, "D:\\BaiduNetdiskDownload\\post0\\0910\\data\\\\ver_lines_shu.txt")
    print("开始视图**")
    
    for i in range(len(pairs)):
        print("一组pairs开始*")
        print(f'开始{pairs[i]}判断')
        print(f'横{heng}')
        print(f'竖{shu}')
        
        if pairs[i][4] == 0:  # 外向标尺线
            print("外向")
            ratio = 0.3  # 增加ratio，让搜索范围更大
            
            if (pairs[i][2] - pairs[i][0]) > (pairs[i][3] - pairs[i][1]):  # 横向标尺线
                print("横向")
                left_straight = np.zeros((0, 6))  # 增加一列存储评分
                right_straight = np.zeros((0, 6))
                middle = np.zeros(6)
                
                for j in range(len(ver_lines_shu)):
                    # 简化：主要根据水平距离匹配
                    left_distance = abs(ver_lines_shu[j][0] - pairs[i][0])
                    right_distance = abs(ver_lines_shu[j][0] - pairs[i][2])
                    
                    # 左引线匹配条件
                    if left_distance < 50:  # 距离小于50像素
                        # 检查Y方向是否有重叠或接近
                        y_overlap_start = max(pairs[i][1], ver_lines_shu[j][1])
                        y_overlap_end = min(pairs[i][3], ver_lines_shu[j][3])
                        y_overlap = max(0, y_overlap_end - y_overlap_start)
                        
                        if y_overlap > 0 or min(abs(pairs[i][1] - ver_lines_shu[j][3]), 
                                                abs(pairs[i][3] - ver_lines_shu[j][1])) < 20:
                            # 计算简化评分
                            left_score = calculate_match_score_simple(pairs[i], ver_lines_shu[j], heng, is_left=True)
                            
                            if left_score > 0:
                                middle[0:4] = ver_lines_shu[j]
                                middle[4] = left_distance  # 存储距离
                                middle[5] = left_score  # 存储评分
                                left_straight = np.r_[left_straight, [middle]]
                                print(f"外向横pairs找到左竖线，距离: {left_distance}, 评分: {left_score}")
                    
                    # 右引线匹配条件
                    if right_distance < 50:  # 距离小于50像素
                        # 检查Y方向是否有重叠或接近
                        y_overlap_start = max(pairs[i][1], ver_lines_shu[j][1])
                        y_overlap_end = min(pairs[i][3], ver_lines_shu[j][3])
                        y_overlap = max(0, y_overlap_end - y_overlap_start)
                        
                        if y_overlap > 0 or min(abs(pairs[i][1] - ver_lines_shu[j][3]), 
                                                abs(pairs[i][3] - ver_lines_shu[j][1])) < 20:
                            # 计算简化评分
                            right_score = calculate_match_score_simple(pairs[i], ver_lines_shu[j], heng, is_left=False)
                            
                            if right_score > 0:
                                middle[0:4] = ver_lines_shu[j]
                                middle[4] = right_distance  # 存储距离
                                middle[5] = right_score  # 存储评分
                                right_straight = np.r_[right_straight, [middle]]
                                print(f"外向横pairs找到右竖线，距离: {right_distance}, 评分: {right_score}")
                
                # 按距离从小到大排序（主要排序依据），距离相同再按评分排序
                if len(left_straight) > 0:
                    # 先按距离排序，距离小的在前
                    left_straight = left_straight[np.argsort(left_straight[:, 4])]
                    # 如果前几个距离相同，再按评分排序
                    # 简单实现：取距离最小的几个，再按评分排序
                    min_distance = left_straight[0, 4]
                    same_distance_indices = np.where(left_straight[:, 4] <= min_distance + 1)[0]
                    if len(same_distance_indices) > 1:
                        # 对距离相同的部分按评分排序
                        same_distance_lines = left_straight[same_distance_indices]
                        same_distance_lines = same_distance_lines[np.argsort(-same_distance_lines[:, 5])]
                        left_straight[same_distance_indices] = same_distance_lines
                
                if len(right_straight) > 0:
                    # 先按距离排序，距离小的在前
                    right_straight = right_straight[np.argsort(right_straight[:, 4])]
                    # 如果前几个距离相同，再按评分排序
                    min_distance = right_straight[0, 4]
                    same_distance_indices = np.where(right_straight[:, 4] <= min_distance + 1)[0]
                    if len(same_distance_indices) > 1:
                        # 对距离相同的部分按评分排序
                        same_distance_lines = right_straight[same_distance_indices]
                        same_distance_lines = same_distance_lines[np.argsort(-same_distance_lines[:, 5])]
                        right_straight[same_distance_indices] = same_distance_lines
                
                # 根据找到的引线数量处理
                if len(left_straight) > 0 and len(right_straight) > 0:
                    # 找到两条引线，直接使用距离最小的
                    pairs_length_middle[0:4] = pairs[i, 0:4]
                    pairs_length_middle[4:8] = left_straight[0, 0:4]
                    pairs_length_middle[8:12] = right_straight[0, 0:4]
                    pairs_length_middle[12] = abs(left_straight[0, 0] - right_straight[0, 0])
                    pairs_length = np.r_[pairs_length, [pairs_length_middle]]
                    print(f"找到两条引线，左距离: {left_straight[0, 4]}, 右距离: {right_straight[0, 4]}")
                elif len(left_straight) > 0 or len(right_straight) > 0:
                    # 只找到一条引线，生成另一条
                    pairs_length_middle[0:4] = pairs[i, 0:4]
                    
                    # 如果找到左侧引线
                    if len(left_straight) > 0:
                        pairs_length_middle[4:8] = left_straight[0, 0:4]
                        left_line = left_straight[0, 0:4]
                        
                        # 计算左侧引线与箭头对的距离
                        left_distance = left_line[0] - pairs[i][0]
                        
                        # 生成右侧引线（与左侧对称）
                        right_line = [
                            pairs[i][2] - abs(left_distance),  # 保持与左侧对称的距离
                            left_line[1],  # y1 - 保持与左侧引线相同的高度
                            pairs[i][2] - abs(left_distance),  # x2 - 保持与左侧对称的距离
                            left_line[3]   # y2 - 保持与左侧引线相同的高度
                        ]
                        pairs_length_middle[8:12] = right_line
                        print("根据左侧引线生成右侧引线")
                    
                    # 如果找到右侧引线
                    elif len(right_straight) > 0:
                        pairs_length_middle[8:12] = right_straight[0, 0:4]
                        right_line = right_straight[0, 0:4]
                        
                        # 计算右侧引线与箭头对的距离
                        right_distance = right_line[0] - pairs[i][2]
                        
                        # 生成左侧引线（与右侧对称）
                        left_line = [
                            pairs[i][0] + abs(right_distance),  # 保持与右侧对称的距离
                            right_line[1],  # y1 - 保持与右侧引线相同的高度
                            pairs[i][0] + abs(right_distance),  # x2 - 保持与右侧对称的距离
                            right_line[3]   # y2 - 保持与右侧引线相同的高度
                        ]
                        pairs_length_middle[4:8] = left_line
                        print("根据右侧引线生成左侧引线")
                    
                    # 计算距离
                    pairs_length_middle[12] = abs(pairs_length_middle[4] - pairs_length_middle[8])
                    pairs_length = np.r_[pairs_length, [pairs_length_middle]]
                else:
                    # 没有找到任何引线，不保存这个pairs
                    print("没有找到任何引线，跳过这个pairs")
            
            # 竖向标尺线的处理类似
            if (pairs[i][2] - pairs[i][0]) < (pairs[i][3] - pairs[i][1]):  # 竖向标尺线
                print("竖向")
                up_straight = np.zeros((0, 5))
                down_straight = np.zeros((0, 5))
                middle = np.zeros((5))
                
                for j in range(len(ver_lines_heng)):
                    # 上端附近的直线
                    if pairs[i][1] - ratio * (pairs[i][3] - pairs[i][1]) < ver_lines_heng[j][1] < pairs[i][1] + ratio * (
                            pairs[i][3] - pairs[i][1]):
                        if (not (pairs[i][0] > ver_lines_heng[j][2] or pairs[i][2] < ver_lines_heng[j][0])) or min(
                                abs(pairs[i][0] - ver_lines_heng[j][2]),
                                abs(pairs[i][2] - ver_lines_heng[j][0])) < ra * (
                                pairs[i][2] - pairs[i][0]):
                            if (pairs[i][0] > shu and ver_lines_heng[j][0] < pairs[i][0]) or (
                                    pairs[i][0] < shu and ver_lines_heng[j][2] > pairs[i][2]):
                                middle[0:4] = ver_lines_heng[j]
                                middle[4] = abs(pairs[i][1] - ver_lines_heng[j][1])
                                up_straight = np.r_[up_straight, [middle]]
                                print("外向竖pairs找到上横线")

                    # 下端附近的直线
                    if pairs[i][3] - ratio * (pairs[i][3] - pairs[i][1]) < ver_lines_heng[j][1] < pairs[i][3] + ratio * (
                            pairs[i][3] - pairs[i][1]):
                        if (not (pairs[i][0] > ver_lines_heng[j][2] or pairs[i][2] < ver_lines_heng[j][0])) or min(
                                abs(pairs[i][0] - ver_lines_heng[j][2]),
                                abs(pairs[i][2] - ver_lines_heng[j][0])) < ra * (
                                pairs[i][2] - pairs[i][0]):
                            if (pairs[i][0] > shu and ver_lines_heng[j][0] < pairs[i][0]) or (
                                    pairs[i][0] < shu and ver_lines_heng[j][2] > pairs[i][2]):
                                middle[0:4] = ver_lines_heng[j]
                                middle[4] = abs(pairs[i][0] - ver_lines_heng[j][0])
                                down_straight = np.r_[down_straight, [middle]]
                                print("外向竖pairs找到下横线")
                
                up_straight = up_straight[np.argsort(up_straight[:, 4])]
                down_straight = down_straight[np.argsort(down_straight[:, 4])]
                
                # 修改：根据找到的引线数量处理
                if len(up_straight) > 0 and len(down_straight) > 0:
                    pairs_length_middle[0:4] = pairs[i, 0:4]
                    pairs_length_middle[4:8] = up_straight[0, 0:4]
                    pairs_length_middle[8:12] = down_straight[0, 0:4]
                    pairs_length_middle[12] = abs(up_straight[0, 1] - down_straight[0, 1])
                    pairs_length = np.r_[pairs_length, [pairs_length_middle]]
                    print("找到两条引线，直接使用")
                elif len(up_straight) > 0 or len(down_straight) > 0:
                    pairs_length_middle[0:4] = pairs[i, 0:4]
                    
                    if len(up_straight) > 0:
                        pairs_length_middle[4:8] = up_straight[0, 0:4]
                        up_line = up_straight[0, 0:4]
                        
                        up_distance = up_line[1] - pairs[i][1]
                        
                        # 生成下侧引线（与上侧对称）
                        down_line = [
                            up_line[0],
                            pairs[i][3] - abs(up_distance),  # 保持与上侧对称的距离
                            up_line[2],
                            pairs[i][3] - abs(up_distance)
                        ]
                        pairs_length_middle[8:12] = down_line
                        print("根据上侧引线生成下侧引线")
                    
                    elif len(down_straight) > 0:
                        pairs_length_middle[8:12] = down_straight[0, 0:4]
                        down_line = down_straight[0, 0:4]
                        
                        down_distance = down_line[1] - pairs[i][3]
                        
                        # 生成上侧引线（与下侧对称）
                        up_line = [
                            down_line[0],
                            pairs[i][1] + abs(down_distance),  # 保持与下侧对称的距离
                            down_line[2],
                            pairs[i][1] + abs(down_distance)
                        ]
                        pairs_length_middle[4:8] = up_line
                        print("根据下侧引线生成上侧引线")
                    
                    pairs_length_middle[12] = abs(pairs_length_middle[5] - pairs_length_middle[9])
                    pairs_length = np.r_[pairs_length, [pairs_length_middle]]
                else:
                    print("没有找到任何引线，跳过这个pairs")
        
        # 内向标尺线的处理（保持原有逻辑）
        # 在内向标尺线的处理部分添加对称性逻辑
        if pairs[i][4] == 1:  # 内向标尺线
            print("内向")
            ratio = 0.5
            ratio_inside = 0.15
            
            if (pairs[i][2] - pairs[i][0]) > (pairs[i][3] - pairs[i][1]):  # 横向标尺线
                print("横向")
                left_straight = np.zeros((0, 6))  # 改为6列，存储[x1,y1,x2,y2,距离,评分]
                right_straight = np.zeros((0, 6))
                middle = np.zeros(6)
                
                for j in range(len(ver_lines_shu)):
                    # 使用简化评分函数为内向箭头也计算评分
                    # 左端附近的直线
                    if pairs[i][0] - ratio_inside * (pairs[i][2] - pairs[i][0]) < ver_lines_shu[j][0] < pairs[i][0] + ratio * (
                            pairs[i][2] - pairs[i][0]):
                        # 计算简化评分
                        left_score = calculate_match_score_simple(pairs[i], ver_lines_shu[j], heng, is_left=True)
                        
                        if left_score > 0:  # 使用评分而不是原始条件
                            middle[0:4] = ver_lines_shu[j]
                            middle[4] = abs(pairs[i][0] - ver_lines_shu[j][0])
                            middle[5] = left_score  # 存储评分
                            left_straight = np.r_[left_straight, [middle]]
                            print(f"内向横pairs找到左竖线，距离: {middle[4]}, 评分: {left_score}")

                    # 右端附近的直线
                    if pairs[i][2] - ratio * (pairs[i][2] - pairs[i][0]) < ver_lines_shu[j][0] < pairs[i][2] + ratio_inside * (pairs[i][2] - pairs[i][0]):
                        # 计算简化评分
                        right_score = calculate_match_score_simple(pairs[i], ver_lines_shu[j], heng, is_left=False)
                        
                        if right_score > 0:  # 使用评分而不是原始条件
                            middle[0:4] = ver_lines_shu[j]
                            middle[4] = abs(pairs[i][2] - ver_lines_shu[j][0])
                            middle[5] = right_score  # 存储评分
                            right_straight = np.r_[right_straight, [middle]]
                            print(f"内向横pairs找到右竖线，距离: {middle[4]}, 评分: {right_score}")
                
                # 关键修改：寻找最对称的引线对
                if len(left_straight) > 0 and len(right_straight) > 0:
                    best_left_idx = 0
                    best_right_idx = 0
                    best_symmetry_score = float('inf')
                    best_combined_score = 0
                    
                    # 限制搜索范围，取评分最高的前几个候选
                    left_candidates = left_straight[np.argsort(-left_straight[:, 5])[:3]]  # 取评分最高的3个
                    right_candidates = right_straight[np.argsort(-right_straight[:, 5])[:3]]  # 取评分最高的3个
                    
                    for li, left_line in enumerate(left_candidates):
                        for ri, right_line in enumerate(right_candidates):
                            # 计算对称性：左右引线到各自箭头的距离应该接近
                            left_distance = left_line[0] - pairs[i][0]  # 左引线到左箭头的距离
                            right_distance = pairs[i][2] - right_line[0]  # 右箭头到右引线的距离
                            
                            # 对称性评分：距离差的绝对值越小越好
                            symmetry = abs(abs(left_distance) - abs(right_distance))
                            
                            # 综合评分：对称性 + 单个引线的评分
                            combined_score = (left_line[5] + right_line[5]) * 0.5 - symmetry * 2
                            
                            if combined_score > best_combined_score:
                                best_combined_score = combined_score
                                best_left_idx = li
                                best_right_idx = ri
                                best_symmetry_score = symmetry
                    
                    print(f"最佳对称性评分: {best_symmetry_score}, 综合评分: {best_combined_score}")
                    
                    pairs_length_middle[0:4] = pairs[i, 0:4]
                    pairs_length_middle[4:8] = left_candidates[best_left_idx, 0:4]
                    pairs_length_middle[8:12] = right_candidates[best_right_idx, 0:4]
                    pairs_length_middle[12] = abs(left_candidates[best_left_idx, 0] - right_candidates[best_right_idx, 0])
                    pairs_length = np.r_[pairs_length, [pairs_length_middle]]
                    print("找到两条引线，使用对称性最佳的配对")
                    
                elif len(left_straight) > 0 or len(right_straight) > 0:
                    # 只有一侧找到引线的处理（保持原有逻辑）
                    pairs_length_middle[0:4] = pairs[i, 0:4]
                    
                    if len(left_straight) > 0:
                        # 按综合评分排序：距离和评分的加权
                        left_straight = left_straight[np.argsort(-left_straight[:, 5])]  # 按评分排序
                        pairs_length_middle[4:8] = left_straight[0, 0:4]
                        left_line = left_straight[0, 0:4]
                        
                        left_distance = left_line[0] - pairs[i][0]
                        
                        # 生成右侧引线（保持对称）
                        right_line = [
                            pairs[i][2] - abs(left_distance),
                            left_line[1],
                            pairs[i][2] - abs(left_distance),
                            left_line[3]
                        ]
                        pairs_length_middle[8:12] = right_line
                        print("根据左侧引线生成右侧引线（内向）")
                    
                    elif len(right_straight) > 0:
                        # 按综合评分排序
                        right_straight = right_straight[np.argsort(-right_straight[:, 5])]  # 按评分排序
                        pairs_length_middle[8:12] = right_straight[0, 0:4]
                        right_line = right_straight[0, 0:4]
                        
                        right_distance = pairs[i][2] - right_line[0]
                        
                        # 生成左侧引线（保持对称）
                        left_line = [
                            pairs[i][0] + abs(right_distance),
                            right_line[1],
                            pairs[i][0] + abs(right_distance),
                            right_line[3]
                        ]
                        pairs_length_middle[4:8] = left_line
                        print("根据右侧引线生成左侧引线（内向）")
                    
                    pairs_length_middle[12] = abs(pairs_length_middle[4] - pairs_length_middle[8])
                    pairs_length = np.r_[pairs_length, [pairs_length_middle]]
                else:
                    print("没有找到任何引线，跳过这个pairs（内向）")
            #####
            # 在竖向标尺线的处理部分也加入对称性逻辑
            if (pairs[i][2] - pairs[i][0]) < (pairs[i][3] - pairs[i][1]):  # 竖向标尺线
                print("竖向")
                up_straight = np.zeros((0, 6))  # 改为6列，存储[x1,y1,x2,y2,距离,评分]
                down_straight = np.zeros((0, 6))
                middle = np.zeros(6)
                
                for j in range(len(ver_lines_heng)):
                    # 使用评分函数计算上端引线
                    # 上端附近的直线
                    if pairs[i][1] + ratio_inside * (pairs[i][3] - pairs[i][1]) < ver_lines_heng[j][1] < pairs[i][1] + ratio * (
                            pairs[i][3] - pairs[i][1]):
                        
                        # 创建临时竖直线表示，用于评分计算
                        # 对于竖直方向的箭头，我们需要转换为水平引线的评分
                        # 使用类似逻辑但方向不同
                        temp_vertical_line = [
                            (ver_lines_heng[j][0] + ver_lines_heng[j][2]) / 2,  # x中心
                            ver_lines_heng[j][1],  # y1
                            (ver_lines_heng[j][0] + ver_lines_heng[j][2]) / 2,  # x中心
                            ver_lines_heng[j][3]   # y2
                        ]
                        
                        # 计算上端评分
                        up_score = calculate_match_score_simple(pairs[i], temp_vertical_line, heng, is_top=True)
                        
                        if up_score > 0:
                            middle[0:4] = ver_lines_heng[j]
                            middle[4] = abs(pairs[i][1] - ver_lines_heng[j][1])
                            middle[5] = up_score  # 存储评分
                            up_straight = np.r_[up_straight, [middle]]
                            print(f"内向竖pairs找到上横线，距离: {middle[4]}, 评分: {up_score}")

                    # 下端附近的直线
                    if pairs[i][3] - ratio * (pairs[i][3] - pairs[i][1]) < ver_lines_heng[j][1] < pairs[i][3] - ratio_inside * (
                            pairs[i][3] - pairs[i][1]):
                        
                        # 创建临时竖直线表示，用于评分计算
                        temp_vertical_line = [
                            (ver_lines_heng[j][0] + ver_lines_heng[j][2]) / 2,  # x中心
                            ver_lines_heng[j][1],  # y1
                            (ver_lines_heng[j][0] + ver_lines_heng[j][2]) / 2,  # x中心
                            ver_lines_heng[j][3]   # y2
                        ]
                        
                        # 计算下端评分
                        down_score = calculate_match_score_simple(pairs[i], temp_vertical_line, heng, is_top=False)
                        
                        if down_score > 0:
                            middle[0:4] = ver_lines_heng[j]
                            middle[4] = abs(pairs[i][3] - ver_lines_heng[j][1])
                            middle[5] = down_score  # 存储评分
                            down_straight = np.r_[down_straight, [middle]]
                            print(f"内向竖pairs找到下横线，距离: {middle[4]}, 评分: {down_score}")
                
                # 关键修改：寻找最对称的引线对
                if len(up_straight) > 0 and len(down_straight) > 0:
                    best_up_idx = 0
                    best_down_idx = 0
                    best_symmetry_score = float('inf')
                    best_combined_score = 0
                    
                    # 限制搜索范围，取评分最高的前几个候选
                    up_candidates = up_straight[np.argsort(-up_straight[:, 5])[:3]]  # 取评分最高的3个
                    down_candidates = down_straight[np.argsort(-down_straight[:, 5])[:3]]  # 取评分最高的3个
                    
                    for ui, up_line in enumerate(up_candidates):
                        for di, down_line in enumerate(down_candidates):
                            # 计算对称性：上下引线到各自箭头的距离应该接近
                            up_distance = up_line[1] - pairs[i][1]  # 上引线到上箭头的距离
                            down_distance = pairs[i][3] - down_line[1]  # 下箭头到下引线的距离
                            
                            # 对称性评分：距离差的绝对值越小越好
                            symmetry = abs(abs(up_distance) - abs(down_distance))
                            
                            # 综合评分：对称性 + 单个引线的评分
                            combined_score = (up_line[5] + down_line[5]) * 0.5 - symmetry * 2
                            
                            if combined_score > best_combined_score:
                                best_combined_score = combined_score
                                best_up_idx = ui
                                best_down_idx = di
                                best_symmetry_score = symmetry
                    
                    print(f"最佳对称性评分: {best_symmetry_score}, 综合评分: {best_combined_score}")
                    
                    pairs_length_middle[0:4] = pairs[i, 0:4]
                    pairs_length_middle[4:8] = up_candidates[best_up_idx, 0:4]
                    pairs_length_middle[8:12] = down_candidates[best_down_idx, 0:4]
                    pairs_length_middle[12] = abs(up_candidates[best_up_idx, 1] - down_candidates[best_down_idx, 1])
                    pairs_length = np.r_[pairs_length, [pairs_length_middle]]
                    print("找到两条引线，使用对称性最佳的配对")
                    
                elif len(up_straight) > 0 or len(down_straight) > 0:
                    # 只有一侧找到引线的处理
                    pairs_length_middle[0:4] = pairs[i, 0:4]
                    
                    if len(up_straight) > 0:
                        # 按综合评分排序：距离和评分的加权
                        up_straight = up_straight[np.argsort(-up_straight[:, 5])]  # 按评分排序
                        pairs_length_middle[4:8] = up_straight[0, 0:4]
                        up_line = up_straight[0, 0:4]
                        
                        up_distance = up_line[1] - pairs[i][1]
                        
                        # 生成下侧引线（保持对称）
                        down_line = [
                            up_line[0],
                            pairs[i][3] - abs(up_distance),
                            up_line[2],
                            pairs[i][3] - abs(up_distance)
                        ]
                        pairs_length_middle[8:12] = down_line
                        print("根据上侧引线生成下侧引线（内向）")
                    
                    elif len(down_straight) > 0:
                        # 按综合评分排序
                        down_straight = down_straight[np.argsort(-down_straight[:, 5])]  # 按评分排序
                        pairs_length_middle[8:12] = down_straight[0, 0:4]
                        down_line = down_straight[0, 0:4]
                        
                        down_distance = pairs[i][3] - down_line[1]
                        
                        # 生成上侧引线（保持对称）
                        up_line = [
                            down_line[0],
                            pairs[i][1] + abs(down_distance),
                            down_line[2],
                            pairs[i][1] + abs(down_distance)
                        ]
                        pairs_length_middle[4:8] = up_line
                        print("根据下侧引线生成上侧引线（内向）")
                    
                    pairs_length_middle[12] = abs(pairs_length_middle[5] - pairs_length_middle[9])
                    pairs_length = np.r_[pairs_length, [pairs_length_middle]]
                else:
                    print("没有找到任何引线，跳过这个pairs（内向）")
        
        print("一组pairs结束*")
    
    # 可视化部分保持不变
    if test_mode == 1:
        drawn_img = cv2.imread(img_path)
        for i in range(len(pairs_length)):
            x1 = int(pairs_length[i][4])
            x2 = int(pairs_length[i][6])
            y1 = int(pairs_length[i][5])
            y2 = int(pairs_length[i][7])
            drawn_img = cv2.line(drawn_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            x1 = int(pairs_length[i][8])
            x2 = int(pairs_length[i][10])
            y1 = int(pairs_length[i][9])
            y2 = int(pairs_length[i][11])
            drawn_img = cv2.line(drawn_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for i in range(len(pairs_length)):
            ptLeftTop = (int(pairs_length[i][0]), int(pairs_length[i][1]))
            ptRightBottom = (int(pairs_length[i][2]), int(pairs_length[i][3]))
            point_color = (0, 0, 255)
            thickness = 2
            lineType = 8
            cv2.rectangle(drawn_img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
        cv2.namedWindow("LSD", 0)
        cv2.imshow("LSD", drawn_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("视图结束**")
    
    try:
        drawn_img = cv2.imread(img_path)
        for i in range(len(pairs_length)):
            x1 = int(pairs_length[i][4])
            x2 = int(pairs_length[i][6])
            y1 = int(pairs_length[i][5])
            y2 = int(pairs_length[i][7])
            drawn_img = cv2.line(drawn_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            x1 = int(pairs_length[i][8])
            x2 = int(pairs_length[i][10])
            y1 = int(pairs_length[i][9])
            y2 = int(pairs_length[i][11])
            drawn_img = cv2.line(drawn_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for i in range(len(pairs_length)):
            ptLeftTop = (int(pairs_length[i][0]), int(pairs_length[i][1]))
            ptRightBottom = (int(pairs_length[i][2]), int(pairs_length[i][3]))
            point_color = (0, 0, 255)
            thickness = 2
            lineType = 8
            cv2.rectangle(drawn_img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
        path = fr'{OPENCV_OUTPUT_LINE}/' + img_path[-7:]
        cv2.imwrite(path, drawn_img)
        print("保存引线+标尺线组合成功:", path)
    except:
        print("保存引线+标尺线组合失败")
    
    print("***/结束引线和标尺线的匹配/***")
    # pairs_length = remove_duplicate_pairs(pairs_length)
    # pairs_length = remove_contained_arrows(pairs_length)
    return pairs_length


def remove_duplicate_pairs(pairs_length):
    """
    去除使用相同引线组合的重复箭头对，只保留引线到箭头对框距离最近的那个

    参数:
    pairs_length: np.二维数组 (n, 13) [pairs_x1_y1_x2_y2, 引线1_x1_y1_x2_y2, 引线2_x1_y1_x2_y2, 两引线距离]

    返回:
    去重后的 pairs_length
    """
    print("开始检查重复引线...")

    # 创建一个字典来存储引线组合和对应的箭头对信息
    line_pairs_dict = {}

    for i in range(len(pairs_length)):
        # 提取两条引线的坐标，作为唯一标识
        line1_key = tuple(pairs_length[i][4:8].astype(int))  # 引线1的坐标
        line2_key = tuple(pairs_length[i][8:12].astype(int))  # 引线2的坐标

        # 创建引线组合的唯一标识符（不考虑顺序）
        line_pair_key = tuple(sorted([line1_key, line2_key]))

        # 计算引线到箭头对框的距离
        arrow_box = [pairs_length[i][0], pairs_length[i][1], pairs_length[i][2], pairs_length[i][3]]

        # 对于每条引线，计算到箭头对框的最小距离
        line1_distance = calculate_line_to_box_distance(
            [pairs_length[i][4], pairs_length[i][5], pairs_length[i][6], pairs_length[i][7]],
            arrow_box
        )

        line2_distance = calculate_line_to_box_distance(
            [pairs_length[i][8], pairs_length[i][9], pairs_length[i][10], pairs_length[i][11]],
            arrow_box
        )

        # 使用两条引线到箭头对框的平均距离作为总距离
        distance = (line1_distance + line2_distance) / 2

        # 如果这个引线组合已经存在，比较距离
        if line_pair_key in line_pairs_dict:
            existing_distance, existing_index = line_pairs_dict[line_pair_key]
            if distance < existing_distance:
                # 当前箭头对距离更近，更新字典
                line_pairs_dict[line_pair_key] = (distance, i)
                print(f"发现重复引线组合，更新为距离更近的箭头对，距离: {distance:.2f}")
            else:
                print(f"发现重复引线组合，保留原有箭头对，距离: {existing_distance:.2f}")
        else:
            # 新的引线组合，添加到字典
            line_pairs_dict[line_pair_key] = (distance, i)

    # 构建去重后的结果
    if len(line_pairs_dict) < len(pairs_length):
        print(f"去重前: {len(pairs_length)} 个箭头对，去重后: {len(line_pairs_dict)} 个箭头对")

        unique_indices = [item[1] for item in line_pairs_dict.values()]
        pairs_length = pairs_length[unique_indices]
    else:
        print("没有发现重复的引线组合")

    return pairs_length


def calculate_line_to_box_distance(line, box):
    """
    计算直线到矩形框的最小距离
    line: [x1, y1, x2, y2] 直线端点坐标
    box: [x1, y1, x2, y2] 矩形框左上角和右下角坐标
    返回: 直线到矩形框的最小距离
    """
    # 提取直线端点
    line_p1 = (line[0], line[1])
    line_p2 = (line[2], line[3])

    # 提取矩形框的四个角点
    box_p1 = (box[0], box[1])  # 左上角
    box_p2 = (box[2], box[1])  # 右上角
    box_p3 = (box[2], box[3])  # 右下角
    box_p4 = (box[0], box[3])  # 左下角

    # 计算直线到矩形框四条边的最小距离
    distances = []

    # 直线到矩形框上边的距离
    distances.append(line_segment_distance(line_p1, line_p2, box_p1, box_p2))
    # 直线到矩形框右边的距离
    distances.append(line_segment_distance(line_p1, line_p2, box_p2, box_p3))
    # 直线到矩形框下边的距离
    distances.append(line_segment_distance(line_p1, line_p2, box_p3, box_p4))
    # 直线到矩形框左边的距离
    distances.append(line_segment_distance(line_p1, line_p2, box_p4, box_p1))

    return min(distances)


def line_segment_distance(p1, p2, p3, p4):
    """
    计算两条线段之间的最短距离
    p1, p2: 第一条线段的两个端点
    p3, p4: 第二条线段的两个端点
    """
    # 计算两条线段端点之间的距离
    d1 = point_to_line_segment_distance(p1, p3, p4)
    d2 = point_to_line_segment_distance(p2, p3, p4)
    d3 = point_to_line_segment_distance(p3, p1, p2)
    d4 = point_to_line_segment_distance(p4, p1, p2)

    return min(d1, d2, d3, d4)


def point_to_line_segment_distance(point, line_p1, line_p2):
    """
    计算点到线段的最短距离
    point: (x, y) 点坐标
    line_p1, line_p2: (x, y) 线段的两个端点
    """
    x, y = point
    x1, y1 = line_p1
    x2, y2 = line_p2

    # 线段长度的平方
    l2 = (x2 - x1) ** 2 + (y2 - y1) ** 2

    if l2 == 0:  # 线段退化为点
        return ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5

    # 计算投影比例 t
    t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / l2))

    # 计算投影点坐标
    projection_x = x1 + t * (x2 - x1)
    projection_y = y1 + t * (y2 - y1)

    # 返回点到投影点的距离
    return ((x - projection_x) ** 2 + (y - projection_y) ** 2) ** 0.5

def get_better_data_1(top_yolox_pairs, bottom_yolox_pairs, side_yolox_pairs, detailed_yolox_pairs, key, top_dbnet_data, bottom_dbnet_data, side_dbnet_data, detailed_dbnet_data):
    # 去除标尺线数据中标记着标尺线内外向的数据
    top_yolox_pairs_copy = top_yolox_pairs.copy()
    bottom_yolox_pairs_copy = bottom_yolox_pairs.copy()
    side_yolox_pairs_copy = side_yolox_pairs.copy()
    detailed_yolox_pairs_copy = detailed_yolox_pairs.copy()

    # 转换为 NumPy 数组
    top_yolox_pairs_np = np.array(top_yolox_pairs)
    bottom_yolox_pairs_np = np.array(bottom_yolox_pairs)
    side_yolox_pairs_np = np.array(side_yolox_pairs)
    detailed_yolox_pairs_np = np.array(detailed_yolox_pairs)

    # 如果 size 为 0 (即空列表)，则创建一个 (0, 4) 的空 2D 数组；否则正常切片
    if top_yolox_pairs_np.size == 0:top_yolox_pairs = np.empty((0, 4))
    else:top_yolox_pairs = top_yolox_pairs_np[:, 0:4]
    if bottom_yolox_pairs_np.size == 0:bottom_yolox_pairs = np.empty((0, 4))
    else:bottom_yolox_pairs = bottom_yolox_pairs_np[:, 0:4]
    if side_yolox_pairs_np.size == 0:side_yolox_pairs = np.empty((0, 4))
    else:side_yolox_pairs = side_yolox_pairs_np[:, 0:4]
    if detailed_yolox_pairs_np.size == 0:detailed_yolox_pairs = np.empty((0, 4))
    else:detailed_yolox_pairs = detailed_yolox_pairs_np[:, 0:4]
    if key == 1:
        print("展示dbnet检测的文本（删除other）top_dbnet_data")
        img_path = f"{DATA}/top.jpg"
        show_data(img_path, top_dbnet_data)  # 展示dbnet框选的尺寸数字
        print("展示dbnet检测的文本（删除other，标记行列序号的数字和字母标注）bottom_dbnet_data")
        img_path = f"{DATA}/bottom.jpg"
        show_data(img_path, bottom_dbnet_data)  # 展示dbnet框选的尺寸数字
        print("展示dbnet检测的文本（删除other）side_dbnet_data")
        img_path = f"{DATA}/side.jpg"
        show_data(img_path, side_dbnet_data)  # 展示dbnet框选的尺寸数字
        img_path = f"{DATA}/detailed.jpg"
        show_data(img_path, detailed_dbnet_data)  # 展示dbnet框选的尺寸数字

    top_dbnet_data_all = top_dbnet_data.copy()
    bottom_dbnet_data_all = bottom_dbnet_data.copy()
    return top_yolox_pairs, bottom_yolox_pairs, side_yolox_pairs, detailed_yolox_pairs, top_yolox_pairs_copy, bottom_yolox_pairs_copy, side_yolox_pairs_copy, detailed_yolox_pairs_copy, top_dbnet_data_all, bottom_dbnet_data_all

#1202优化之后
def process_view_ocr(view_name, dbnet_data):
    """辅助函数：处理单个视图的 OCR"""
    path = f"{DATA}/{view_name}.jpg"
    if not os.path.exists(path):
        return []
    return ocr_data(path, dbnet_data)
def SVTR(top_dbnet_data_all, bottom_dbnet_data_all, side_dbnet_data, detailed_dbnet_data):
    print("---开始各个视图的SVTR识别 (并行优化版)---")
    start1 = time.time()

    # 定义任务列表
    tasks = {
        'top': top_dbnet_data_all,
        'bottom': bottom_dbnet_data_all,
        'side': side_dbnet_data,
        'detailed': detailed_dbnet_data
    }

    results = {}

    # 使用线程池并行处理 (IO密集型或ONNXRuntime释放GIL时有效)
    # max_workers=4 对应 4 个视图
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_view = {
            executor.submit(process_view_ocr, view, data): view
            for view, data in tasks.items()
        }

        for future in future_to_view:
            view = future_to_view[future]
            try:
                data = future.result()
                results[view] = data
            except Exception as e:
                print(f"视图 {view} OCR 识别出错: {e}")
                results[view] = []

    print("---结束各个视图的SVTR识别---")
    end = time.time()

    return start1, end, results['top'], results['bottom'], results['side'], results['detailed']

class OCRDataProcessor:
    def __init__(self):
        # 预编译正则，提高效率
        self.re_comma = re.compile(r"[,，]")
        self.re_nx = re.compile(r"(\d+\.?\d*)[Xx]")
        self.re_eq = re.compile(r"=")
        self.re_pin = re.compile(r"[Pp][Ii][Nn]1*", re.IGNORECASE)
        self.re_a1 = re.compile(r"[Aa]1")
        self.re_note = re.compile(r"[Nn][Oo][Tt][Ee][2-9]", re.IGNORECASE)
        self.re_single_char = re.compile(r"^[AaBbCcDd]$")
        self.re_angle = re.compile(r"°")
        # 核心提取正则
        self.re_key_info = re.compile(r"\d+(?:\.\d+)?|=|\+|-|Φ|±|[Mm][Aa][Xx]|[Nn][Oo][Mm]|[Mm][Ii][Nn]|[Xx*]")

    def process_pipeline(self, ocr_data_list):
        """
        主处理流水线：接收原始 OCR 列表，返回清洗后的列表
        """
        cleaned_data = []
        for item in ocr_data_list:
            # 1. 基础判空
            if not item.get('ocr_strings'):
                continue

            # --- 第一阶段：字符串清洗 ---
            text = item['ocr_strings']

            # 符号修正
            text = self.re_comma.sub('.', text)

            # 数量标识处理 (Nx)
            if not self.re_eq.search(text):
                if self.re_nx.search(text):
                    item['Absolutely'] = 'mb_pin_diameter'
                    text = self.re_nx.sub('', text)

            # 去噪
            text = self.re_pin.sub('', text)
            text = self.re_a1.sub('', text)
            text = self.re_note.sub('', text)
            if self.re_single_char.match(text):
                text = self.re_single_char.sub('', text)

            text = text.replace('O', '0').replace('o', '0').replace('±-','+-').replace('.00','')

            # 更新清洗后的文本
            item['ocr_strings'] = text.strip()
            if not item['ocr_strings']:  # 如果清洗完没剩东西了
                continue

            # --- 第二阶段：关键信息提取 ---
            if self.re_angle.search(text):
                item['Absolutely'] = 'angle'

            # 提取 raw tokens
            tokens = [x.strip() for x in self.re_key_info.findall(text) if x.strip()]

            # 处理 '±' 的特殊逻辑 (保留前一数字、符号、后一数字、Φ)
            if '±' in tokens:
                try:
                    idx = tokens.index('±')
                    if idx > 0 and idx + 1 < len(tokens):
                        # 保留 [前数, ±, 后数] + 所有 Φ
                        new_tokens = tokens[idx - 1: idx + 2]
                        new_tokens.extend([t for t in tokens if t == 'Φ'])
                        tokens = new_tokens
                except:
                    pass  # 保持原样

            # --- 第三阶段：Token 清洗与修正 ---
            valid_tokens = []
            has_number = False

            for token in tokens:
                # 过滤点和x
                if token == '.' or (token.lower() == 'x' and len(tokens) == 1):
                    continue

                # 过滤无意义的0 (非角度时)
                if item.get('Absolutely') != 'angle' and token in ['0', '0.', '00']:
                    continue

                # 修复逻辑: 0开头且第二位非小数点的数字 (Bug Fix)
                # 原逻辑意图：05 -> 0.5
                if len(token) > 1 and token.startswith('0') and token[1] != '.':
                    try:
                        float(token)  # 确保是数字
                        token = '0.' + token[1:]  # 字符串切片重组
                    except ValueError:
                        pass

                # 检查是否为数字
                try:
                    float(token)
                    has_number = True
                except ValueError:
                    pass

                valid_tokens.append(token)

            # 只有包含有效数字的数据才保留
            if not has_number:
                continue

            item['key_info'] = valid_tokens
            if not item['key_info']:
                continue

            cleaned_data.append(item)

        return cleaned_data

    def filter_large_tolerance(self, ocr_data_list):
        """
        独立步骤：剔除公差过大的数据
        (需在计算完 max_medium_min 后调用)
        """
        final_data = []
        for item in ocr_data_list:
            # 假设 max_medium_min 格式为 [max, mid, min]
            mmm = item.get('max_medium_min')
            if mmm is None or len(mmm) < 3:
                # 如果没有计算该字段，暂时保留或按需处理
                final_data.append(item)
                continue

            is_angle = item.get('Absolutely') == 'angle'
            # 逻辑：(Max - Mid <= 1 且 Mid - Min <= 1) 或者 是角度
            if (abs(mmm[0] - mmm[1]) <= 1 and abs(mmm[1] - mmm[2]) <= 1) or is_angle:
                final_data.append(item)
            else:
                print(f"删除公差特别大的标注: {mmm}")
        return final_data
def data_wrangling_optimized(key, top_dbnet_data, bottom_dbnet_data, side_dbnet_data, detailed_dbnet_data,
                             top_ocr_data, bottom_ocr_data, side_ocr_data, detailed_ocr_data,
                             top_yolox_num, bottom_yolox_num, side_yolox_num, detailed_yolox_num):
    # 初始化处理器
    processor = OCRDataProcessor()

    # 1. 转换字典格式
    top_ocr_data = convert_Dic(top_dbnet_data, top_ocr_data)
    bottom_ocr_data = convert_Dic(bottom_dbnet_data, bottom_ocr_data)
    side_ocr_data = convert_Dic(side_dbnet_data, side_ocr_data)
    detailed_ocr_data = convert_Dic(detailed_dbnet_data, detailed_ocr_data)

    # 2. 统一执行清洗流水线 (替代了 filter_ocr_data_0 到 filter_ocr_data__2 的所有调用)
    top_ocr_data = processor.process_pipeline(top_ocr_data)
    bottom_ocr_data = processor.process_pipeline(bottom_ocr_data)
    side_ocr_data = processor.process_pipeline(side_ocr_data)
    detailed_ocr_data = processor.process_pipeline(detailed_ocr_data)

    tee_print("经过预处理和初步提取后的结果 (TOP):\n", *top_ocr_data, sep='\n')
    tee_print("经过预处理得到的bottom视图的SVTR结果:\n", *bottom_ocr_data, sep='\n')
    tee_print("经过预处理得到的side视图的SVTR结果:\n", *side_ocr_data, sep='\n')
    tee_print("经过预处理得到的detailed视图的SVTR结果:\n", *detailed_ocr_data, sep='\n')

    
    print(f'side_yolox_num调试{side_yolox_num}')
    # 3. 数据绑定 (Bind Data - Logic remains external as it involves geometry)
    top_ocr_data = bind_data(top_yolox_num, top_ocr_data)
    bottom_ocr_data = bind_data(bottom_yolox_num, bottom_ocr_data)
    side_ocr_data = bind_data(side_yolox_num, side_ocr_data)
    detailed_ocr_data = bind_data(detailed_yolox_num, detailed_ocr_data)

    # 再次清洗：Bind Data 可能会引入新的杂乱数据，再次确保 key_info 包含数字
    # 原代码此处调用了 filter_ocr_data__2
    # 我们可以复用 pipeline，或者简单检查
    # 这里建议简单过滤即可

    # 4. 英寸清除与数值计算
    top_ocr_data = clear_inch(top_ocr_data)
    bottom_ocr_data = clear_inch(bottom_ocr_data)
    side_ocr_data = clear_inch(side_ocr_data)
    detailed_ocr_data = clear_inch(detailed_ocr_data)

    # 计算出各个标注的max_medium_min以及是否从符号上就可以判定该标注的语义
    top_ocr_data = cal_max_medium_min_top(top_ocr_data)
    bottom_ocr_data = cal_max_medium_min_bottom(bottom_ocr_data)
    side_ocr_data = cal_max_medium_min_side(side_ocr_data)
    detailed_ocr_data = cal_max_medium_min_side(detailed_ocr_data)

    # 5. 最后过滤大公差 (需要 max_medium_min 存在)
    top_ocr_data = processor.filter_large_tolerance(top_ocr_data)
    bottom_ocr_data = processor.filter_large_tolerance(bottom_ocr_data)
    side_ocr_data = processor.filter_large_tolerance(side_ocr_data)
    detailed_ocr_data = processor.filter_large_tolerance(detailed_ocr_data)

    # ... 后续的 BGA_side_filter 和 display 逻辑保持不变 ...
    side_ocr_data = BGA_side_filter(side_ocr_data)

    tee_print("经过处理后的结果 (TOP):\n", *top_ocr_data, sep='\n')
    tee_print("经过处理的bottom视图的SVTR结果:\n", *bottom_ocr_data, sep='\n')
    tee_print("经过处理的side视图的SVTR结果:\n", *side_ocr_data, sep='\n')
    tee_print("经过处理的detailed视图的SVTR结果:\n", *detailed_ocr_data, sep='\n')

    return top_ocr_data, bottom_ocr_data, side_ocr_data, detailed_ocr_data
#1201优化之前
def data_wrangling(key, top_dbnet_data, bottom_dbnet_data, side_dbnet_data, detailed_dbnet_data, top_ocr_data, bottom_ocr_data, side_ocr_data, detailed_ocr_data, top_yolox_num, bottom_yolox_num, side_yolox_num, detailed_yolox_num):
    # 编辑为字典类型
    top_ocr_data = convert_Dic(top_dbnet_data, top_ocr_data)
    # [{'location': array([1245., 88., 1302., 135.]), 'ocr_strings': ''},{'location': array([635., 110., 725., 156.]), 'ocr_strings': '15.0'}]
    bottom_ocr_data = convert_Dic(bottom_dbnet_data, bottom_ocr_data)
    side_ocr_data = convert_Dic(side_dbnet_data, side_ocr_data)
    detailed_ocr_data = convert_Dic(detailed_dbnet_data, detailed_ocr_data)

    # 筛除非参数信息
    # 删除SVTR识别为空
    top_ocr_data = filter_ocr_data_0(top_ocr_data)
    bottom_ocr_data = filter_ocr_data_0(bottom_ocr_data)
    side_ocr_data = filter_ocr_data_0(side_ocr_data)
    detailed_ocr_data = filter_ocr_data_0(detailed_ocr_data)

    # 逗号改为.
    top_ocr_data = filter_ocr_data_4(top_ocr_data)
    bottom_ocr_data = filter_ocr_data_4(bottom_ocr_data)
    side_ocr_data = filter_ocr_data_4(side_ocr_data)
    detailed_ocr_data = filter_ocr_data_4(detailed_ocr_data)

    # 删除’数字‘+’X‘（没有’=‘时）并标记'Absolutely'为'mb_pin_diameter'
    top_ocr_data = filter_ocr_data_1(top_ocr_data)
    bottom_ocr_data = filter_ocr_data_1(bottom_ocr_data)
    side_ocr_data = filter_ocr_data_1(side_ocr_data)
    detailed_ocr_data = filter_ocr_data_1(detailed_ocr_data)

    # 删除’PIN1‘
    top_ocr_data = filter_ocr_data_2(top_ocr_data)
    bottom_ocr_data = filter_ocr_data_2(bottom_ocr_data)
    side_ocr_data = filter_ocr_data_2(side_ocr_data)
    detailed_ocr_data = filter_ocr_data_2(detailed_ocr_data)
    # 删除’A1‘
    top_ocr_data = filter_ocr_data_3(top_ocr_data)
    bottom_ocr_data = filter_ocr_data_3(bottom_ocr_data)
    side_ocr_data = filter_ocr_data_3(side_ocr_data)
    detailed_ocr_data = filter_ocr_data_3(detailed_ocr_data)
    # 删除'note' + '数字'
    top_ocr_data = filter_ocr_data_11(top_ocr_data)
    bottom_ocr_data = filter_ocr_data_11(bottom_ocr_data)
    side_ocr_data = filter_ocr_data_11(side_ocr_data)
    detailed_ocr_data = filter_ocr_data_11(detailed_ocr_data)

    # 删除'A''B''C''D'
    top_ocr_data = filter_ocr_data_5(top_ocr_data)
    bottom_ocr_data = filter_ocr_data_5(bottom_ocr_data)
    side_ocr_data = filter_ocr_data_5(side_ocr_data)
    detailed_ocr_data = filter_ocr_data_5(detailed_ocr_data)

    print("经过预处理得到的top视图的SVTR结果:\n", *top_ocr_data, sep='\n')
    print("经过预处理得到的bottom视图的SVTR结果:\n", *bottom_ocr_data, sep='\n')
    print("经过预处理得到的side视图的SVTR结果:\n", *side_ocr_data, sep='\n')
    print("经过预处理得到的detailed视图的SVTR结果:\n", *detailed_ocr_data, sep='\n')

    # (1).提取’数字‘’+‘’-‘’=‘’Φ‘’±‘’max‘’nom‘’min‘’x'‘°’(2).当出现'±'时仅保留'±''数字''Φ'(3)当出现'°'时，标记absolute为'angle'
    top_ocr_data = filter_ocr_data_6(top_ocr_data)
    bottom_ocr_data = filter_ocr_data_6(bottom_ocr_data)
    side_ocr_data = filter_ocr_data_6(side_ocr_data)
    detailed_ocr_data = filter_ocr_data_6(detailed_ocr_data)

    # key_info中的数字如果以'0'开头而第二个字符却没有小数点，则添加小数点
    top_ocr_data = filter_ocr_data_9(top_ocr_data)
    bottom_ocr_data = filter_ocr_data_9(bottom_ocr_data)
    side_ocr_data = filter_ocr_data_9(side_ocr_data)
    detailed_ocr_data = filter_ocr_data_9(detailed_ocr_data)

    # 如果absolutely不是'angle',删除key_info中的'0''0.','00'
    top_ocr_data = filter_ocr_data_10(top_ocr_data)
    bottom_ocr_data = filter_ocr_data_10(bottom_ocr_data)
    side_ocr_data = filter_ocr_data_10(side_ocr_data)
    detailed_ocr_data = filter_ocr_data_10(detailed_ocr_data)

    # (1)删除key_info中的'.'(2)当key_info中只有'x'时删除
    top_ocr_data = filter_ocr_data_7(top_ocr_data)
    bottom_ocr_data = filter_ocr_data_7(bottom_ocr_data)
    side_ocr_data = filter_ocr_data_7(side_ocr_data)
    detailed_ocr_data = filter_ocr_data_7(detailed_ocr_data)

    # 删除标注关键信息检测识别为空
    top_ocr_data = filter_ocr_data__1(top_ocr_data)
    bottom_ocr_data = filter_ocr_data__1(bottom_ocr_data)
    side_ocr_data = filter_ocr_data__1(side_ocr_data)
    detailed_ocr_data = filter_ocr_data__1(detailed_ocr_data)

    tee_print("经过第一步后处理得到的top视图的SVTR结果:\n", *top_ocr_data, sep='\n')
    tee_print("经过第一步后处理得到的bottom视图的SVTR结果:\n", *bottom_ocr_data, sep='\n')
    tee_print("经过第一步后处理得到的side视图的SVTR结果:\n", *side_ocr_data, sep='\n')
    tee_print("经过第一步后处理得到的detailed视图的SVTR结果:\n", *detailed_ocr_data, sep='\n')

    # 6.1.1借助yolox将dbnet标注的框线坐标以及ocr内容合并
    top_ocr_data = bind_data(top_yolox_num, top_ocr_data)
    bottom_ocr_data = bind_data(bottom_yolox_num, bottom_ocr_data)
    side_ocr_data = bind_data(side_yolox_num, side_ocr_data)
    detailed_ocr_data = bind_data(detailed_yolox_num, detailed_ocr_data)
    # 清理key_info中不含数字的数据
    top_ocr_data = filter_ocr_data__2(top_ocr_data)
    bottom_ocr_data = filter_ocr_data__2(bottom_ocr_data)
    side_ocr_data = filter_ocr_data__2(side_ocr_data)
    detailed_ocr_data = filter_ocr_data__2(detailed_ocr_data)

    print("经过bind得到的top视图的SVTR结果:\n", *top_ocr_data, sep='\n')
    print("经过bind得到的bottom视图的SVTR结果:\n", *bottom_ocr_data, sep='\n')
    print("经过bind得到的side视图的SVTR结果:\n", *side_ocr_data, sep='\n')
    print("经过bind得到的detailed视图的SVTR结果:\n", *detailed_ocr_data, sep='\n')

    # 清除key_info中的英寸标注
    top_ocr_data = clear_inch(top_ocr_data)
    bottom_ocr_data = clear_inch(bottom_ocr_data)
    side_ocr_data = clear_inch(side_ocr_data)
    detailed_ocr_data = clear_inch(detailed_ocr_data)
    # 计算出各个标注的max_medium_min以及是否从符号上就可以判定该标注的语义
    top_ocr_data = cal_max_medium_min_top(top_ocr_data)
    bottom_ocr_data = cal_max_medium_min_bottom(bottom_ocr_data)
    side_ocr_data = cal_max_medium_min_side(side_ocr_data)
    detailed_ocr_data = cal_max_medium_min_side(detailed_ocr_data)
    # 0111注释（方便测试）
    print("经过第二步后处理（yolox）得到的top视图的SVTR结果:\n", *top_ocr_data, sep='\n')
    print("经过第二步后处理（yolox）得到的bottom视图的SVTR结果:\n", *bottom_ocr_data, sep='\n')
    print("经过第二步后处理（yolox）得到的side视图的SVTR结果:\n", *side_ocr_data, sep='\n')
    print("经过第二步后处理（yolox）得到的detailed视图的SVTR结果:\n", *detailed_ocr_data, sep='\n')

    # 删除公差特别大的标注
    top_ocr_data = filter_ocr_data_12(top_ocr_data)
    bottom_ocr_data = filter_ocr_data_12(bottom_ocr_data)
    side_ocr_data = filter_ocr_data_12(side_ocr_data)
    detailed_ocr_data = filter_ocr_data_12(detailed_ocr_data)
    # 补充test模式下可展示并修改ocr结果
    if key == 1:
        img_path = f"{DATA}/top.jpg"
        print("top的ocr结果")
        top_ocr_data = show_ocr_result(img_path, top_ocr_data)  # 展示dbnet框选的尺寸数字
        print("top_ocr_data\n", top_ocr_data)

        img_path = f"{DATA}/bottom.jpg"
        print("bottom的ocr结果")
        bottom_ocr_data = show_ocr_result(img_path, bottom_ocr_data)  # 展示dbnet框选的尺寸数字
        print("bottom_ocr_data\n", bottom_ocr_data)

        img_path = f"{DATA}/side.jpg"
        print("side的ocr结果")
        side_ocr_data = show_ocr_result(img_path, side_ocr_data)  # 展示dbnet框选的尺寸数字
        print("side_ocr_data\n", side_ocr_data)

        img_path = f"{DATA}/detailed.jpg"
        print("detailed的ocr结果")
        detailed_ocr_data = show_ocr_result(img_path, detailed_ocr_data)  # 展示dbnet框选的尺寸数字
        print("detailed_ocr_data\n", detailed_ocr_data)

    # 6.1.2.2针对各个视图，剔除对参数输出有影响的尺寸数字
    side_ocr_data = BGA_side_filter(side_ocr_data)

    return top_ocr_data, bottom_ocr_data, side_ocr_data, detailed_ocr_data


def find_PIN(top_yolox_serial_num, bottom_yolox_serial_num, top_ocr_data, bottom_ocr_data):
    # 找到QFP的序号，记录序号并删除标注
    top_serial_numbers = top_yolox_serial_num
    bottom_serial_numbers = bottom_yolox_serial_num

    top_serial_numbers_data, bottom_serial_numbers_data, top_ocr_data, bottom_ocr_data = find_serial_number_letter_QFP(
        top_serial_numbers,
        bottom_serial_numbers,
        top_ocr_data, bottom_ocr_data)
    return top_serial_numbers_data, bottom_serial_numbers_data, top_ocr_data, bottom_ocr_data


def find_BGA_PIN(serial_numbers, serial_letters, bottom_ocr_data):
    '''
        serial_numbers:np(,4)[x1,y1,x2,y2]
        serial_letters:np(,4)[x1,y1,x2,y2]
        bottom_dbnet_data:np(,4)[x1,y1,x2,y2]
        '''
    # 将serial提取出唯一值
    if len(serial_numbers) >= 1:
        maxlength = 0
        max_no = -1
        for i in range(len(serial_numbers)):
            if maxlength < max(serial_numbers[i][2] - serial_numbers[i][0],
                               serial_numbers[i][3] - serial_numbers[i][1]):
                maxlength = max(serial_numbers[i][2] - serial_numbers[i][0],
                                serial_numbers[i][3] - serial_numbers[i][1])
                max_no = i
        only_serial_numbers = serial_numbers[max_no]
        # print("only_serial_numbers", only_serial_numbers)
    if len(serial_letters) >= 1:
        maxlength = 0
        max_no = -1
        for i in range(len(serial_letters)):
            if maxlength < max(serial_letters[i][2] - serial_letters[i][0],
                               serial_letters[i][3] - serial_letters[i][1]):
                maxlength = max(serial_letters[i][2] - serial_letters[i][0],
                                serial_letters[i][3] - serial_letters[i][1])
                max_no = i
        only_serial_letters = serial_letters[max_no]
        # print("only_serial_letters", only_serial_letters)
    # 提取serial中的数字和字母
    ratio = 0.2
    serial_numbers_data = []
    serial_letters_data = []
    bottom_ocr_data_account = np.zeros((len(bottom_ocr_data)))  # 1 = 是serial的文本，需要剔除
    new_bottom_ocr_data = []
    if len(serial_numbers) != 0:
        for i in range(len(bottom_ocr_data)):
            if not (bottom_ocr_data[i]['location'][0] > only_serial_numbers[2] or bottom_ocr_data[i]['location'][2] < only_serial_numbers[
                0]):  # 两矩形在x坐标上的长有重叠
                if not (bottom_ocr_data[i]['location'][1] > only_serial_numbers[3] or bottom_ocr_data[i]['location'][3] <
                        only_serial_numbers[
                            1]):  # 两矩形在y坐标上的高有重叠
                    # print('********')
                    l = abs(bottom_ocr_data[i]['location'][2] - bottom_ocr_data[i]['location'][0]) + abs(only_serial_numbers[2] - \
                                                                                     only_serial_numbers[0]) - (
                                max(only_serial_numbers[2], bottom_ocr_data[i]['location'][2]) - min(only_serial_numbers[0],
                                                                                           bottom_ocr_data[i]['location'][0]))

                    w = bottom_ocr_data[i]['location'][3] - bottom_ocr_data[i]['location'][1] + only_serial_numbers[3] - \
                        only_serial_numbers[1] - (
                                max(only_serial_numbers[3], bottom_ocr_data[i]['location'][3]) - min(only_serial_numbers[1],
                                                                                           bottom_ocr_data[i]['location'][1]))
                    if l * w / (bottom_ocr_data[i]['location'][2] - bottom_ocr_data[i]['location'][0]) * (
                            bottom_ocr_data[i]['location'][3] - bottom_ocr_data[i]['location'][1]) > ratio or l * w / (
                            only_serial_numbers[2] - only_serial_numbers[0]) * (
                            only_serial_numbers[3] - only_serial_numbers[1]) > ratio:
                        serial_numbers_data.append(bottom_ocr_data[i])
                        bottom_ocr_data_account[i] = 1
    if len(serial_letters) != 0:
        for i in range(len(bottom_ocr_data)):
            if not (bottom_ocr_data[i]['location'][0] > only_serial_letters[2] or bottom_ocr_data[i]['location'][2] < only_serial_letters[
                0]):  # 两矩形在x坐标上的长有重叠
                if not (bottom_ocr_data[i]['location'][1] > only_serial_letters[3] or bottom_ocr_data[i]['location'][3] <
                        only_serial_letters[
                            1]):  # 两矩形在y坐标上的高有重叠
                    l = abs(bottom_ocr_data[i]['location'][2] - bottom_ocr_data[i]['location'][0]) + abs(only_serial_letters[2] - \
                                                                                     only_serial_letters[0]) - (
                                max(only_serial_letters[2], bottom_ocr_data[i]['location'][2]) - min(only_serial_letters[0],
                                                                                           bottom_ocr_data[i]['location'][0]))

                    w = bottom_ocr_data[i]['location'][3] - bottom_ocr_data[i]['location'][1] + only_serial_letters[3] - \
                        only_serial_letters[1] - (
                                max(only_serial_letters[3], bottom_ocr_data[i]['location'][3]) - min(only_serial_letters[1],
                                                                                           bottom_ocr_data[i]['location'][1]))
                    if l * w / (bottom_ocr_data[i]['location'][2] - bottom_ocr_data[i]['location'][0]) * (
                            bottom_ocr_data[i]['location'][3] - bottom_ocr_data[i]['location'][1]) > ratio or l * w / (
                            only_serial_letters[2] - only_serial_letters[0]) * (
                            only_serial_letters[3] - only_serial_letters[1]) > ratio:
                        serial_letters_data.append(bottom_ocr_data[i])
                        bottom_ocr_data_account[i] = 1
    for i in range(len(bottom_ocr_data_account)):
        if bottom_ocr_data_account[i] == 0:
            new_bottom_ocr_data.append(bottom_ocr_data[i])
    return serial_numbers_data, serial_letters_data, new_bottom_ocr_data


def find_pin_num_pin_1(serial_numbers_data, serial_letters_data, serial_numbers, serial_letters):
    '''
    serial_numbers_data:np.(,4)['x1','y1','x2','y2','str']
    serial_letters_data:np.(,4)['x1','y1','x2','y2','str']
    serial_numbers:np.(,4)[x1,y1,x2,y2)
    serial_letters:np.(,4)[x1,y1,x2,y2)
    '''
    # 默认输出
    pin_num_x_serial = 0
    pin_num_y_serial = 0
    pin_num_serial_number = 0
    pin_num_serial_letter = 0
    pin_1_location = np.array([-1, -1])
    # pin_1_location = [X, Y],X = 0:横向用数字标记序号，纵向用字母标记序号；X= 1，横向用字母标记序号，纵向用数字标记序号
    # pin_1_location = [X, Y],Y = 0 = 左上角,1= 右上角，2 = 右下角，3 = 左下角
    if len(serial_numbers_data) > 0 or len(serial_numbers_data) > 0:

        # ocr识别serial_number,serial_letter
        # img_path = 'data/bottom.jpg'
        # serial_numbers_data = ocr_en_cn_onnx(img_path, serial_numbers_data)
        # print('serial_numbers_data', serial_numbers_data)
        # serial_letters_data = ocr_en_cn_onnx(img_path, serial_letters_data)
        # print('serial_letters_data', serial_letters_data)
        # 根据经验修改ocr识别的错误
        serial_letters_data = correct_serial_letters_data(serial_letters_data)
        print('修正之后的serial_letters_data', serial_letters_data)
        # 根据serial_number最大值找行列数
        serial_number = np.zeros((0))
        new_serial_numbers_data = np.array([['0', '0', '0', '0', '0']])
        for i in range(len(serial_numbers_data)):
            try:
                serial_number = np.append(serial_number, int(serial_numbers_data[i][4]))
                new_serial_numbers_data = np.r_[new_serial_numbers_data, [serial_numbers_data[i]]]
            except:
                print("在用数字标识的pin行列序号中ocr识别到非数字信息，删除")

        serial_numbers_data = new_serial_numbers_data
        print('修正之后的serial_numbers_data', serial_numbers_data)
        serial_number = -(np.sort(-serial_number))  # 从大到小排列
        pin_num_serial_number = 0
        for i in range(len(serial_number)):
            if len(serial_number) > 1 and i + 1 < len(serial_number):
                if serial_number[i] - serial_number[i + 1] < 3:
                    pin_num_serial_number = serial_number[i]
                    break

        # 根据serial_letter最大值找行列数
        letter_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'T', 'U', 'V', 'W',
                       'Y']
        letter_list_a = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 't', 'u', 'v', 'w',
                         'y']
        serial_letter = np.zeros((0))
        # 将字母序列转为数字序列
        for i in range(len(serial_letters_data)):
            letter_number = 0
            no = 0
            letters = serial_letters_data[i][4]
            letters = letters[::-1]  # 倒序
            for every_letter in letters:
                no += 1
                for j in range(len(letter_list)):
                    if letter_list[j] == every_letter or letter_list_a[j] == every_letter:
                        letter_number += 20 ** (no - 1) * (j + 1)
            serial_letter = np.append(serial_letter, letter_number)
            serial_letters_data[i][4] = str(letter_number)
        print("serial_letters_data", serial_letters_data)
        serial_letter = -(np.sort(-serial_letter))  # 从大到小排列
        pin_num_serial_letter = 0
        for i in range(len(serial_letter)):
            if len(serial_letter) > 1 and i + 1 < len(serial_letter):
                if serial_letter[i] - serial_letter[i + 1] < 3:
                    pin_num_serial_letter = serial_letter[i]
                    break
        print('pin_num_serial_number, pin_num_serial_letter', pin_num_serial_number, pin_num_serial_letter)
    if pin_num_serial_number != 0:
        if len(serial_numbers) > 0:
            if abs(serial_numbers[0][0] - serial_numbers[0][2]) > abs(serial_numbers[0][1] - serial_numbers[0][3]):
                pin_num_x_serial = pin_num_serial_number
            else:
                pin_num_y_serial = pin_num_serial_number
    if pin_num_serial_letter != 0:
        if len(serial_letters) > 0:
            if abs(serial_letters[0][0] - serial_letters[0][2]) > abs(serial_letters[0][1] - serial_letters[0][3]):
                pin_num_x_serial = pin_num_serial_letter
            else:
                pin_num_y_serial = pin_num_serial_letter
    print("pin_num_x_serial, pin_num_y_serial", pin_num_x_serial, pin_num_y_serial)
    # pin_1_location = [X, Y],X = 0:横向用数字标记序号，纵向用字母标记序号；X= 1，横向用字母标记序号，纵向用数字标记序号
    # pin_1_location = [X, Y],Y = 0 = 左上角,1= 右上角，2 = 右下角，3 = 左下角
    # pin1定位
    if len(serial_numbers_data) > 0 or len(serial_numbers_data) > 0:
        if len(serial_numbers) > 0:
            if abs(serial_numbers[0][0] - serial_numbers[0][2]) > abs(serial_numbers[0][1] - serial_numbers[0][3]):
                pin_1_location[0] = 0
            else:
                pin_1_location[0] = 1
        if len(serial_letters) > 0:
            if abs(serial_letters[0][0] - serial_letters[0][2]) > abs(serial_letters[0][1] - serial_letters[0][3]):
                pin_1_location[0] = 1
            else:
                pin_1_location[0] = 0

        heng_begain = -1
        shu_begain = -1
        serial_numbers_data = serial_numbers_data.astype(np.float32)
        serial_numbers_data = serial_numbers_data.astype(np.int32)
        # 删除0
        new_serial_numbers_data = np.zeros((0, 5))
        for i in range(len(serial_numbers_data)):
            if serial_numbers_data[i][4] != 0:
                new_serial_numbers_data = np.r_[new_serial_numbers_data, [serial_numbers_data[i]]]
        serial_numbers_data = new_serial_numbers_data

        if len(serial_numbers_data) > 0:
            if abs(serial_numbers[0][0] - serial_numbers[0][2]) > abs(serial_numbers[0][1] - serial_numbers[0][3]):
                if len(serial_numbers_data) >= 2:
                    if len(serial_numbers_data[0]) == 5:
                        serial_numbers_data = serial_numbers_data[np.argsort(serial_numbers_data[:, 4])]  # 按序号从小到大排序
                        if serial_numbers_data[0, 0] < serial_numbers_data[(len(serial_numbers_data) - 1), 0]:
                            heng_begain = 0
                        else:
                            heng_begain = 1
            if abs(serial_numbers[0][0] - serial_numbers[0][2]) < abs(serial_numbers[0][1] - serial_numbers[0][3]):
                if len(serial_numbers_data) >= 2:
                    if len(serial_numbers_data[0]) == 5:
                        serial_numbers_data = serial_numbers_data[np.argsort(serial_numbers_data[:, 4])]  # 按序号从小到大排序
                        if serial_numbers_data[0, 1] < serial_numbers_data[(len(serial_numbers_data) - 1), 1]:
                            shu_begain = 0
                        else:
                            shu_begain = 1
        if len(serial_letters_data) > 0:
            serial_letters_data = serial_letters_data.astype(np.float32)
            serial_letters_data = serial_letters_data.astype(np.int32)
            # 删除0
            new_serial_letters_data = np.zeros((0, 5))
            for i in range(len(serial_letters_data)):
                if serial_letters_data[i][4] != 0:
                    new_serial_letters_data = np.r_[new_serial_letters_data, [serial_letters_data[i]]]
            serial_letters_data = new_serial_letters_data
            if abs(serial_letters[0][0] - serial_letters[0][2]) > abs(serial_letters[0][1] - serial_letters[0][3]):
                if len(serial_letters_data) >= 2:
                    if len(serial_letters_data[0]) == 5:
                        serial_letters_data = serial_letters_data[np.argsort(serial_letters_data[:, 4])]  # 按序号从小到大排序
                        if serial_letters_data[0, 0] < serial_letters_data[(len(serial_letters_data) - 1), 0]:
                            heng_begain = 0
                        else:
                            heng_begain = 1
            if abs(serial_letters[0][0] - serial_letters[0][2]) < abs(serial_letters[0][1] - serial_letters[0][3]):
                if len(serial_letters_data) >= 2:
                    if len(serial_letters_data[0]) == 5:
                        serial_letters_data = serial_letters_data[np.argsort(serial_letters_data[:, 4])]  # 按序号从小到大排序
                        if serial_letters_data[0, 1] < serial_letters_data[(len(serial_letters_data) - 1), 1]:
                            shu_begain = 0
                        else:
                            shu_begain = 1
        if heng_begain == 0 and shu_begain == 0:
            pin_1_location[1] = 0
        if heng_begain == 1 and shu_begain == 0:
            pin_1_location[1] = 1
        if heng_begain == 1 and shu_begain == 1:
            pin_1_location[1] = 2
        if heng_begain == 0 and shu_begain == 1:
            pin_1_location[1] = 3
        if heng_begain == 0 and shu_begain == -1:
            pin_1_location[1] = 0
        if heng_begain == 1 and shu_begain == -1:
            pin_1_location[1] = 1
        if heng_begain == -1 and shu_begain == 0:
            pin_1_location[1] = 0
        if heng_begain == -1 and shu_begain == 1:
            pin_1_location[1] = 3

    return pin_num_x_serial, pin_num_y_serial, pin_1_location

def MPD(key, top_yolox_pairs, bottom_yolox_pairs, side_yolox_pairs, detailed_yolox_pairs, side_angle_pairs, detailed_angle_pairs, top_border, bottom_border, top_ocr_data, bottom_ocr_data, side_ocr_data, detailed_ocr_data):
    empty_list = []
    img_path = f"{DATA}/top.jpg"
    # top_border[[28.90794373 130.14555359 743.2119751  827.08398438]]
    if not os.path.exists(img_path):
        top_ocr_data = empty_list
    else:
        top_ocr_data = match_pairs_data(img_path, top_yolox_pairs, top_ocr_data, top_border)
    print("top_ocr_data\n", top_ocr_data)
    if key == 1:
        print("展示top匹配的尺寸线和标注")
        show_matched_pairs_data(img_path, top_ocr_data)
    img_path = f"{DATA}/bottom.jpg"
    if not os.path.exists(img_path):
            bottom_ocr_data = empty_list
    else:
        bottom_ocr_data = match_pairs_data(img_path, bottom_yolox_pairs, bottom_ocr_data, bottom_border)
    print("bottom_ocr_data\n", bottom_ocr_data)
    if key == 1:
        print("展示bottom匹配的尺寸线和标注")
        show_matched_pairs_data(img_path, bottom_ocr_data)
    img_path = f"{DATA}/side.jpg"
    if not os.path.exists(img_path):
        side_ocr_data = empty_list
    # else:
    #     side_border = np.zeros((1, 4))
    #     side_ocr_data = match_pairs_data(img_path, side_yolox_pairs, side_ocr_data, side_border)
    #     side_ocr_data = match_pairs_data_angle(img_path, side_angle_pairs, side_ocr_data, side_border)
    print("side_pairs_data\n", side_ocr_data)
    if key == 1:
        print("展示side匹配的尺寸线和标注")
        show_matched_pairs_data(img_path, side_ocr_data)
    img_path = f"{DATA}/detailed.jpg"
    if not os.path.exists(img_path):
            detailed_ocr_data = empty_list
    else:
        detailed_border = np.zeros((1, 4))
        detailed_ocr_data = match_pairs_data(img_path, detailed_yolox_pairs, detailed_ocr_data, detailed_border)
        detailed_ocr_data = match_pairs_data_angle(img_path, detailed_angle_pairs, detailed_ocr_data, detailed_border)
    print("detailed_pairs_data\n", detailed_ocr_data)
    if key == 1:
        print("展示detailed匹配的尺寸线和标注")
        show_matched_pairs_data(img_path, detailed_ocr_data)
    return top_ocr_data, bottom_ocr_data, side_ocr_data, detailed_ocr_data

def get_better_data_2(top_ocr_data, bottom_ocr_data, side_ocr_data, detailed_ocr_data, top_yolox_pairs_length, bottom_yolox_pairs_length, side_yolox_pairs_length, detailed_yolox_pairs_length, top_yolox_pairs_copy, bottom_yolox_pairs_copy, side_yolox_pairs_copy, detailed_yolox_pairs_copy):
    # 引线信息和标尺线的种类写入字典
    top_ocr_data = get_yinxian_info(top_ocr_data, top_yolox_pairs_length)
    bottom_ocr_data = get_yinxian_info(bottom_ocr_data, bottom_yolox_pairs_length)
    side_ocr_data = get_yinxian_info(side_ocr_data, side_yolox_pairs_length)
    detailed_ocr_data = get_yinxian_info(detailed_ocr_data, detailed_yolox_pairs_length)
    top_ocr_data = get_pairs_info(top_ocr_data, top_yolox_pairs_copy)
    bottom_ocr_data = get_pairs_info(bottom_ocr_data, bottom_yolox_pairs_copy)
    side_ocr_data = get_pairs_info(side_ocr_data, side_yolox_pairs_copy)
    detailed_ocr_data = get_pairs_info(detailed_ocr_data, detailed_yolox_pairs_copy)
    # 0111注释（方便测试）
    tee_print("经过第三步后处理（opencv）得到的top视图的SVTR结果:\n", *top_ocr_data, sep='\n')
    tee_print("经过第三步后处理（opencv）得到的bottom视图的SVTR结果:\n", *bottom_ocr_data, sep='\n')
    tee_print("经过第三步后处理（opencv）得到的side视图的SVTR结果:\n", *side_ocr_data, sep='\n')
    tee_print("经过第三步后处理（opencv）得到的detailed视图的SVTR结果:\n", *detailed_ocr_data, sep='\n')
    # yolox_pairs_top,np.二维数组（，11）[pairs_x1_y1_x2_y2,标注x1_y1_x2_y2，max,medium,min]
    # top_yolox_pairs_length,np.二维数组（，13）[pairs_x1_y1_x2_y2,引线1_x1_y1_x2_y2,引线2_x1_y1_x2_y2,两引线距离]
    yolox_pairs_top = io_1(top_ocr_data)
    yolox_pairs_bottom = io_1(bottom_ocr_data)
    yolox_pairs_side = io_1(side_ocr_data)
    yolox_pairs_detailed = io_1(detailed_ocr_data)
    return top_ocr_data, bottom_ocr_data, side_ocr_data, detailed_ocr_data, yolox_pairs_top, yolox_pairs_bottom, yolox_pairs_side, yolox_pairs_detailed

def get_QFP_pitch(side_ocr_data, body_x, body_y, nx, ny):
    '''
    0.7 * body_x < (nx - 1) * pitch < body_x
    '''
    pitch_x = []
    pitch_y = []
    if len(side_ocr_data) > 0 and nx > 0 and len(body_x) > 0:
        for i in range(len(side_ocr_data)):
            # 0.7 * body_x < (nx - 1) * pitch < body_x
            if (nx - 1) * 0.7 * body_x[0]['max_medium_min'][2] <= side_ocr_data[i]['max_medium_min'][0] <= body_x[0]['max_medium_min'][0]:
                pitch_x = [side_ocr_data[i]]
    if len(side_ocr_data) > 0 and ny > 0 and len(body_y) > 0:
        for i in range(len(side_ocr_data)):
            # 0.7 * body_y < (ny - 1) * pitch < body_y
            if (ny - 1) * 0.7 * body_y[0]['max_medium_min'][2] <= side_ocr_data[i]['max_medium_min'][0] <= body_y[0]['max_medium_min'][0]:
                pitch_y = [side_ocr_data[i]]
    return pitch_x, pitch_y
def get_QFP_high(side_ocr_data):
    '''
    (1)
        当只找到一个max则判断绝对是high
        当找到多个max哪个最大哪个就是high
    (2)
        在参数列表中找最大的标注
    '''
    high_max = []
    for i in range(len(side_ocr_data)):
        if side_ocr_data[i]['Absolutely'] == 'high':
            try:
                high_max.append(side_ocr_data[i])
            except:
                pass
    if len(high_max) > 0:
        high_max.sort(key=lambda x: x['max_medium_min'][0])
        high = [high_max[0]]
        print("找到绝对正确的high:", high)
    else:

        high = []
        if len(side_ocr_data) == 0:
            high = []
            print("side视图未找到data")
            return high
        if len(side_ocr_data) != 0:

            high_max = 0
            key_no = 0

            for i in range(len(side_ocr_data)):
                # if side_data[i][0] != 0.5:
                if side_ocr_data[i]['max_medium_min'][0] > high_max:
                    high_max = side_ocr_data[i]['max_medium_min'][0]
                    high = [side_ocr_data[i]]
            print("通过数值大小找到high:", high)

    return high


def YOLO_DBnet_get_data(path):
    top_yolox_pairs, top_yolox_num, top_yolox_serial_num, top_pin, top_dbnet_data, top_other, top_pad, top_border, top_dbnet_time, top_angle_pairs = get_pairs_data(
        path)
    # print("top_border", top_border)
    top_yolox_pairs = np.around(top_yolox_pairs, decimals=2)
    top_yolox_num = np.around(top_yolox_num, decimals=2)
    top_dbnet_data = np.around(top_dbnet_data, decimals=2)
    top_angle_pairs = np.around(top_angle_pairs, decimals=2)
    # 参数格式:top_yolox_pairs  np.二维数组[x1,y1,x2,y2,0 = outside 1 = inside]
    # 参数格式:top_yolox_num np.二维数组[x1,y1,x2,y2]
    # 参数格式:top_dbnet_data np.二维数组[x1,y1,x2,y2]
    return top_yolox_pairs, top_yolox_num, top_yolox_serial_num, top_pin, top_dbnet_data, top_other, top_pad, top_border, top_dbnet_time, top_angle_pairs

def choose_x(binary):
    height, width = binary.shape[:2]
    cnt_length = []
    length_cnt = []
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        length = w
        cnt_length.append(contour)
        length_cnt.append(length)

    length_num = sorted(length_cnt, reverse=True)
    th = 0
    for i in range(len(length_num) - 1):
        if length_num[i] <= width * 0.8 and length_num[i] / length_num[i + 1] <= 1.05:
            th = length_num[i] * 0.6
            break

    cnts = []
    for cnt, lgt in zip(cnt_length, length_cnt):
        if width * 0.8 >= lgt >= th:
            cnts.append(cnt)

    binary_image = np.zeros((height, width, 1), dtype=np.uint8)
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.line(binary_image, (int(x - (w * 0.05)), int(y + (h * 0.5))), (int(x + (w * 1.05)), int(y + (h * 0.5))),
                 (255), 1)

    return binary_image


def choose_y(binary):
    height, width = binary.shape[:2]
    cnt_length = []
    length_cnt = []
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        length = h
        cnt_length.append(contour)
        length_cnt.append(length)

    length_num = sorted(length_cnt, reverse=True)
    th = 0
    for i in range(len(length_num) - 1):
        if length_num[i] <= height * 0.8 and length_num[i] / length_num[i + 1] <= 1.05:
            th = length_num[i] * 0.6
            break

    cnts = []
    for cnt, lgt in zip(cnt_length, length_cnt):
        if height * 0.8 >= lgt >= th:
            cnts.append(cnt)

    binary_image = np.zeros((height, width, 1), dtype=np.uint8)
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.line(binary_image, (int(x + (w * 0.5)), int(y - (h * 0.05))), (int(x + (w * 0.5)), int(y + (h * 1.05))),
                 (255), 1)

    return binary_image


def output_body(img_path, name):
    src_img = cv2.imread(img_path)
    src_img1 = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    AdaptiveThreshold = cv2.bitwise_not(src_img1)
    thresh, AdaptiveThreshold = cv2.threshold(AdaptiveThreshold, 10, 255, 0)

    horizontal = AdaptiveThreshold.copy()
    vertical = AdaptiveThreshold.copy()

    horizontalSize = int(horizontal.shape[1] / 8)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    verticalSize = int(vertical.shape[0] / 8)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalSize))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    horizontal = choose_x(horizontal)
    vertical = choose_y(vertical)
    mask = horizontal + vertical

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    height, width = src_img.shape[:2]
    binary_image = np.zeros((height, width, 1), dtype=np.uint8)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        area_cnt = cv2.contourArea(cnt)
        if w > width / 8 and h > height / 8 and area_cnt / area >= 0.95:
            cv2.rectangle(binary_image, (x, y), (x + w, y + h), (255), 3)

    contours2, hierarchy2 = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rec = src_img.copy()
    for cnt in contours2:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(rec, (x, y), (x + w - 3, y + h - 3), (255, 0, 0), 3)
    file_name = f'{OPENCV_OUTPUT}/' + name + 'output_waikuang.jpg'
    cv2.imwrite(file_name, rec)
    # 测试用
    # print("opencv函数找到外框",file_name)
    # cv2.namedWindow('body', 0)
    # cv2.imshow('body', rec)
    # cv2.waitKey(0)
    try:
        location = np.array([[x, y, x + w - 3, y + h - 3]])
    except:
        location = np.array([])
        print("opencv函数找不到外框")
    return location


def find_all_lines(img_path, test_mode):
    '''
    找到一张图中所有的直线
    '''
    # img_path = r'data_copy/bottom.jpg'
    src_img = cv2.imread(img_path)
    src_img1 = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    # src_img1 = cv2.GaussianBlur(src_img1, (3, 3), 0)
    thresh, AdaptiveThreshold = cv2.threshold(src_img1, 240, 255, 0)
    AdaptiveThreshold = cv2.bitwise_not(AdaptiveThreshold)

    horizontal = AdaptiveThreshold.copy()
    vertical = AdaptiveThreshold.copy()

    horizontalSize = int(horizontal.shape[1] / 25)  # 默认40
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    verticalSize = int(vertical.shape[0] / 25)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalSize))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    syz_heng = []
    syz_shu = []

    contours1, hierarchy1 = cv2.findContours(horizontal, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for cnt in contours1:
        x, y, w, h = cv2.boundingRect(cnt)
        xq = x
        yq = int(y + (h / 2))
        xz = x + w
        yz = int(y + (h / 2))
        syz_heng.append([min(xq, xz), min(yq, yz), max(xq, xz), max(yq, yz)])
        # [min(xq, xz), min(yq, yz), max(xq, xz), max(yq, yz)]

    contours2, hierarchy2 = cv2.findContours(vertical, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for cnt in contours2:
        x, y, w, h = cv2.boundingRect(cnt)
        xq = int(x + (w / 2))
        yq = y
        xz = int(x + (w / 2))
        yz = y + h
        syz_shu.append([min(xq, xz), min(yq, yz), max(xq, xz), max(yq, yz)])
    syz_heng = np.array(syz_heng)
    syz_shu = np.array(syz_shu)

    if test_mode == 1:
        for point in syz_heng:
            cv2.line(src_img, [point[0], point[1]], [point[2], point[3]], (0, 0, 255), 2)
        for point in syz_shu:
            cv2.line(src_img, [point[0], point[1]], [point[2], point[3]], (0, 0, 255), 2)
        cv2.namedWindow('all_line', 0)
        cv2.imshow("all_line", src_img)
        cv2.waitKey(0)

    return syz_heng, syz_shu


def get_rotate_crop_image(img, points):  # 图片分割，在ultil中的原有函数,from utils import get_rotate_crop_image
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height), borderMode=cv2.BORDER_REPLICATE,
                                  flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    # if dst_img_height * 1.0 / dst_img_width >= 1:
    #     dst_img = np.rot90(dst_img)
    return dst_img

def img_resize(image):
    height, width = image.shape[0], image.shape[1]
    # 设置新的图片分辨率框架
    width_new = 160
    height_new = 80
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))
    return img_new


def img_clear(img):
    img = cv2.bilateralFiler(img, 9, 75, 75)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    ret, thresh = cv2.threshold(laplacian, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MOEPH_RECT, (3, 3))
    img = cv2.dilate(thresh, kernel)
    return img


def comma_inter_point(str_data):  # 将字符串中的comma转换为point
    str_data = list(str_data)  # str不可修改，转换成list可以修改元素
    for i in range(len(str_data)):
        if str_data[i] == ',':
            str_data[i] = '.'
    str_data = ''.join(str_data)  # 将列表转换成字符串
    return str_data


def jump_inter_comma(str_data):
    str_data = list(str_data)
    for i in range(len(str_data)):
        if str_data[i] == ' ':
            str_data[i] = ','
    str_data = ''.join(str_data)
    return str_data


def get_data_and_del_en(string):  # 将输入字符串，从中提取数字（含小数点），删除中英文
    # import re
    # string = "轻型车：共有198家企业4747个车型（12305个信息公开编号）15498915辆车进行了轻型车国六环保信息公开，与上周汇总环比增加105个车型、386379辆车。其中，国内生产企业177家、4217个车型、14645390辆，国外生产企业21家、530个车型、853525辆；轻型汽油"
    # 打印结果：['198', '4747', '12305', '15498915', '105', '386379', '177', '4217', '14645390', '21', '530', '853525']
    # 最终打印str:"198474712305..."
    str_data = []
    str_data_another = []
    # str_data = re.findall("\d+\.?\d*", string)  # 正则表达式:小数或者整数
    str_data = re.findall("\d*\.?\d*", string)  # 正则表达式:小数或者整数 + .40
    # 问题：如果string是6.250.10时怎么解决：
    # 方法1：根据小数点数量判断是#公差类型#还是#后面括号包含英寸#
    # 实际方法：对正则输出两个数的默认为公差类型，直接将公差化为小数点后一位的数量级
    str_data = [x.strip() for x in str_data if x.strip() != '']  # 将字符串列表中空项删除

    # print("str_data,存储一个data里面一行的所有文本里面的数字(字符串列表)\n", str_data)
    # list转化为字符串numpy数组，再转化为数字numpy数组
    str_data = np.asarray(str_data)
    str_data = str_data.astype(np.float_)
    #####################################
    # print("str_data（数字数组）\n", str_data)
    if len(str_data) == 2:  # 一个data里面的一行文本里面如果有两个数字则判断为公差形式，则判断第二个数字是公差，将公差数量级降为第一个数字的相匹配的数量级
        # 数字大于等于1时，公差为0.1型，数字为0.1型，公差为0.01型
        if str_data[0] >= 1:
            while str_data[1] >= 1:
                str_data[1] = str_data[1] * 0.1
        if 0.1 <= str_data[0] < 1 and str_data[1] >= 1:
            while str_data[1] >= 0.1:
                str_data[1] = str_data[1] * 0.1

    str_data = str_data.astype(str).tolist()  # 数字numpy数组转换为字符串numpy数组再转化为字符串list
    # print("str_data\n", str_data)

    if len(str_data) != 1:
        if len(str_data) == 2:
            str_data_another = str_data[1]
            str_data = str_data[0]

    str_data = ''.join(str_data)  # 将列表转换成字符串
    str_data_another = ''.join(str_data_another)
    # print("str_data（字符串）\n", str_data)
    # print("str_data_another（字符串）\n", str_data_another)
    return str_data, str_data_another


def get_np_array_in_txt(file_path):  # 提取txt中保存的数组，要求：浮点数且用逗号隔开
    # import numpy as np
    with open(file_path) as f:
        line = f.readline()
        data_array = []
        while line:
            # num = list(map(int, line.split(',')))
            num = list(map(float, line.split(',')))
            data_array.append(num)
            line = f.readline()
        data_array = np.array(data_array)

    # print(data_array[0][:])
    # print('*' * 50)
    # print(data_array)
    return data_array

def get_high_pin_high_max_1(side_data, body_x, body_y):
    # print("side_data",side_data)
    high = np.zeros(3)
    if len(side_data) == 0:
        high = np.zeros(3)
        print("side视图未找到data")
        return high
    if len(side_data) != 0:

        high_max = 0
        key_no = 0

        for i in range(len(side_data)):
            if (side_data[i][1:4] < body_x).all() and (side_data[i][1:4] < body_y).all():
                if side_data[i][1] > high_max:
                    high_max = side_data[i][1]
                    high = side_data[i, 1:]
        if high[0] == high[1] == 0:
            for i in range(len(side_data)):
                if (side_data[i][1:4] < body_x).all() and (side_data[i][1:4] < body_y).all():
                    if side_data[i][1] > high_max:
                        high_max = side_data[i][1]
                        high = side_data[i, 1:]

    return high


def get_high_pin_high_max(side_data):
    # print("side_data",side_data)
    high = np.zeros(3)
    if len(side_data) == 0:
        high = np.zeros(3)
        print("side视图未找到data")
        return high
    if len(side_data) != 0:

        high_max = 0
        key_no = 0

        for i in range(len(side_data)):
            # if side_data[i][0] != 0.5:
            if side_data[i][1] > high_max:
                high_max = side_data[i][1]
                high = side_data[i, 1:]
        if high[0] == high[1] == 0:
            for i in range(len(side_data)):
                # if side_data[i][0] != 0.5:
                if side_data[i][1] > high_max:
                    high_max = side_data[i][1]
                    high = side_data[i, 1:]

    return high


def get_pin_diameter(pitch_x, pitch_y, pin_x_number, pin_y_number, body_x, body_y, bottom_data_list_np,
                     side_data_list_np,
                     top_data_list_np):  # 从三视图中找pin直径，方法是看data的最大值和行列数减一相乘是否小于长和宽，最小值和行列数减一是否大于长和宽的一半
    pin_diameter = np.zeros((1, 3))  # 存储可能的pin直径值，
    # print(bottom_data_list_np,pin_diameter,bottom_data_list_np[1][1:4])
    for i in range(len(bottom_data_list_np)):

        if (bottom_data_list_np[i][1] < pitch_x).any() and (
                bottom_data_list_np[i][1] < pitch_y).any():  # 如果data最大值比pitch值小
            if bottom_data_list_np[i][1] * pin_x_number < body_x[2] and bottom_data_list_np[i][1] * pin_y_number < \
                    body_y[2]:
                if bottom_data_list_np[i][1] * (pin_x_number) > body_x[0] * 0.5 and bottom_data_list_np[i][1] * (
                        pin_y_number) > body_y[0] * 0.5 and bottom_data_list_np[i, 1:3] not in pin_diameter:
                    pin_diameter = np.row_stack((pin_diameter, bottom_data_list_np[i, 1:4]))
                    # print(pin_diameter)
    for i in range(len(side_data_list_np)):
        if (side_data_list_np[i][1] < pitch_x).any() and (side_data_list_np[i][1] < pitch_y).any():
            if side_data_list_np[i][1] * pin_x_number < body_x[2] and side_data_list_np[i][1] * pin_y_number < body_y[
                2]:
                if side_data_list_np[i][1] * (pin_x_number) > body_x[0] * 0.5 and side_data_list_np[i][1] * (
                        pin_y_number) > body_y[0] * 0.5 and side_data_list_np[i, 1:3] not in pin_diameter:
                    pin_diameter = np.row_stack((pin_diameter, side_data_list_np[i, 1:4]))
    for i in range(len(top_data_list_np)):
        if (top_data_list_np[i][1] < pitch_x).any() and (top_data_list_np[i][1] < pitch_y).any():
            if top_data_list_np[i][1] * pin_x_number < body_x[2] and top_data_list_np[i][1] * pin_y_number < body_y[2]:
                if top_data_list_np[i][1] * (pin_x_number) > body_x[0] * 0.5 and top_data_list_np[i][1] * (
                        pin_y_number) > body_y[0] * 0.5 and top_data_list_np[i, 1:3] not in pin_diameter:
                    pin_diameter = np.row_stack((pin_diameter, top_data_list_np[i, 1:4]))
    pin_diameter = pin_diameter[1:]
    # 洗去重复项
    if len(pin_diameter) > 1:
        pin_diameter_2 = np.zeros((0, 3))
        for i in range(len(pin_diameter)):
            if pin_diameter[i] not in pin_diameter_2:
                pin_diameter_2 = np.r_[pin_diameter_2, [pin_diameter[i]]]
        pin_diameter = pin_diameter_2
    # 进一步筛选0.45 - 0.85
    pin_diameter_3 = np.zeros((0, 3))
    if len(pin_diameter) > 1:
        for i in range(len(pin_diameter)):
            if ((pin_diameter[i] / pitch_x[0]) > 0.85).any() or ((pin_diameter[i] / pitch_x[0]) < 0.45).any():
                print("filter")
            else:
                pin_diameter_3 = np.r_[pin_diameter_3, [pin_diameter[i]]]
        pin_diameter = pin_diameter_3
    # 进一步筛选0.45 - 0.8
    pin_diameter_3 = np.zeros((0, 3))
    if len(pin_diameter) > 1:
        for i in range(len(pin_diameter)):
            if ((pin_diameter[i] / pitch_x[0]) > 0.80).any() or ((pin_diameter[i] / pitch_x[0]) < 0.45).any():
                print("filter")
            else:
                pin_diameter_3 = np.r_[pin_diameter_3, [pin_diameter[i]]]
        if len(pin_diameter_3) != 0 and len(pin_diameter_3) != len(pin_diameter):
            pin_diameter = pin_diameter_3
    # 进一步筛选0.45 - 0.75
    pin_diameter_3 = np.zeros((0, 3))
    if len(pin_diameter) > 1:
        for i in range(len(pin_diameter)):
            if ((pin_diameter[i] / pitch_x[0]) >= 0.75).any() or ((pin_diameter[i] / pitch_x[0]) < 0.45).any():
                print("filter")
            else:
                pin_diameter_3 = np.r_[pin_diameter_3, [pin_diameter[i]]]
        if len(pin_diameter_3) != 0 and len(pin_diameter_3) != len(pin_diameter):
            pin_diameter = pin_diameter_3
    # 进一步筛选如果多个尺寸数字中仅有一个含误差的，确定含误差为直径
    pin_diameter_3 = np.zeros((0, 3))
    j = 0
    if len(pin_diameter) > 1:
        for i in range(len(pin_diameter)):
            if pin_diameter[i][0] == pin_diameter[i][1] == pin_diameter[i][2]:
                print("filter")
            else:
                j += 1
                pin_diameter_3 = np.r_[pin_diameter_3, [pin_diameter[i]]]
        if j == 1:
            pin_diameter = pin_diameter_3

    return pin_diameter  # numpy二维数组，可能不止一行

#1202优化之后
def clear_inch(ocr_data):
    '''
    清除key_info中inch标注，保留mm标注
    '''
    # print("clear_inch之前\n", *ocr_data, sep='\n') # 可选：减少打印以提升速度
    try:
        for item in ocr_data:
            # 展平当前 item 的 key_info 中的所有有效数值
            # 格式转换：确保是 float
            flat_values = []
            valid_indices = []  # 记录 (row_idx, col_idx)

            raw_info = item["key_info"]

            # 第一步：解析所有数字，构建查找表
            for r, sublist in enumerate(raw_info):
                for c, val in enumerate(sublist):
                    if isinstance(val, str):
                        # 简单的数字检查
                        if val.replace('.', '', 1).isdigit():
                            val_float = float(val)
                            flat_values.append(val_float)
                            valid_indices.append((r, c))
                    elif isinstance(val, (int, float)):
                        flat_values.append(float(val))
                        valid_indices.append((r, c))

            # 第二步：检查英寸关系
            # 如果存在 B ~= A * 0.03937，则 A 是 mm (保留)，B 是 inch (删除)
            # 或者 B = A / 25.4

            indices_to_delete = set()

            for i in range(len(flat_values)):
                val_inch = flat_values[i]
                for j in range(len(flat_values)):
                    if i == j: continue
                    val_mm = flat_values[j]

                    # 避免除零错误
                    if val_inch == 0: continue

                    # 动态容差计算 (保持原有逻辑)
                    tolerance = 10 ** (math.log10(val_inch) - 2) if val_inch > 0 else 1e-3

                    # 检查 val_inch 是否是 val_mm 的英寸表示
                    if abs(val_mm * 0.03937008 - val_inch) < tolerance:
                        indices_to_delete.add(valid_indices[i])
                        break  # 确定是英寸后，跳出内层循环

            # 第三步：重构 key_info
            if indices_to_delete:
                new_key_info = []
                for r, sublist in enumerate(raw_info):
                    new_sublist = []
                    for c, val in enumerate(sublist):
                        if (r, c) not in indices_to_delete:
                            new_sublist.append(val)
                        # else: 被删除了
                    if new_sublist:
                        new_key_info.append(new_sublist)
                item["key_info"] = new_key_info

    except Exception as e:
        print("删除英寸标注错误", e)

    # print("clear_inch之后\n", *ocr_data, sep='\n')
    return ocr_data
#1202优化之前
# def clear_inch(ocr_data):
#     '''
#     清除key_info中inch标注，保留mm标注
#     :param ocr_data:
#     :return:
#     '''
#     print("clear_inch之前\n", *ocr_data, sep='\n')
#     try:
#         # tolerance = 1e-3
#         for i in range(len(ocr_data)):
#             new_info = copy.deepcopy(ocr_data[i]["key_info"])
#             new_info = [[float(item) if item and item.replace('.', '', 1).isdigit() else item for item in sublist] for
#                         sublist in new_info]
#             for j in range(len(new_info)):
#                 for k in range(len(new_info[j])):
#
#                         for l in range(len(new_info)):
#                             for m in range(len(new_info[l])):
#                                 try:
#                                     if new_info[j][k] < new_info[l][m]:
#                                         #动态容差
#                                         tolerance = 10**(math.log10(new_info[j][k]) - 2)
#                                         if abs(new_info[l][m] * 0.03937008 - new_info[j][k]) < tolerance:
#                                             # print(new_info[l][m], new_info[j][k])
#                                             new_info[j][k] = 'delete'
#
#
#                                 except:
#                                     pass
#             # 删除new_info中的'delete'
#             # print(new_info)
#             new_info = [[item for item in sublist if item != 'delete'] for sublist in new_info]
#             # print(new_info)
#             # 浮点数转为字符串
#             new_info = [[str(item) if isinstance(item, (int, float)) else item for item in sublist] for sublist in
#                         new_info]
#             # print(new_info)
#             ocr_data[i]["key_info"] = new_info
#     except Exception as e:
#         print("删除英寸标注错误", e)
#     print("clear_inch之后\n", *ocr_data, sep='\n')
#     return ocr_data



def cal_max_medium_min_top(ocr_data):
    '''
    根据key_info计算出max-medium_min
    '''
    # 排查是否存在'Φ'
    for i in range(len(ocr_data)):
        # === [新增开始] ===
        key_info_list = ocr_data[i]['key_info']
        found_eq = False
        for sublist in key_info_list:
            if '=' in sublist:
                try:
                    eq_idx = sublist.index('=')
                    if eq_idx + 1 < len(sublist):
                        val = float(sublist[eq_idx + 1])
                        ocr_data[i]['max_medium_min'] = np.array([val, val, val])
                        found_eq = True
                        break
                except:
                    pass
        if found_eq: continue

        ed = 0
        num = 0
        list = ocr_data[i]['key_info']
        dic = {'+': 0, '-': 0, '±': 0, 'none': []}
        m_3 = np.array([0, 0, 0])
        for j in range(len(list)):
            for k in range(len(list[j])):
                if list[j][k] == 'Φ' and ed != 1:
                    if ocr_data[i]['Absolutely'] == 'mb_pin_diameter':
                        ocr_data[i]['Absolutely'] = 'pin_diameter+'
                        ed = 1
                    else:
                        ocr_data[i]['Absolutely'] = 'pin_diameter'
                        ed = 1
                try:
                    a = float(list[j][k])
                    if k > 0:
                        if list[j][k - 1] == '+':
                            dic['+'] = a
                        elif list[j][k - 1] == '-':
                            dic['-'] = a
                        elif list[j][k - 1] == '±':
                            dic['±'] = a
                        else:
                            dic['none'].append(a)
                    else:
                        dic['none'].append(a)
                    num += 1
                except:
                    a = 0
        compire_ratio = 3
        if len(dic['none']) == num == 3:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if y[0] / y[1] > compire_ratio:
                m_3 = np.array([y[0] + y[1], y[0], y[0] - y[2]])
            else:
                m_3 = np.array([y[0], y[1], y[2]])
        elif len(dic['none']) == 2 and num == 3:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if dic['±'] != 0:
                m_3 = np.array([y[0] + dic['±'], y[0], y[0] - dic['±']])
            elif dic['+'] != 0:
                m_3 = np.array([y[0] + dic['+'], y[0], y[0] - y[1]])
            else:
                m_3 = np.array([y[0] + y[1], y[0], y[0] - dic['-']])
        elif len(dic['none']) == 1 and num == 3:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if dic['±'] != 0:
                m_3 = np.array([y[0] + dic['±'], y[0], y[0] - dic['±']])
            else:
                m_3 = np.array([y[0] + dic['+'], y[0], y[0] - dic['-']])
        elif len(dic['none']) == 0 and num == 3:
            m_3 = m_3
        elif len(dic['none']) == 2 and num == 2:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if y[0] / y[1] > compire_ratio:
                m_3 = np.array([y[0] + y[1], y[0], y[0] - y[1]])
            else:
                m_3 = np.array([y[0], (y[0] + y[1]) * 0.5, y[1]])
        elif len(dic['none']) == 1 and num == 2:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if dic['±'] != 0:
                m_3 = np.array([y[0] + dic['±'], y[0], y[0] - dic['±']])
            elif dic['+'] != 0:
                m_3 = np.array([y[0] + dic['+'], y[0], y[0] - dic['+']])
            else:
                m_3 = np.array([y[0] + dic['-'], y[0], y[0] - dic['-']])
        elif len(dic['none']) == 0 and num == 2:
            m_3 = m_3
        elif len(dic['none']) == 1 and num == 1:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            m_3 = np.array([y[0], y[0], y[0]])
        elif len(dic['none']) == 0 and num == 1:
            m_3 = np.array(
                [dic['±'] + dic['+'] + dic['-'], dic['±'] + dic['+'] + dic['-'], dic['±'] + dic['+'] + dic['-']])
        else:
            m_3 = m_3
        ocr_data[i]['max_medium_min'] = m_3
    return ocr_data


def cal_max_medium_min_bottom(ocr_data):
    '''
    根据key_info计算出max-medium_min
    '''
    # 排查是否存在'Φ'
    # for i in range(len(ocr_data)):
    #     if np.array(ocr_data[i]['key_info']).ndim == 1:
    #         ocr_data[i]['key_info'] = [ocr_data[i]['key_info']]
    for i in range(len(ocr_data)):
        # === [新增开始] ===
        key_info_list = ocr_data[i]['key_info']
        found_eq = False
        for sublist in key_info_list:
            if '=' in sublist:
                try:
                    eq_idx = sublist.index('=')
                    if eq_idx + 1 < len(sublist):
                        val = float(sublist[eq_idx + 1])
                        ocr_data[i]['max_medium_min'] = np.array([val, val, val])
                        found_eq = True
                        break
                except:
                    pass
        if found_eq: continue

        num = 0
        ed = 0
        list = ocr_data[i]['key_info']
        dic = {'+': 0, '-': 0, '±': 0, 'none': []}
        m_3 = np.array([0, 0, 0])
        for j in range(len(list)):
            for k in range(len(list[j])):
                if list[j][k] == 'Φ' and ed != 1:
                    if ocr_data[i]['Absolutely'] == 'mb_pin_diameter':
                        ocr_data[i]['Absolutely'] = 'pin_diameter+'
                        ed = 1
                    else:
                        ocr_data[i]['Absolutely'] = 'pin_diameter'
                        ed = 1
                try:
                    a = float(list[j][k])
                    if k > 0:
                        if list[j][k - 1] == '+':
                            dic['+'] = a
                        elif list[j][k - 1] == '-':
                            dic['-'] = a
                        elif list[j][k - 1] == '±':
                            dic['±'] = a
                        else:
                            dic['none'].append(a)
                    else:
                        dic['none'].append(a)
                    num += 1
                except:
                    a = 0
        compire_ratio = 3
        if len(dic['none']) == num == 3:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if y[0] / y[1] > compire_ratio:
                m_3 = np.array([y[0] + y[1], y[0], y[0] - y[2]])
            else:
                m_3 = np.array([y[0], y[1], y[2]])
        elif len(dic['none']) == 2 and num == 3:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if dic['±'] != 0:
                m_3 = np.array([y[0] + dic['±'], y[0], y[0] - dic['±']])
            elif dic['+'] != 0:
                m_3 = np.array([y[0] + dic['+'], y[0], y[0] - y[1]])
            else:
                m_3 = np.array([y[0] + y[1], y[0], y[0] - dic['-']])
        elif len(dic['none']) == 1 and num == 3:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if dic['±'] != 0:
                m_3 = np.array([y[0] + dic['±'], y[0], y[0] - dic['±']])
            else:
                m_3 = np.array([y[0] + dic['+'], y[0], y[0] - dic['-']])
        elif len(dic['none']) == 0 and num == 3:
            m_3 = m_3
        elif len(dic['none']) == 2 and num == 2:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if y[0] / y[1] > compire_ratio:
                m_3 = np.array([y[0] + y[1], y[0], y[0] - y[1]])
            else:
                m_3 = np.array([y[0], (y[0] + y[1]) * 0.5, y[1]])
        elif len(dic['none']) == 1 and num == 2:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if dic['±'] != 0:
                m_3 = np.array([y[0] + dic['±'], y[0], y[0] - dic['±']])
            elif dic['+'] != 0:
                m_3 = np.array([y[0] + dic['+'], y[0], y[0] - dic['+']])
            else:
                m_3 = np.array([y[0] + dic['-'], y[0], y[0] - dic['-']])
        elif len(dic['none']) == 0 and num == 2:
            m_3 = m_3
        elif len(dic['none']) == 1 and num == 1:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            m_3 = np.array([y[0], y[0], y[0]])
        elif len(dic['none']) == 0 and num == 1:
            m_3 = np.array(
                [dic['±'] + dic['+'] + dic['-'], dic['±'] + dic['+'] + dic['-'], dic['±'] + dic['+'] + dic['-']])
        else:
            m_3 = m_3
        ocr_data[i]['max_medium_min'] = m_3
    return ocr_data


def cal_max_medium_min_side(ocr_data):
    '''
    根据key_info计算出max-medium_min
    '''
    # 排查是否存在唯一'max'
    for i in range(len(ocr_data)):
        max_num = 0
        no_acc = -1
        str = re.findall("[Mm][Aa][Xx]", ocr_data[i]['ocr_strings'])
        if len(str) > 0:
            max_num += len(str)
            no_acc = i
        if max_num > 0:
            ocr_data[no_acc]['Absolutely'] = 'high'

    # 排查是否存在'Φ'
    for i in range(len(ocr_data)):
        # === [新增开始] ===
        key_info_list = ocr_data[i]['key_info']
        found_eq = False
        for sublist in key_info_list:
            if '=' in sublist:
                try:
                    eq_idx = sublist.index('=')
                    if eq_idx + 1 < len(sublist):
                        val = float(sublist[eq_idx + 1])
                        ocr_data[i]['max_medium_min'] = np.array([val, val, val])
                        found_eq = True
                        break
                except:
                    pass
        if found_eq: continue

        num = 0
        ed = 0
        list = ocr_data[i]['key_info']
        dic = {'+': 0, '-': 0, '±': 0, 'none': []}
        m_3 = np.array([0, 0, 0])
        for j in range(len(list)):
            for k in range(len(list[j])):
                if list[j][k] == 'Φ' and ed != 1:
                    if ocr_data[i]['Absolutely'] == 'mb_pin_diameter':
                        ocr_data[i]['Absolutely'] = 'pin_diameter+'
                        ed = 1
                    else:
                        ocr_data[i]['Absolutely'] = 'pin_diameter'
                        ed = 1
                try:
                    a = float(list[j][k])
                    if k > 0:
                        if list[j][k - 1] == '+':
                            dic['+'] = a
                        elif list[j][k - 1] == '-':
                            dic['-'] = a
                        elif list[j][k - 1] == '±':
                            dic['±'] = a
                        else:
                            dic['none'].append(a)
                    else:
                        dic['none'].append(a)
                    num += 1
                except:
                    a = 0
        compire_ratio = 3
        if len(dic['none']) == num == 3:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if y[0] / y[1] > compire_ratio:
                m_3 = np.array([y[0] + y[1], y[0], y[0] - y[2]])
            else:
                m_3 = np.array([y[0], y[1], y[2]])
        elif len(dic['none']) == 2 and num == 3:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if dic['±'] != 0:
                m_3 = np.array([y[0] + dic['±'], y[0], y[0] - dic['±']])
            elif dic['+'] != 0:
                m_3 = np.array([y[0] + dic['+'], y[0], y[0] - y[1]])
            else:
                m_3 = np.array([y[0] + y[1], y[0], y[0] - dic['-']])
        elif len(dic['none']) == 1 and num == 3:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if dic['±'] != 0:
                m_3 = np.array([y[0] + dic['±'], y[0], y[0] - dic['±']])
            else:
                m_3 = np.array([y[0] + dic['+'], y[0], y[0] - dic['-']])
        elif len(dic['none']) == 0 and num == 3:
            m_3 = m_3
        elif len(dic['none']) == 2 and num == 2:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if y[0] / y[1] > compire_ratio:
                m_3 = np.array([y[0] + y[1], y[0], y[0] - y[1]])
            else:
                m_3 = np.array([y[0], (y[0] + y[1]) * 0.5, y[1]])
        elif len(dic['none']) == 1 and num == 2:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            if dic['±'] != 0:
                m_3 = np.array([y[0] + dic['±'], y[0], y[0] - dic['±']])
            elif dic['+'] != 0:
                m_3 = np.array([y[0] + dic['+'], y[0], y[0] - dic['+']])
            else:
                m_3 = np.array([y[0] + dic['-'], y[0], y[0] - dic['-']])
        elif len(dic['none']) == 0 and num == 2:
            m_3 = m_3
        elif len(dic['none']) == 1 and num == 1:
            y = np.sort(dic['none'])
            y = abs(np.sort(-y))
            m_3 = np.array([y[0], y[0], y[0]])
        elif len(dic['none']) == 0 and num == 1:
            m_3 = np.array(
                [dic['±'] + dic['+'] + dic['-'], dic['±'] + dic['+'] + dic['-'], dic['±'] + dic['+'] + dic['-']])
        else:
            m_3 = m_3
        ocr_data[i]['max_medium_min'] = m_3
    return ocr_data


def bind_data(yolox_num, ocr_data):
    '''
    按照特殊yolox的框线将一个或者多个dbnet的框线合并，并把标注合并
    'key_info': [['3.505'], ['3.445']]
    '''
    new_ocr_data = []
    remember_no_arr = np.zeros(len(ocr_data))  # 记录dbnet的ocr是否被合并
    remember_no_arr_yolox = np.zeros(len(yolox_num))  # 记录yolox的框线是否用于合并
    # 1.针对每个yolox检测的data，看是否有两个及其以上的dbnet数据与之相交重叠
    for i in range(len(yolox_num)):
        k = 0
        x = np.zeros((len(ocr_data)))

        for j in range(len(ocr_data)):
            if remember_no_arr[j] == 0:
                if not (yolox_num[i][0] > ocr_data[j]['location'][2] or yolox_num[i][2] < ocr_data[j]['location'][
                    0]):  # 两矩形在x坐标上的长有重叠
                    if not (yolox_num[i][1] > ocr_data[j]['location'][3] or yolox_num[i][3] < ocr_data[j]['location'][
                        1]):  # 两矩形在y坐标上的高有重叠
                        k += 1
                        x[j] = 1
        # 2.如果有的重叠data框，则将yolox的框添加到新的dbnet的data框
        if k > 0:
            new_ocr_data_mid = {'location': yolox_num[i], 'ocr_strings': '', 'key_info': [],
                                'matched_pairs_location': [], 'matched_pairs_outside_or_inside': [],
                                'matched_pairs_yinXian': [], 'Absolutely': [], 'max_medium_min': []}
            for m in range(len(x)):
                if x[m] == 1:
                    new_ocr_data_mid['ocr_strings'] += (' ' + ocr_data[m]['ocr_strings'])
                    new_ocr_data_mid['key_info'].append(ocr_data[m]['key_info'])
                    if ocr_data[m]['Absolutely'] != []:
                        new_ocr_data_mid['Absolutely'] = ocr_data[m]['Absolutely']
            new_ocr_data.append(new_ocr_data_mid)

            remember_no_arr_yolox[i] += 1
            for l in range(len(x)):
                if x[l] == 1:
                    remember_no_arr[l] = 1

    # 3.将剩下的没有与yolox框重合的dbnet的data框添加到新的dbnet的data
    for i in range(len(ocr_data)):
        if remember_no_arr[i] == 0:
            ocr_data[i]['key_info'] = [ocr_data[i]['key_info']]
            new_ocr_data.append(ocr_data[i])

    # 4将剩下的没有与dbnet框重合的yolox框添加到新的dbnet的data
    # for i in range(len(yolox_num)):
    #     if remember_no_arr_yolox[i] == 0:
    #         new_dbnet_data = np.r_[new_dbnet_data, [yolox_num[i]]]
    # # new_dbnet_data = new_dbnet_data[1:,]
    # print(new_ocr_data)
    # for i in range(len(new_ocr_data)):
    #     if np.array(new_ocr_data[i]['key_info']).ndim == 1:
    #         new_ocr_data[i]['key_info'] = [new_ocr_data[i]['key_info']]

    return new_ocr_data

def ocr_get_data_onnx(image_path,
                      yolox_pairs):  # 输入yolox输出的pairs坐标和匹配的data坐标以及图片地址，ocr识别文本后输出data内容按序保存在data_list_np（numpy二维数组）
    ocr_data = []  # 按序存储pairs的data
    dt_boxes = []
    # 裁剪识别区域的时候需要扩展一圈，以防yolox极限检测框导致某些数据边缘没有被检测
    for i in range(len(yolox_pairs)):
        KuoZhan_x = 0
        KuoZhan_y = 0
        dt_boxes_middle = np.array([[yolox_pairs[i][0] - KuoZhan_x, yolox_pairs[i][1] - KuoZhan_y],
                                    [yolox_pairs[i][2] + KuoZhan_x, yolox_pairs[i][1] - KuoZhan_y],
                                    [yolox_pairs[i][2] + KuoZhan_x, yolox_pairs[i][3] + KuoZhan_y],
                                    [yolox_pairs[i][0] - KuoZhan_x, yolox_pairs[i][3] + KuoZhan_y]], dtype=np.float32)
        dt_boxes.append(dt_boxes_middle)
    ocr_data = Run_onnx(image_path, dt_boxes)
    return ocr_data

def ocr_data(img_path, dbnet_data):
    # ocr_get_data(img_path,top_yolox_num)
    dbnet_data = ocr_get_data_onnx(img_path, dbnet_data)
    return dbnet_data

def get_img_info(img_path):
    # import cv2
    image = cv2.imread(img_path)
    size = image.shape
    w = size[1]  # 宽度
    h = size[0]  # 高度
    return w, h

def show_data(img_path, data):
    wh_key1 = True
    while wh_key1:
        auto_key = input("是否展示dbnet框选的尺寸数字:y/n:")

        if auto_key == 'y' or auto_key == 'Y':
            auto_bool = False
            wh_key1 = False
        elif auto_key == 'n' or auto_key == 'N':
            auto_bool = True
            wh_key1 = False
        else:
            print("未输入正确，请重新输入：y/n:")
            wh_key1 = True
    if auto_bool == False:
        # import numpy as np
        # import cv2 as cv

        with open(img_path, 'rb') as f:
            np_arr = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # 以彩图读取
            # img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)  # 以灰度图读取

            for i in range(len(data)):
                # 绘制一个红色矩形
                ptLeftTop = (int(data[i][0]), int(data[i][1]))
                ptRightBottom = (int(data[i][2]), int(data[i][3]))
                point_color = (0, 0, 255)  # BGR
                thickness = 2
                lineType = 8
                cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)

        cv2.namedWindow("data(red)", 0)
        cv2.imshow('data(red)', img)
        cv2.waitKey(0)  # 显示 10000 ms 即 10s 后消失
        cv2.destroyAllWindows()

def show_ocr_result(img_path, ocr):
    # import numpy as np
    # import cv2 as cv
    wh_key1 = True
    while wh_key1:
        auto_key = input("是否展示ocr识别的标注:y/n:")

        if auto_key == 'y' or auto_key == 'Y':
            auto_bool = False
            wh_key1 = False
        elif auto_key == 'n' or auto_key == 'N':
            auto_bool = True
            wh_key1 = False
        else:
            print("未输入正确，请重新输入：y/n:")
            wh_key1 = True
    if auto_bool == False:

        img = cv2.imread(img_path)
        for i in range(len(ocr)):
            # 绘制一个红色矩形
            ptLeftTop = (int(ocr[i]['location'][0]), int(ocr[i]['location'][1]))
            ptRightBottom = (int(ocr[i]['location'][2]), int(ocr[i]['location'][3]))
            point_color = (0, 0, 255)  # BGR
            thickness = 2
            lineType = 8
            cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)

            # 调用cv.putText()添加文字
            if ocr[i]['max_medium_min'] != []:
                text = str(ocr[i]['max_medium_min'][0]) + ',' + str(ocr[i]['max_medium_min'][1]) + ',' + str(
                    ocr[i]['max_medium_min'][2])
                # AddText = img.copy()
                cv2.putText(img, text, (int(ocr[i]['location'][0]), int(ocr[i]['location'][1])),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 139), 2)
            # import numpy as np
            # 将原图片和添加文字后的图片拼接起来
            # img = np.hstack([img, AddText])

        cv2.namedWindow("data(red)", 0)
        cv2.imshow('data(red)', img)
        cv2.waitKey(0)  # 显示 10000 ms 即 10s 后消失
        cv2.destroyAllWindows()
    wh_key1 = True
    while wh_key1:
        auto_key = input("是否修改ocr识别的标注:y/n:")

        if auto_key == 'y' or auto_key == 'Y':
            auto_bool = False
            wh_key1 = False
        elif auto_key == 'n' or auto_key == 'N':
            auto_bool = True
            wh_key1 = False
        else:
            print("未输入正确，请重新输入：y/n:")
            wh_key1 = True
    if auto_bool == False:
        for i in range(len(ocr)):
            img = cv2.imread(img_path)
            # 绘制一个红色矩形
            ptLeftTop = (int(ocr[i]['location'][0]), int(ocr[i]['location'][1]))
            ptRightBottom = (int(ocr[i]['location'][2]), int(ocr[i]['location'][3]))
            point_color = (0, 0, 255)  # BGR
            thickness = 2
            lineType = 8
            cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)

            cv2.namedWindow("data_to_be_correct", 0)
            cv2.imshow('data_to_be_correct', img)
            cv2.waitKey(0)  # 显示 10000 ms 即 10s 后消失
            cv2.destroyAllWindows()
            print("修改格式: max medium min，正在修改第", i + 1, "个")
            print("ocr识别结果max medium min：", ocr[i])
            right_data = input("请输入修改值，无需修改直接回车:")
            str_list = right_data.split()
            ocr[i]['max_medium_min'] = str_list
    return ocr


def io_1(ocr_data):
    '''
    # yolox_pairs_top,np.二维数组（，11）[pairs_x1_y1_x2_y2,标注x1_y1_x2_y2，max,medium,min]
    '''
    result = np.zeros((0, 11))
    for i in range(len(ocr_data)):
        mid = np.zeros((11))
        # 获取 location 数据
        loc = ocr_data[i].get('matched_pairs_location')

        if loc is not None and len(loc) > 0:
            # 无论 loc 是 4 位还是 13 位，只取前 4 位 (x1, y1, x2, y2)
            mid[0: 4] = np.array(loc)[:4]
        else:
            mid[0: 4] = np.array([0, 0, 0, 0])

        mid[4: 8] = ocr_data[i]['location']
        if ocr_data[i]['max_medium_min'] != []:
            mid[8: 11] = ocr_data[i]['max_medium_min']
        else:
            mid[8: 11] = np.array([0, 0, 0])
        result = np.r_[result, [mid]]
    return result

def get_pairs_info(ocr_data, yolox_pairs_copy):
    '''
    0 = outside 1 = inside
    '''
    for i in range(len(yolox_pairs_copy)):
        for j in range(len(ocr_data)):
            if ocr_data[j]['matched_pairs_location'] != []:
                if (yolox_pairs_copy[i][4] == 0):
                    ocr_data[j]['matched_pairs_outside_or_inside'] = 'outside'
                if (yolox_pairs_copy[i][4] == 1):
                    ocr_data[j]['matched_pairs_outside_or_inside'] = 'inside'
    return ocr_data

def get_yinxian_info(ocr_data, yolox_pairs_length):
    '''
    top_yolox_pairs_length np.二维数组（，13）[pairs_x1_y1_x2_y2,引线1_x1_y1_x2_y2,引线2_x1_y1_x2_y2,两引线距离]
    '''
    for i in range(len(yolox_pairs_length)):
        for j in range(len(ocr_data)):
            loc = ocr_data[j].get('matched_pairs_location')

            if loc is not None and len(loc) > 0:
                loc_coords = np.array(loc)[:4]
                if np.array_equal(yolox_pairs_length[i][0: 4], loc_coords):
                    ocr_data[j]['matched_pairs_yinXian'] = yolox_pairs_length[i][4: 12]

    return ocr_data

def Divide_regions_ocr(ocr, border):
    '''
    根据border给标注划分区域，并给出值标记是否处于该区域的中心区域
    '''
    for i in range(len(ocr)):
        point_x = (ocr[i]['location'][0] + ocr[i]['location'][2]) * 0.5
        point_y = (ocr[i]['location'][1] + ocr[i]['location'][3]) * 0.5
        line1_start_x = border[0]
        line1_start_y = border[1]
        line1_end_x = border[2]
        line1_end_y = border[3]

        m1 = (line1_start_y - line1_end_y) / (line1_start_x - line1_end_x)
        c1 = line1_start_y - m1 * line1_start_x

        line2_start_x = border[2]
        line2_start_y = border[1]
        line2_end_x = border[0]
        line2_end_y = border[3]

        m2 = (line2_start_y - line2_end_y) / (line2_start_x - line2_end_x)
        c2 = line2_start_y - m2 * line2_start_x

        if (ocr[i]['location'][0] + ocr[i]['location'][2]) * 0.5 < border[0]:
            if point_y < m1 * point_x + c1:
                ocr[i]['region'] = 'top'
            elif point_y > m2 * point_x + c2:
                ocr[i]['region'] = 'bottom'
            else:
                ocr[i]['region'] = 'left'
        elif (ocr[i]['location'][0] + ocr[i]['location'][2]) * 0.5 > border[2]:
            if point_y < m2 * point_x + c2:
                ocr[i]['region'] = 'top'
            elif point_y > m1 * point_x + c1:
                ocr[i]['region'] = 'bottom'
            else:
                ocr[i]['region'] = 'right'
        else:
            if point_y < border[1]:
                ocr[i]['region'] = 'top'
            elif border[1] < point_y < border[2]:
                ocr[i]['region'] = 'center'
            else:
                ocr[i]['region'] = 'bottom'
    return ocr

def Divide_regions_pairs(pairs, border):
    pairs_region = []
    for i in range(len(pairs)):
        point_x = (pairs[i][0] + pairs[i][2]) * 0.5
        point_y = (pairs[i][1] + pairs[i][3]) * 0.5
        line1_start_x = border[0]
        line1_start_y = border[1]
        line1_end_x = border[2]
        line1_end_y = border[3]

        m1 = (line1_start_y - line1_end_y) / (line1_start_x - line1_end_x)
        c1 = line1_start_y - m1 * line1_start_x

        line2_start_x = border[2]
        line2_start_y = border[1]
        line2_end_x = border[0]
        line2_end_y = border[3]

        m2 = (line2_start_y - line2_end_y) / (line2_start_x - line2_end_x)
        c2 = line2_start_y - m2 * line2_start_x

        if (pairs[i][0] + pairs[i][2]) * 0.5 < border[0]:
            if point_y < m1 * point_x + c1:
                pairs_region.append('top')
            elif point_y > m2 * point_x + c2:
                pairs_region.append('bottom')
            else:
                pairs_region.append('left')
        elif (pairs[i][0] + pairs[i][2]) * 0.5 > border[2]:
            if point_y < m2 * point_x + c2:
                pairs_region.append('top')
            elif point_y > m1 * point_x + c1:
                pairs_region.append('bottom')
            else:
                pairs_region.append('right')
        else:
            if point_y < border[1]:
                pairs_region.append('top')
            elif border[1] < point_y < border[2]:
                pairs_region.append('center')
            else:
                pairs_region.append('bottom')
    return pairs_region

def match_pairs_data(img_path, pairs, ocr, border):  # pairs[[0,1,2,3],[0,1,2,3]];data[[0,1,2,3,m,m,m],[0,1,2,3,m,m,m]]
    '''
    ocr= {'location': yolox_num[i], 'ocr_strings': '', 'key_info': [],
               'matched_pairs_location': [], 'matched_pairs_outside_or_inside': [],
               'matched_pairs_yinXian': [], 'Absolutely': [], 'max_medium_min': []}
    '''
    print("---开始视图的标注和标尺线的匹配---")
    divide_key = 0  # 是否需要分割区域匹配
    # 仅在特定视图下采用分区域匹配
    if img_path == "data/top.jpg" or img_path == "data/bottom.jpg":
        if not (np.array_equal(border, np.zeros((1, 4))) or np.array_equal(border, np.zeros((0, 4)))):
            border = [border[0][0], border[0][1], border[0][2], border[0][3]]
            ocr = Divide_regions_ocr(ocr, border)
            pairs_region = Divide_regions_pairs(pairs, border)
            print("---分割完成---")
            print(pairs_region)
            print(ocr)
            divide_key = 1

    # 设置最大匹配距离，超过这一距离无法匹配在一起
    w, h = get_img_info(img_path)
    max_length = round(min(w, h) * 0.5, 2)
    matched_ocr = []  # 匹配的尺寸线和尺寸数字存在这里
    middle_arr = np.zeros((11))  # 作为中间量存入matched_pairs_data
    matched_data = np.zeros((len(ocr)))  # 标记是否该data被匹配
    matched_pairs = np.zeros((len(pairs)))  # 标记是否该pairs被匹配
    # 1.直接匹配和标尺线重叠的标注，并从标注池中删除。完成后删除匹配到的标注，保证一个标注可以和多个重叠标注匹配
    for i in range(len(pairs)):
        for j in range(len(ocr)):
            if ocr[j]['Absolutely'] != 'angle':
                if not (pairs[i][0] > ocr[j]['location'][2] or pairs[i][2] < ocr[j]['location'][0]):  # 两矩形在x坐标上的长有重叠
                    if not (pairs[i][1] > ocr[j]['location'][3] or pairs[i][3] < ocr[j]['location'][1]):  # 两矩形在y坐标上的高有重叠
                        matched_data[j] = 1
                        matched_pairs[i] = 1
                        ocr[j]['matched_pairs_location'] = pairs[i]
                        matched_ocr.append(ocr[j])
                        print("匹配有重叠的标尺线与标注\n", ocr[j])

    ruler_match = matched_ocr.copy()
    # print(matched_pairs_data)
    new_ocr = []  # 过滤已经匹配了的数据的尺寸数字数组
    new_pairs_region = []
    for i in range(len(ocr)):
        if matched_data[i] == 0:
            new_ocr.append(ocr[i])
    new_pairs = np.zeros((0, 4))
    for i in range(len(pairs)):
        if matched_pairs[i] == 0:
            current_item = np.array(pairs[i])
            # 2. 强制截断为4列（不管原长度是多少，只保留前4列）
            current_item_truncated = current_item[:4]  # 核心：把5列砍成4列
            # ========== 预处理结束 ==========
            
            # 3. 用截断后的数组拼接（替换原来的 [pairs[i]]）
            new_pairs = np.r_[new_pairs, [current_item_truncated]]
            
            if divide_key == 1:
                new_pairs_region.append(pairs_region[i])
    # 2.针对没有重叠的pairs，（1）横向pair只能匹配横向data（2）竖向pair可能匹配横向或者竖向data
    # 将pairs和data重叠的作为标尺，其他匹配的pairs和data比例不能过大
    # 横向pairs 1.重叠的data 2.同方向且中心轴相同的离得近的data（紧贴） 3.横向水平且离得近（欧氏距离不超过两个pairs长）4.欧氏距离
    # 竖向pairs 1.重叠的data 2.同方向且中心轴相同的离得近的data（紧贴） 3.竖向水平且离得近（欧氏距离不超过两个pairs长）4.欧氏距离
    # 每次匹配三个data并排序，如果有条件差不多的data：找相近的pairs按照pairs长度分别将两个data匹配
    # 3.方法：同方向且中心轴相同的离得近的data（紧贴）#####pairs的方向按照长宽比判断，但方向区分度不大的pairs不能判断方向
    # 3.实现版本：找pairs中心轴穿过的data，根据水平距离排序再通过标尺筛选
    # 可以改进：根据ocr识别的图片判断data方向而不是根据data长宽
    # print('***', new_ocr)
    #####################################横向
    mid_arr = np.zeros((8))  # 前七位存储数据，第八位存储尺寸数据和尺寸线的最近距离
    matched_new_data = np.zeros((len(new_ocr)))  # 标记是否该data被匹配
    matched_new_pairs = np.zeros((len(new_pairs)))  # 标记是否该pairs被匹配
    for i in range(len(new_pairs)):
        # mid_ocr_acc = np.zeros((len(new_ocr)))  # 记录哪些是可能的
        maybe_match_data = np.zeros((0, 8))  # 存储可能的尺寸数字，前七位存储数据，第八位存储尺寸数据和尺寸线的最近距离

        if (min((new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1])) / max(
                (new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1]))) < 0.75:
            if new_pairs[i][2] - new_pairs[i][0] > new_pairs[i][3] - new_pairs[i][1]:  # 横向pairs
                for j in range(len(new_ocr)):
                    if new_ocr[j]['Absolutely'] != 'angle':
                        if new_ocr[j]['location'][2] - new_ocr[j]['location'][0] > new_ocr[j]['location'][3] - \
                                new_ocr[j]['location'][1]:  # 横向data
                            if (new_pairs[i][2] + new_pairs[i][0]) * 0.5 < new_ocr[j]['location'][2] and (
                                    new_pairs[i][2] + new_pairs[i][0]) * 0.5 > new_ocr[j]['location'][
                                0]:  # 横向pairs中心轴穿过横向data
                                if (new_pairs[i][3] - new_pairs[i][1]) * 4 > min(
                                        abs(new_pairs[i][3] - new_ocr[j]['location'][1]), abs(
                                            new_pairs[i][1] - new_ocr[j]['location'][3])):  # pairs与data距离不超过pairs高度2倍
                                    mid_arr[0:4] = new_ocr[j]['location']
                                    mid_arr[4: 7] = new_ocr[j]['max_medium_min']
                                    mid_arr[7] = min(abs(new_pairs[i][3] - new_ocr[j]['location'][1]),
                                                     abs(new_pairs[i][1] - new_ocr[j]['location'][3]))
                                    if divide_key == 1:
                                        if new_ocr[j]['region'] == new_pairs_region[i]:
                                            maybe_match_data = np.r_[maybe_match_data, [mid_arr]]
                                    else:
                                        maybe_match_data = np.r_[maybe_match_data, [mid_arr]]
                                    # mid_ocr_acc[j] = len(maybe_match_data)

                maybe_match_data = maybe_match_data[np.argsort(maybe_match_data[:, 7])]  # 按距离从小到大排序
                # 当ruler中data比maybe中data大时，ruler中pairs一定比data中
                for k in range(len(maybe_match_data)):
                    buer = True
                    for l in range(len(ruler_match)):
                        if (max(ruler_match[l]['matched_pairs_location'][2] - ruler_match[l]['matched_pairs_location'][
                            0], ruler_match[l]['matched_pairs_location'][3] - ruler_match[l]['matched_pairs_location'][
                            1]) - max(
                            new_pairs[i][2] - new_pairs[i][0], new_pairs[i][3] - new_pairs[i][1])) * (
                                ruler_match[l]['max_medium_min'][0] - maybe_match_data[k][4]) < 0:
                            buer = False
                            break
                    if buer == True:
                        for m in range(len(new_ocr)):
                            if (new_ocr[m]['location'] == maybe_match_data[k][0:4]).all():
                                new_ocr[m]['matched_pairs_location'] = new_pairs[i]
                        print("匹配同方向且中心轴相同的离得近的标注（紧贴）\n", new_ocr[m])
                        matched_ocr.append(new_ocr[m])
                        matched_new_pairs[i] = 1
                        matched_new_data[m] = 1

    new_new_pairs = np.zeros((0, 4))
    new_new_pairs_region = []
    for i in range(len(new_pairs)):
        if matched_new_pairs[i] == 0:
            new_new_pairs = np.r_[new_new_pairs, [new_pairs[i]]]
            if divide_key == 1:
                new_new_pairs_region.append(new_pairs_region[i])
    new_pairs = new_new_pairs
    new_pairs_region = new_new_pairs_region
    new_new_data = []
    for i in range(len(new_ocr)):
        if matched_new_data[i] == 0:
            new_new_data.append(new_ocr[i])
    new_ocr = new_new_data
    # print('***', new_ocr)
    ###########################竖向
    mid_arr = np.zeros((8))
    matched_new_data = np.zeros((len(new_ocr)))
    matched_new_pairs = np.zeros((len(new_pairs)))
    for i in range(len(new_pairs)):
        # mid_ocr_acc = np.zeros((len(new_ocr)))  # 记录哪些是可能的
        maybe_match_data = np.zeros((0, 8))
        if (min((new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1])) / max(
                (new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1]))) < 0.75:
            if new_pairs[i][2] - new_pairs[i][0] < new_pairs[i][3] - new_pairs[i][1]:  # 竖向pairs
                for j in range(len(new_ocr)):
                    if new_ocr[j]['Absolutely'] != 'angle':
                        if new_ocr[j]['location'][2] - new_ocr[j]['location'][0] < new_ocr[j]['location'][3] - \
                                new_ocr[j]['location'][1]:  # 竖向data
                            if (new_pairs[i][3] + new_pairs[i][1]) * 0.5 < new_ocr[j]['location'][3] and (
                                    new_pairs[i][3] + new_pairs[i][1]) * 0.5 > new_ocr[j]['location'][
                                1]:  # 竖向pairs中心轴穿过竖向data
                                if (new_pairs[i][2] - new_pairs[i][0]) * 2 > min(
                                        abs(new_pairs[i][2] - new_ocr[j]['location'][0]), abs(
                                            new_pairs[i][0] - new_ocr[j]['location'][2])):  # pairs与data距离不超过pairs高度2倍
                                    mid_arr[0:4] = new_ocr[j]['location']
                                    mid_arr[4: 7] = new_ocr[j]['max_medium_min']
                                    mid_arr[7] = min(abs(new_pairs[i][3] - new_ocr[j]['location'][1]),
                                                     abs(new_pairs[i][1] - new_ocr[j]['location'][3]))
                                    if divide_key == 1:
                                        if new_ocr[j]['region'] == new_pairs_region[i]:
                                            maybe_match_data = np.r_[maybe_match_data, [mid_arr]]
                                    else:
                                        maybe_match_data = np.r_[maybe_match_data, [mid_arr]]

                                    # mid_ocr_acc[j] = len(maybe_match_data)
                maybe_match_data = maybe_match_data[np.argsort(maybe_match_data[:, 7])]  # 按距离从小到大排序
                # 当ruler中data比maybe中data大时，ruler中pairs一定比data中
                for k in range(len(maybe_match_data)):
                    buer = True
                    for l in range(len(ruler_match)):
                        if (max(ruler_match[l]['matched_pairs_location'][2] - ruler_match[l]['matched_pairs_location'][
                            0], ruler_match[l]['matched_pairs_location'][3] - ruler_match[l]['matched_pairs_location'][
                            1]) - max(
                            new_pairs[i][2] - new_pairs[i][0], new_pairs[i][3] - new_pairs[i][1])) * (
                                ruler_match[l]['max_medium_min'][0] - maybe_match_data[k][4]) < 0:
                            buer = False
                            break
                    if buer == True:
                        for m in range(len(new_ocr)):
                            if (new_ocr[m]['location'] == maybe_match_data[k][0:4]).all():
                                new_ocr[m]['matched_pairs_location'] = new_pairs[i]
                        print("匹配同方向且中心轴相同的离得近的data（紧贴）\n", new_ocr[m])
                        matched_ocr.append(new_ocr[m])
                        matched_new_pairs[i] = 1
                        matched_new_data[m] = 1
    new_new_pairs = np.zeros((0, 4))
    new_new_pairs_region = []
    for i in range(len(new_pairs)):
        if matched_new_pairs[i] == 0:
            new_new_pairs = np.r_[new_new_pairs, [new_pairs[i]]]
            if divide_key == 1:
                new_new_pairs_region.append(new_pairs_region[i])
    new_pairs = new_new_pairs
    new_pairs_region = new_new_pairs_region
    new_new_data = []
    for i in range(len(new_ocr)):
        if matched_new_data[i] == 0:
            new_new_data.append(new_ocr[i])
    new_ocr = new_new_data
    # print('***', new_ocr)
    # 4.1横向箭头横向水平(三个高范围内)且离得近（欧氏距离不超过两个pairs长）（横向箭头一般只会匹配横向数据）
    # 实际：横向箭头匹配数据（不管方向）：先在pairs的y坐标上有重叠的data填入待match，再将pairs高度扩充三倍，在y轴上与之相交的data按欧氏距离大小排列为待匹配，用ruler检验
    matched_new_pairs = np.zeros((len(new_pairs)))
    matched_new_data = np.zeros((len(new_ocr)))
    for i in range(len(new_pairs)):
        # mid_ocr_acc = np.zeros((len(new_ocr)))  # 记录哪些是可能的
        maybe_match_data_a = np.zeros((0, 7))
        maybe_match_data_b = np.zeros((0, 8))
        middle = np.zeros((8))
        if (min((new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1])) / max(
                (new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1]))) < 0.75:
            if new_pairs[i][2] - new_pairs[i][0] > new_pairs[i][3] - new_pairs[i][1]:  # 横向pairs
                for j in range(len(new_ocr)):
                    if new_ocr[j]['Absolutely'] != 'angle':
                        if not (new_pairs[i][1] > new_ocr[j]['location'][3] or new_pairs[i][3] < new_ocr[j]['location'][
                            1]):  # 两矩形在y坐标上的高有重叠
                            if min(abs(new_ocr[j]['location'][0] - new_pairs[i][2]),
                                   abs(new_ocr[j]['location'][2] - new_pairs[i][0])) < 5 * (
                                    new_pairs[i][3] - new_pairs[i][1]):  # pairs和data横向距离很近
                                mid = np.zeros(7)
                                mid[0: 4] = new_ocr[j]['location']
                                mid[4: 7] = new_ocr[j]['max_medium_min']
                                if divide_key == 1:
                                    if new_ocr[j]['region'] == new_pairs_region[i]:
                                        maybe_match_data_a = np.r_[maybe_match_data_a, [mid]]
                                else:
                                    maybe_match_data_a = np.r_[maybe_match_data_a, [mid]]

                                # mid_ocr_acc[j] = len(maybe_match_data_a)
                        if not (new_pairs[i][1] - (new_pairs[i][3] - new_pairs[i][1]) * 2 > new_ocr[j]['location'][3] or
                                new_pairs[i][
                                    3] + (new_pairs[i][3] - new_pairs[i][1]) * 2 < new_ocr[j]['location'][
                                    1]):  # 两矩形在y坐标上的高有重叠
                            if min(abs(new_ocr[j]['location'][0] - new_pairs[i][2]),
                                   abs(new_ocr[j]['location'][2] - new_pairs[i][0])) < 7 * (
                                    new_pairs[i][3] - new_pairs[i][1]):
                                if new_ocr[j]['location'] not in maybe_match_data_a[0: 4]:
                                    middle[0:4] = new_ocr[j]['location']
                                    middle[4:7] = new_ocr[j]['max_medium_min']
                                    middle[7] = min(abs(new_ocr[j]['location'][0] - new_pairs[i][2]),
                                                    abs(new_ocr[j]['location'][2] - new_pairs[i][0]))
                                    if divide_key == 1:
                                        if new_ocr[j]['region'] == new_pairs_region[i]:
                                            maybe_match_data_b = np.r_[maybe_match_data_b, [middle]]
                                    else:
                                        maybe_match_data_b = np.r_[maybe_match_data_b, [middle]]
                                    # maybe_match_data_b = np.r_[maybe_match_data_b, [middle]]
                maybe_match_data_b = maybe_match_data_b[np.argsort(maybe_match_data_b[:, 7])]  # 按距离从小到大排序
                maybe_match_data_b = maybe_match_data_b[:, 0:7]
                maybe_match_data = np.append(maybe_match_data_a, maybe_match_data_b, axis=0)
                for k in range(len(maybe_match_data)):
                    buer = True
                    for l in range(len(ruler_match)):
                        if (max(ruler_match[l]['matched_pairs_location'][2] - ruler_match[l]['matched_pairs_location'][
                            0], ruler_match[l]['matched_pairs_location'][3] - ruler_match[l]['matched_pairs_location'][
                            1]) - max(
                            new_pairs[i][2] - new_pairs[i][0], new_pairs[i][3] - new_pairs[i][1])) * (
                                ruler_match[l]['max_medium_min'][0] - maybe_match_data[k][4]) < 0:
                            buer = False
                            break
                    if buer == True:
                        for m in range(len(new_ocr)):
                            if (new_ocr[m]['location'] == maybe_match_data[k][0:4]).all():
                                new_ocr[m]['matched_pairs_location'] = new_pairs[i]
                        print(
                            "横向箭头横向水平(三个高范围内)且离得近（欧氏距离不超过两个pairs长）（横向箭头一般只会匹配横向数据）\n",
                            new_ocr[m])
                        matched_ocr.append(new_ocr[m])
                        matched_new_pairs[i] = 1
                        matched_new_data[m] = 1
    new_new_pairs = np.zeros((0, 4))
    new_new_pairs_region = []
    for i in range(len(new_pairs)):
        if matched_new_pairs[i] == 0:
            new_new_pairs = np.r_[new_new_pairs, [new_pairs[i]]]
            if divide_key == 1:
                new_new_pairs_region.append(new_pairs_region[i])
    new_pairs = new_new_pairs
    new_pairs_region = new_new_pairs_region
    new_new_data = []
    for i in range(len(new_ocr)):
        if matched_new_data[i] == 0:
            new_new_data.append(new_ocr[i])
    new_ocr = new_new_data
    # print('***', new_ocr)
    # print("new_pairs,new_data",new_pairs,new_data)
    # print("matched_pairs_data",matched_pairs_data)
    # 4.2竖向箭头竖向水平(三个高范围内)且离得近（欧氏距离不超过两个pairs长）（竖向箭头可能匹配到两个方向的数据）
    # 实际：竖向箭头匹配数据（不管方向）：先在pairs的x坐标上有重叠的data填入待match，再将pairs宽度扩充三倍，在x轴上与之相交的data按欧氏距离大小排列为待匹配，用ruler检验
    matched_new_pairs = np.zeros((len(new_pairs)))
    matched_new_data = np.zeros((len(new_ocr)))
    for i in range(len(new_pairs)):
        maybe_match_data_a = np.zeros((0, 7))
        maybe_match_data_b = np.zeros((0, 8))
        middle = np.zeros((8))
        if (min((new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1])) / max(
                (new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1]))) < 0.75:
            if new_pairs[i][2] - new_pairs[i][0] < new_pairs[i][3] - new_pairs[i][1]:  # 竖向pairs
                for j in range(len(new_ocr)):
                    if new_ocr[j]['Absolutely'] != 'angle':
                        if not (pairs[i][0] > new_ocr[j]['location'][2] or pairs[i][2] < new_ocr[j]['location'][
                            0]):  # 两矩形在x坐标上的长有重叠
                            if min(abs(new_ocr[j]['location'][1] - new_pairs[i][3]),
                                   abs(new_ocr[j]['location'][3] - new_pairs[i][1])) < 5 * (
                                    new_pairs[i][2] - new_pairs[i][0]):  # pairs和data横向距离很近
                                mid = np.zeros(7)
                                mid[0: 4] = new_ocr[j]['location']
                                mid[4: 7] = new_ocr[j]['max_medium_min']
                                if divide_key == 1:
                                    if new_ocr[j]['region'] == new_pairs_region[i]:
                                        maybe_match_data_a = np.r_[maybe_match_data_a, [mid]]
                                else:
                                    maybe_match_data_a = np.r_[maybe_match_data_a, [mid]]
                                # maybe_match_data_a = np.r_[maybe_match_data_a, [mid]]
                        if not (new_pairs[i][0] - (new_pairs[i][2] - new_pairs[i][0]) * 2 > new_ocr[j]['location'][2] or
                                new_pairs[i][
                                    2] + (new_pairs[i][2] - new_pairs[i][0]) * 2 < new_ocr[j]['location'][
                                    0]):  # 两矩形在y坐标上的高有重叠
                            if min(abs(new_ocr[j]['location'][1] - new_pairs[i][3]),
                                   abs(new_ocr[j]['location'][3] - new_pairs[i][1])) < 7 * (
                                    new_pairs[i][2] - new_pairs[i][0]):
                                if new_ocr[j]['location'] not in maybe_match_data_a[0: 4]:
                                    middle[0:4] = new_ocr[j]['location']
                                    middle[4:7] = new_ocr[j]['max_medium_min']
                                    middle[7] = min(abs(new_ocr[j]['location'][1] - new_pairs[i][3]),
                                                    abs(new_ocr[j]['location'][3] - new_pairs[i][1]))
                                    if divide_key == 1:
                                        if new_ocr[j]['region'] == new_pairs_region[i]:
                                            maybe_match_data_b = np.r_[maybe_match_data_b, [middle]]
                                    else:
                                        maybe_match_data_b = np.r_[maybe_match_data_b, [middle]]
                                    # maybe_match_data_b = np.r_[maybe_match_data_b, [middle]]
                maybe_match_data_b = maybe_match_data_b[np.argsort(maybe_match_data_b[:, 7])]  # 按距离从小到大排序
                maybe_match_data_b = maybe_match_data_b[:, 0:7]
                maybe_match_data = np.append(maybe_match_data_a, maybe_match_data_b, axis=0)
                for k in range(len(maybe_match_data)):
                    buer = True
                    for l in range(len(ruler_match)):
                        if (max(ruler_match[l]['matched_pairs_location'][2] - ruler_match[l]['matched_pairs_location'][
                            0], ruler_match[l]['matched_pairs_location'][3] - ruler_match[l]['matched_pairs_location'][
                            1]) - max(
                            new_pairs[i][2] - new_pairs[i][0], new_pairs[i][3] - new_pairs[i][1])) * (
                                ruler_match[l]['max_medium_min'][0] - maybe_match_data[k][4]) < 0:
                            buer = False
                            break
                    if buer == True:
                        for m in range(len(new_ocr)):
                            if (new_ocr[m]['location'] == maybe_match_data[k][0:4]).all():
                                new_ocr[m]['matched_pairs_location'] = new_pairs[i]
                        print(
                            "竖向箭头竖向水平(三个高范围内)且离得近（欧氏距离不超过两个pairs长）\n",
                            new_ocr[m])
                        matched_ocr.append(new_ocr[m])
                        matched_new_pairs[i] = 1
                        matched_new_data[m] = 1
    new_new_pairs = np.zeros((0, 4))
    new_new_pairs_region = []
    for i in range(len(new_pairs)):
        if matched_new_pairs[i] == 0:
            new_new_pairs = np.r_[new_new_pairs, [new_pairs[i]]]
            if divide_key == 1:
                new_new_pairs_region.append(new_pairs_region[i])

    new_pairs = new_new_pairs
    new_pairs_region = new_new_pairs_region
    new_new_data = []
    for i in range(len(new_ocr)):
        if matched_new_data[i] == 0:
            new_new_data.append(new_ocr[i])
    new_ocr = new_new_data
    # print('***', new_ocr)
    # 5.剩余标尺线按欧式距离匹配
    # from math import sqrt
    right_matched_pairs = []
    x = len(new_pairs)
    while_count = 0
    while len(right_matched_pairs) != x and len(new_ocr) != 0 and len(new_pairs) != 0:
        matched_pairs = np.zeros((len(new_pairs)))
        matched_pairs_len = np.zeros((len(new_pairs)))
        matched_pairs[:] = -1  # 存储匹配到的data在new_data中的序号
        # 5.1.将所有pairs按照最近data匹配，记录匹配data序号和距离
        for i in range(len(new_pairs)):
            min_lenth = 99999
            min_no = -1
            for j in range(len(new_ocr)):
                if new_ocr[j]['Absolutely'] != 'angle':
                    lenth = sqrt(
                        (((new_pairs[i][2] + new_pairs[i][0]) * 0.5) - (
                                new_ocr[j]['location'][2] + new_ocr[j]['location'][0]) * 0.5) ** 2 + (
                                ((new_pairs[i][3] + new_pairs[i][1]) * 0.5) - (
                                new_ocr[j]['location'][3] + new_ocr[j]['location'][1]) * 0.5) ** 2)
                    if lenth < min_lenth:
                        min_lenth = lenth
                        min_no = j
            if min_no != -1 and min_lenth < max_length:
                matched_pairs[i] = min_no
                matched_pairs_len[i] = min_lenth
        # 5.2.将相同匹配的pairs中距离大的项清零
        for i in range(len(matched_pairs)):
            if matched_pairs[i] != -1:
                for j in range(len(matched_pairs)):
                    if matched_pairs[j] != -1:
                        if i != j and matched_pairs[i] == matched_pairs[j]:
                            if matched_pairs_len[i] > matched_pairs_len[j]:
                                matched_pairs[i] = -1
                                matched_pairs_len[i] = 0
                            else:
                                matched_pairs[j] = -1
                                matched_pairs_len[j] = 0
        # 5.3.将未匹配的data和pairs分离重复1 2 3直到
        no_matched_pairs = np.zeros((0, 4))
        no_matched_data = []
        for i in range(len(new_pairs)):
            if matched_pairs[i] == -1:
                no_matched_pairs = np.r_[no_matched_pairs, [new_pairs[i]]]
        for i in range(len(new_ocr)):
            if i not in matched_pairs:
                no_matched_data.append(new_ocr[i])

        middle = np.zeros((11))
        for i in range(len(new_pairs)):
            if matched_pairs[i] != -1:
                new_ocr[int(matched_pairs[i])]['matched_pairs_location'] = new_pairs[i]
                right_matched_pairs.append(new_ocr[int(matched_pairs[i])])
        new_pairs = no_matched_pairs
        new_ocr = no_matched_data
        # 限定循环次数，防止死循环（不保证所有标尺线会匹配到标注）
        while_count += 1
        if while_count == 3:
            break
    # 5.4将剩余data添加pairs为空传到最后结果
    if len(new_ocr) != 0:
        for i in range(len(new_ocr)):
            right_matched_pairs.append(new_ocr[i])
    # 5.5将剩余pairs添加data为空传到最后结果
    # middle = np.zeros((11))
    # if len(new_pairs) != 0:
    #     for i in range(len(new_pairs)):
    #         middle[0: 4] = new_pairs[i]
    #         middle[4: 11] = np.array([0, 0, 0, 0, 0, 0, 0])
    #         right_matched_pairs = np.r_[right_matched_pairs, [middle]]
    # 输出匹配
    for i in range(len(right_matched_pairs)):
        matched_ocr.append(right_matched_pairs[i])

    result = []
    for i in range(len(matched_ocr)):
        if i == 0:
            result.append(matched_ocr[i])
            continue

        is_duplicate = False
        for j in range(len(result)):
            # 使用 np.array_equal 替代直接的 == 比较
            if np.array_equal(matched_ocr[i]['location'], result[j]['location']):
                is_duplicate = True
                break

        if not is_duplicate:
            result.append(matched_ocr[i])
    print("---结束视图的标注和标尺线的匹配---")
    return result

def match_pairs_data_angle(img_path, pairs, ocr, border):  # pairs[[0,1,2,3],[0,1,2,3]];data[[0,1,2,3,m,m,m],[0,1,2,3,m,m,m]]
    '''
    ocr= {'location': yolox_num[i], 'ocr_strings': '', 'key_info': [],
               'matched_pairs_location': [], 'matched_pairs_outside_or_inside': [],
               'matched_pairs_yinXian': [], 'Absolutely': [], 'max_medium_min': []}
    '''
    print("---开始视图的标注和标尺线的匹配---")
    divide_key = 0  # 是否需要分割区域匹配
    # 仅在特定视图下采用分区域匹配
    if img_path == "data/top.jpg" or img_path == "data/bottom.jpg":
        if not (np.array_equal(border, np.zeros((1, 4))) or np.array_equal(border, np.zeros((0, 4)))):
            border = [border[0][0], border[0][1], border[0][2], border[0][3]]
            ocr = Divide_regions_ocr(ocr, border)
            pairs_region = Divide_regions_pairs(pairs, border)
            print("---分割完成---")
            print(pairs_region)
            print(ocr)
            divide_key = 1

    # 设置最大匹配距离，超过这一距离无法匹配在一起
    w, h = get_img_info(img_path)
    max_length = round(min(w, h) * 0.5, 2)
    matched_ocr = []  # 匹配的尺寸线和尺寸数字存在这里
    middle_arr = np.zeros((11))  # 作为中间量存入matched_pairs_data
    matched_data = np.zeros((len(ocr)))  # 标记是否该data被匹配
    matched_pairs = np.zeros((len(pairs)))  # 标记是否该pairs被匹配
    # 1.直接匹配和标尺线重叠的标注，并从标注池中删除。完成后删除匹配到的标注，保证一个标注可以和多个重叠标注匹配
    for i in range(len(pairs)):
        for j in range(len(ocr)):
            if ocr[j]['Absolutely'] == 'angle':
                if not (pairs[i][0] > ocr[j]['location'][2] or pairs[i][2] < ocr[j]['location'][0]):  # 两矩形在x坐标上的长有重叠
                    if not (pairs[i][1] > ocr[j]['location'][3] or pairs[i][3] < ocr[j]['location'][1]):  # 两矩形在y坐标上的高有重叠
                        matched_data[j] = 1
                        matched_pairs[i] = 1
                        ocr[j]['matched_pairs_location'] = pairs[i]
                        matched_ocr.append(ocr[j])
                        print("匹配有重叠的标尺线与标注\n", ocr[j])

    ruler_match = matched_ocr.copy()
    # print(matched_pairs_data)
    new_ocr = []  # 过滤已经匹配了的数据的尺寸数字数组
    new_pairs_region = []
    for i in range(len(ocr)):
        if matched_data[i] == 0:
            new_ocr.append(ocr[i])
    new_pairs = np.zeros((0, 4))
    for i in range(len(pairs)):
        if matched_pairs[i] == 0:
            new_pairs = np.r_[new_pairs, [pairs[i]]]
            if divide_key == 1:
                new_pairs_region.append(pairs_region[i])
    # 2.针对没有重叠的pairs，（1）横向pair只能匹配横向data（2）竖向pair可能匹配横向或者竖向data
    # 将pairs和data重叠的作为标尺，其他匹配的pairs和data比例不能过大
    # 横向pairs 1.重叠的data 2.同方向且中心轴相同的离得近的data（紧贴） 3.横向水平且离得近（欧氏距离不超过两个pairs长）4.欧氏距离
    # 竖向pairs 1.重叠的data 2.同方向且中心轴相同的离得近的data（紧贴） 3.竖向水平且离得近（欧氏距离不超过两个pairs长）4.欧氏距离
    # 每次匹配三个data并排序，如果有条件差不多的data：找相近的pairs按照pairs长度分别将两个data匹配
    # 3.方法：同方向且中心轴相同的离得近的data（紧贴）#####pairs的方向按照长宽比判断，但方向区分度不大的pairs不能判断方向
    # 3.实现版本：找pairs中心轴穿过的data，根据水平距离排序再通过标尺筛选
    # 可以改进：根据ocr识别的图片判断data方向而不是根据data长宽
    # print('***', new_ocr)
    #####################################横向
    mid_arr = np.zeros((8))  # 前七位存储数据，第八位存储尺寸数据和尺寸线的最近距离
    matched_new_data = np.zeros((len(new_ocr)))  # 标记是否该data被匹配
    matched_new_pairs = np.zeros((len(new_pairs)))  # 标记是否该pairs被匹配
    for i in range(len(new_pairs)):
        # mid_ocr_acc = np.zeros((len(new_ocr)))  # 记录哪些是可能的
        maybe_match_data = np.zeros((0, 8))  # 存储可能的尺寸数字，前七位存储数据，第八位存储尺寸数据和尺寸线的最近距离

        if (min((new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1])) / max(
                (new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1]))) < 0.75:
            if new_pairs[i][2] - new_pairs[i][0] > new_pairs[i][3] - new_pairs[i][1]:  # 横向pairs
                for j in range(len(new_ocr)):
                    if new_ocr[j]['Absolutely'] == 'angle':
                        if new_ocr[j]['location'][2] - new_ocr[j]['location'][0] > new_ocr[j]['location'][3] - \
                                new_ocr[j]['location'][1]:  # 横向data
                            if (new_pairs[i][2] + new_pairs[i][0]) * 0.5 < new_ocr[j]['location'][2] and (
                                    new_pairs[i][2] + new_pairs[i][0]) * 0.5 > new_ocr[j]['location'][
                                0]:  # 横向pairs中心轴穿过横向data
                                if (new_pairs[i][3] - new_pairs[i][1]) * 4 > min(
                                        abs(new_pairs[i][3] - new_ocr[j]['location'][1]), abs(
                                            new_pairs[i][1] - new_ocr[j]['location'][3])):  # pairs与data距离不超过pairs高度2倍
                                    mid_arr[0:4] = new_ocr[j]['location']
                                    mid_arr[4: 7] = new_ocr[j]['max_medium_min']
                                    mid_arr[7] = min(abs(new_pairs[i][3] - new_ocr[j]['location'][1]),
                                                     abs(new_pairs[i][1] - new_ocr[j]['location'][3]))
                                    if divide_key == 1:
                                        if new_ocr[j]['region'] == new_pairs_region[i]:
                                            maybe_match_data = np.r_[maybe_match_data, [mid_arr]]
                                    else:
                                        maybe_match_data = np.r_[maybe_match_data, [mid_arr]]
                                    # mid_ocr_acc[j] = len(maybe_match_data)

                maybe_match_data = maybe_match_data[np.argsort(maybe_match_data[:, 7])]  # 按距离从小到大排序
                # 当ruler中data比maybe中data大时，ruler中pairs一定比data中
                for k in range(len(maybe_match_data)):
                    buer = True
                    for l in range(len(ruler_match)):
                        if (max(ruler_match[l]['matched_pairs_location'][2] - ruler_match[l]['matched_pairs_location'][
                            0], ruler_match[l]['matched_pairs_location'][3] - ruler_match[l]['matched_pairs_location'][
                            1]) - max(
                            new_pairs[i][2] - new_pairs[i][0], new_pairs[i][3] - new_pairs[i][1])) * (
                                ruler_match[l]['max_medium_min'][0] - maybe_match_data[k][4]) < 0:
                            buer = False
                            break
                    if buer == True:
                        for m in range(len(new_ocr)):
                            if (new_ocr[m]['location'] == maybe_match_data[k][0:4]).all():
                                new_ocr[m]['matched_pairs_location'] = new_pairs[i]
                        print("匹配同方向且中心轴相同的离得近的标注（紧贴）\n", new_ocr[m])
                        matched_ocr.append(new_ocr[m])
                        matched_new_pairs[i] = 1
                        matched_new_data[m] = 1

    new_new_pairs = np.zeros((0, 4))
    new_new_pairs_region = []
    for i in range(len(new_pairs)):
        if matched_new_pairs[i] == 0:
            new_new_pairs = np.r_[new_new_pairs, [new_pairs[i]]]
            if divide_key == 1:
                new_new_pairs_region.append(new_pairs_region[i])
    new_pairs = new_new_pairs
    new_pairs_region = new_new_pairs_region
    new_new_data = []
    for i in range(len(new_ocr)):
        if matched_new_data[i] == 0:
            new_new_data.append(new_ocr[i])
    new_ocr = new_new_data
    # print('***', new_ocr)
    ###########################竖向
    mid_arr = np.zeros((8))
    matched_new_data = np.zeros((len(new_ocr)))
    matched_new_pairs = np.zeros((len(new_pairs)))
    for i in range(len(new_pairs)):
        # mid_ocr_acc = np.zeros((len(new_ocr)))  # 记录哪些是可能的
        maybe_match_data = np.zeros((0, 8))
        if (min((new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1])) / max(
                (new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1]))) < 0.75:
            if new_pairs[i][2] - new_pairs[i][0] < new_pairs[i][3] - new_pairs[i][1]:  # 竖向pairs
                for j in range(len(new_ocr)):
                    if new_ocr[j]['Absolutely'] == 'angle':
                        if new_ocr[j]['location'][2] - new_ocr[j]['location'][0] < new_ocr[j]['location'][3] - \
                                new_ocr[j]['location'][1]:  # 竖向data
                            if (new_pairs[i][3] + new_pairs[i][1]) * 0.5 < new_ocr[j]['location'][3] and (
                                    new_pairs[i][3] + new_pairs[i][1]) * 0.5 > new_ocr[j]['location'][
                                1]:  # 竖向pairs中心轴穿过竖向data
                                if (new_pairs[i][2] - new_pairs[i][0]) * 2 > min(
                                        abs(new_pairs[i][2] - new_ocr[j]['location'][0]), abs(
                                            new_pairs[i][0] - new_ocr[j]['location'][2])):  # pairs与data距离不超过pairs高度2倍
                                    mid_arr[0:4] = new_ocr[j]['location']
                                    mid_arr[4: 7] = new_ocr[j]['max_medium_min']
                                    mid_arr[7] = min(abs(new_pairs[i][3] - new_ocr[j]['location'][1]),
                                                     abs(new_pairs[i][1] - new_ocr[j]['location'][3]))
                                    if divide_key == 1:
                                        if new_ocr[j]['region'] == new_pairs_region[i]:
                                            maybe_match_data = np.r_[maybe_match_data, [mid_arr]]
                                    else:
                                        maybe_match_data = np.r_[maybe_match_data, [mid_arr]]

                                    # mid_ocr_acc[j] = len(maybe_match_data)
                maybe_match_data = maybe_match_data[np.argsort(maybe_match_data[:, 7])]  # 按距离从小到大排序
                # 当ruler中data比maybe中data大时，ruler中pairs一定比data中
                for k in range(len(maybe_match_data)):
                    buer = True
                    for l in range(len(ruler_match)):
                        if (max(ruler_match[l]['matched_pairs_location'][2] - ruler_match[l]['matched_pairs_location'][
                            0], ruler_match[l]['matched_pairs_location'][3] - ruler_match[l]['matched_pairs_location'][
                            1]) - max(
                            new_pairs[i][2] - new_pairs[i][0], new_pairs[i][3] - new_pairs[i][1])) * (
                                ruler_match[l]['max_medium_min'][0] - maybe_match_data[k][4]) < 0:
                            buer = False
                            break
                    if buer == True:
                        for m in range(len(new_ocr)):
                            if (new_ocr[m]['location'] == maybe_match_data[k][0:4]).all():
                                new_ocr[m]['matched_pairs_location'] = new_pairs[i]
                        print("匹配同方向且中心轴相同的离得近的data（紧贴）\n", new_ocr[m])
                        matched_ocr.append(new_ocr[m])
                        matched_new_pairs[i] = 1
                        matched_new_data[m] = 1
    new_new_pairs = np.zeros((0, 4))
    new_new_pairs_region = []
    for i in range(len(new_pairs)):
        if matched_new_pairs[i] == 0:
            new_new_pairs = np.r_[new_new_pairs, [new_pairs[i]]]
            if divide_key == 1:
                new_new_pairs_region.append(new_pairs_region[i])
    new_pairs = new_new_pairs
    new_pairs_region = new_new_pairs_region
    new_new_data = []
    for i in range(len(new_ocr)):
        if matched_new_data[i] == 0:
            new_new_data.append(new_ocr[i])
    new_ocr = new_new_data
    # print('***', new_ocr)
    # 4.1横向箭头横向水平(三个高范围内)且离得近（欧氏距离不超过两个pairs长）（横向箭头一般只会匹配横向数据）
    # 实际：横向箭头匹配数据（不管方向）：先在pairs的y坐标上有重叠的data填入待match，再将pairs高度扩充三倍，在y轴上与之相交的data按欧氏距离大小排列为待匹配，用ruler检验
    matched_new_pairs = np.zeros((len(new_pairs)))
    matched_new_data = np.zeros((len(new_ocr)))
    for i in range(len(new_pairs)):
        # mid_ocr_acc = np.zeros((len(new_ocr)))  # 记录哪些是可能的
        maybe_match_data_a = np.zeros((0, 7))
        maybe_match_data_b = np.zeros((0, 8))
        middle = np.zeros((8))
        if (min((new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1])) / max(
                (new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1]))) < 0.75:
            if new_pairs[i][2] - new_pairs[i][0] > new_pairs[i][3] - new_pairs[i][1]:  # 横向pairs
                for j in range(len(new_ocr)):
                    if new_ocr[j]['Absolutely'] == 'angle':
                        if not (new_pairs[i][1] > new_ocr[j]['location'][3] or new_pairs[i][3] < new_ocr[j]['location'][
                            1]):  # 两矩形在y坐标上的高有重叠
                            if min(abs(new_ocr[j]['location'][0] - new_pairs[i][2]),
                                   abs(new_ocr[j]['location'][2] - new_pairs[i][0])) < 5 * (
                                    new_pairs[i][3] - new_pairs[i][1]):  # pairs和data横向距离很近
                                mid = np.zeros(7)
                                mid[0: 4] = new_ocr[j]['location']
                                mid[4: 7] = new_ocr[j]['max_medium_min']
                                if divide_key == 1:
                                    if new_ocr[j]['region'] == new_pairs_region[i]:
                                        maybe_match_data_a = np.r_[maybe_match_data_a, [mid]]
                                else:
                                    maybe_match_data_a = np.r_[maybe_match_data_a, [mid]]

                                # mid_ocr_acc[j] = len(maybe_match_data_a)
                        if not (new_pairs[i][1] - (new_pairs[i][3] - new_pairs[i][1]) * 2 > new_ocr[j]['location'][3] or
                                new_pairs[i][
                                    3] + (new_pairs[i][3] - new_pairs[i][1]) * 2 < new_ocr[j]['location'][
                                    1]):  # 两矩形在y坐标上的高有重叠
                            if min(abs(new_ocr[j]['location'][0] - new_pairs[i][2]),
                                   abs(new_ocr[j]['location'][2] - new_pairs[i][0])) < 7 * (
                                    new_pairs[i][3] - new_pairs[i][1]):
                                if new_ocr[j]['location'] not in maybe_match_data_a[0: 4]:
                                    middle[0:4] = new_ocr[j]['location']
                                    middle[4:7] = new_ocr[j]['max_medium_min']
                                    middle[7] = min(abs(new_ocr[j]['location'][0] - new_pairs[i][2]),
                                                    abs(new_ocr[j]['location'][2] - new_pairs[i][0]))
                                    if divide_key == 1:
                                        if new_ocr[j]['region'] == new_pairs_region[i]:
                                            maybe_match_data_b = np.r_[maybe_match_data_b, [middle]]
                                    else:
                                        maybe_match_data_b = np.r_[maybe_match_data_b, [middle]]
                                    # maybe_match_data_b = np.r_[maybe_match_data_b, [middle]]
                maybe_match_data_b = maybe_match_data_b[np.argsort(maybe_match_data_b[:, 7])]  # 按距离从小到大排序
                maybe_match_data_b = maybe_match_data_b[:, 0:7]
                maybe_match_data = np.append(maybe_match_data_a, maybe_match_data_b, axis=0)
                for k in range(len(maybe_match_data)):
                    buer = True
                    for l in range(len(ruler_match)):
                        if (max(ruler_match[l]['matched_pairs_location'][2] - ruler_match[l]['matched_pairs_location'][
                            0], ruler_match[l]['matched_pairs_location'][3] - ruler_match[l]['matched_pairs_location'][
                            1]) - max(
                            new_pairs[i][2] - new_pairs[i][0], new_pairs[i][3] - new_pairs[i][1])) * (
                                ruler_match[l]['max_medium_min'][0] - maybe_match_data[k][4]) < 0:
                            buer = False
                            break
                    if buer == True:
                        for m in range(len(new_ocr)):
                            if (new_ocr[m]['location'] == maybe_match_data[k][0:4]).all():
                                new_ocr[m]['matched_pairs_location'] = new_pairs[i]
                        print(
                            "横向箭头横向水平(三个高范围内)且离得近（欧氏距离不超过两个pairs长）（横向箭头一般只会匹配横向数据）\n",
                            new_ocr[m])
                        matched_ocr.append(new_ocr[m])
                        matched_new_pairs[i] = 1
                        matched_new_data[m] = 1
    new_new_pairs = np.zeros((0, 4))
    new_new_pairs_region = []
    for i in range(len(new_pairs)):
        if matched_new_pairs[i] == 0:
            new_new_pairs = np.r_[new_new_pairs, [new_pairs[i]]]
            if divide_key == 1:
                new_new_pairs_region.append(new_pairs_region[i])
    new_pairs = new_new_pairs
    new_pairs_region = new_new_pairs_region
    new_new_data = []
    for i in range(len(new_ocr)):
        if matched_new_data[i] == 0:
            new_new_data.append(new_ocr[i])
    new_ocr = new_new_data
    # print('***', new_ocr)
    # print("new_pairs,new_data",new_pairs,new_data)
    # print("matched_pairs_data",matched_pairs_data)
    # 4.2竖向箭头竖向水平(三个高范围内)且离得近（欧氏距离不超过两个pairs长）（竖向箭头可能匹配到两个方向的数据）
    # 实际：竖向箭头匹配数据（不管方向）：先在pairs的x坐标上有重叠的data填入待match，再将pairs宽度扩充三倍，在x轴上与之相交的data按欧氏距离大小排列为待匹配，用ruler检验
    matched_new_pairs = np.zeros((len(new_pairs)))
    matched_new_data = np.zeros((len(new_ocr)))
    for i in range(len(new_pairs)):
        maybe_match_data_a = np.zeros((0, 7))
        maybe_match_data_b = np.zeros((0, 8))
        middle = np.zeros((8))
        if (min((new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1])) / max(
                (new_pairs[i][2] - new_pairs[i][0]), (new_pairs[i][3] - new_pairs[i][1]))) < 0.75:
            if new_pairs[i][2] - new_pairs[i][0] < new_pairs[i][3] - new_pairs[i][1]:  # 竖向pairs
                for j in range(len(new_ocr)):
                    if new_ocr[j]['Absolutely'] == 'angle':
                        if not (pairs[i][0] > new_ocr[j]['location'][2] or pairs[i][2] < new_ocr[j]['location'][
                            0]):  # 两矩形在x坐标上的长有重叠
                            if min(abs(new_ocr[j]['location'][1] - new_pairs[i][3]),
                                   abs(new_ocr[j]['location'][3] - new_pairs[i][1])) < 5 * (
                                    new_pairs[i][2] - new_pairs[i][0]):  # pairs和data横向距离很近
                                mid = np.zeros(7)
                                mid[0: 4] = new_ocr[j]['location']
                                mid[4: 7] = new_ocr[j]['max_medium_min']
                                if divide_key == 1:
                                    if new_ocr[j]['region'] == new_pairs_region[i]:
                                        maybe_match_data_a = np.r_[maybe_match_data_a, [mid]]
                                else:
                                    maybe_match_data_a = np.r_[maybe_match_data_a, [mid]]
                                # maybe_match_data_a = np.r_[maybe_match_data_a, [mid]]
                        if not (new_pairs[i][0] - (new_pairs[i][2] - new_pairs[i][0]) * 2 > new_ocr[j]['location'][2] or
                                new_pairs[i][
                                    2] + (new_pairs[i][2] - new_pairs[i][0]) * 2 < new_ocr[j]['location'][
                                    0]):  # 两矩形在y坐标上的高有重叠
                            if min(abs(new_ocr[j]['location'][1] - new_pairs[i][3]),
                                   abs(new_ocr[j]['location'][3] - new_pairs[i][1])) < 7 * (
                                    new_pairs[i][2] - new_pairs[i][0]):
                                if new_ocr[j]['location'] not in maybe_match_data_a[0: 4]:
                                    middle[0:4] = new_ocr[j]['location']
                                    middle[4:7] = new_ocr[j]['max_medium_min']
                                    middle[7] = min(abs(new_ocr[j]['location'][1] - new_pairs[i][3]),
                                                    abs(new_ocr[j]['location'][3] - new_pairs[i][1]))
                                    if divide_key == 1:
                                        if new_ocr[j]['region'] == new_pairs_region[i]:
                                            maybe_match_data_b = np.r_[maybe_match_data_b, [middle]]
                                    else:
                                        maybe_match_data_b = np.r_[maybe_match_data_b, [middle]]
                                    # maybe_match_data_b = np.r_[maybe_match_data_b, [middle]]
                maybe_match_data_b = maybe_match_data_b[np.argsort(maybe_match_data_b[:, 7])]  # 按距离从小到大排序
                maybe_match_data_b = maybe_match_data_b[:, 0:7]
                maybe_match_data = np.append(maybe_match_data_a, maybe_match_data_b, axis=0)
                for k in range(len(maybe_match_data)):
                    buer = True
                    for l in range(len(ruler_match)):
                        if (max(ruler_match[l]['matched_pairs_location'][2] - ruler_match[l]['matched_pairs_location'][
                            0], ruler_match[l]['matched_pairs_location'][3] - ruler_match[l]['matched_pairs_location'][
                            1]) - max(
                            new_pairs[i][2] - new_pairs[i][0], new_pairs[i][3] - new_pairs[i][1])) * (
                                ruler_match[l]['max_medium_min'][0] - maybe_match_data[k][4]) < 0:
                            buer = False
                            break
                    if buer == True:
                        for m in range(len(new_ocr)):
                            if (new_ocr[m]['location'] == maybe_match_data[k][0:4]).all():
                                new_ocr[m]['matched_pairs_location'] = new_pairs[i]
                        print(
                            "竖向箭头竖向水平(三个高范围内)且离得近（欧氏距离不超过两个pairs长）\n",
                            new_ocr[m])
                        matched_ocr.append(new_ocr[m])
                        matched_new_pairs[i] = 1
                        matched_new_data[m] = 1
    new_new_pairs = np.zeros((0, 4))
    new_new_pairs_region = []
    for i in range(len(new_pairs)):
        if matched_new_pairs[i] == 0:
            new_new_pairs = np.r_[new_new_pairs, [new_pairs[i]]]
            if divide_key == 1:
                new_new_pairs_region.append(new_pairs_region[i])

    new_pairs = new_new_pairs
    new_pairs_region = new_new_pairs_region
    new_new_data = []
    for i in range(len(new_ocr)):
        if matched_new_data[i] == 0:
            new_new_data.append(new_ocr[i])
    new_ocr = new_new_data
    # print('***', new_ocr)
    # 5.剩余标尺线按欧式距离匹配
    # from math import sqrt # 确保前面导入了 sqrt
    print("---进入欧氏距离匹配循环---")  # 调试打印

    # 移除原本无用的 x 和 right_matched_pairs 逻辑
    # 只要还有剩下的 pair 和 data，且还能匹配到新的，就继续
    while len(new_ocr) > 0 and len(new_pairs) > 0:

        matched_pairs = np.full((len(new_pairs)), -1)  # 初始化为 -1
        matched_pairs_len = np.zeros((len(new_pairs)))

        # 5.1.将所有pairs按照最近data匹配，记录匹配data序号和距离
        has_new_match = False  # 标记本轮是否有新匹配

        for i in range(len(new_pairs)):
            min_lenth = 99999
            min_no = -1
            for j in range(len(new_ocr)):
                # 计算中心点距离
                pair_cx = (new_pairs[i][2] + new_pairs[i][0]) * 0.5
                pair_cy = (new_pairs[i][3] + new_pairs[i][1]) * 0.5
                ocr_cx = (new_ocr[j]['location'][2] + new_ocr[j]['location'][0]) * 0.5
                ocr_cy = (new_ocr[j]['location'][3] + new_ocr[j]['location'][1]) * 0.5

                lenth = sqrt((pair_cx - ocr_cx) ** 2 + (pair_cy - ocr_cy) ** 2)

                if lenth < min_lenth:
                    min_lenth = lenth
                    min_no = j

            if min_no != -1 and min_lenth < max_length:
                matched_pairs[i] = min_no
                matched_pairs_len[i] = min_lenth

        # 5.2.冲突解决：将相同匹配的pairs中距离大的项清零
        # (这部分逻辑保留你的原意，但在Python中如果matched_pairs[i]为-1需要跳过)
        for i in range(len(matched_pairs)):
            if matched_pairs[i] != -1:
                for j in range(len(matched_pairs)):
                    if i != j and matched_pairs[j] != -1:
                        if matched_pairs[i] == matched_pairs[j]:
                            if matched_pairs_len[i] > matched_pairs_len[j]:
                                matched_pairs[i] = -1
                                matched_pairs_len[i] = 0
                            else:
                                matched_pairs[j] = -1
                                matched_pairs_len[j] = 0

        # === 【关键修改点 1】: 将匹配成功的放入结果集 ===
        for i in range(len(new_pairs)):
            if matched_pairs[i] != -1:
                idx = int(matched_pairs[i])
                matched_item = new_ocr[idx]
                matched_item['matched_pairs_location'] = new_pairs[i]
                matched_ocr.append(matched_item)  # 加入结果列表
                print(f"欧氏距离匹配成功: {matched_item['ocr_strings']}")  # 调试打印
                has_new_match = True

        # === 【关键修改点 2】: 准备下一轮的剩余数据 ===
        no_matched_pairs = np.zeros((0, 4))
        no_matched_data = []

        # 筛选未匹配的 Pairs
        for i in range(len(new_pairs)):
            if matched_pairs[i] == -1:
                no_matched_pairs = np.r_[no_matched_pairs, [new_pairs[i]]]

        # 筛选未匹配的 Data (注意：这里要排除掉本轮 matched_pairs 中所有的索引)
        matched_indices = set(matched_pairs)  # 这一轮被匹配掉的 data 索引集合
        for i in range(len(new_ocr)):
            if i not in matched_indices:
                no_matched_data.append(new_ocr[i])

        # === 【关键修改点 3】: 更新循环变量，通过判断是否有进展来防止死循环 ===
        if not has_new_match:
            print("本轮无新匹配，跳出循环")
            break

        # 更新变量以进入下一轮
        new_pairs = no_matched_pairs
        new_ocr = no_matched_data

        # 如果更新后列表为空，自然也会在下一次 while 判断时退出

    # 去重 (保留你原来的逻辑)
    result = []
    seen = set()
    for item in matched_ocr:
        # 处理 ndarray不可哈希的问题
        loc_key = item['location'].tobytes() if isinstance(item['location'], np.ndarray) else str(item['location'])
        if loc_key not in seen:
            result.append(item)
            seen.add(loc_key)
    return result

def show_matched_pairs_data(img_path, pairs_data):
    wh_key1 = True
    while wh_key1:
        auto_key = input("是否展示匹配好的pairs_data:y/n:")

        if auto_key == 'y' or auto_key == 'Y':
            auto_bool = False
            wh_key1 = False
        elif auto_key == 'n' or auto_key == 'N':
            auto_bool = True
            wh_key1 = False
        else:
            print("未输入正确，请重新输入：y/n:")
            wh_key1 = True
    if auto_bool == False:
        # import numpy as np
        # import cv2 as cv

        for i in range(len(pairs_data)):
            with open(img_path, 'rb') as f:
                np_arr = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # 以彩图读取
                # img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)  # 以灰度图读取
            # 矩形左上角和右上角的坐标，绘制一个绿色矩形
            if pairs_data[i]['matched_pairs_location'] != []:
                ptLeftTop = (
                    int(pairs_data[i]['matched_pairs_location'][0]), int(pairs_data[i]['matched_pairs_location'][1]))
                ptRightBottom = (
                    int(pairs_data[i]['matched_pairs_location'][2]), int(pairs_data[i]['matched_pairs_location'][3]))
                point_color = (0, 255, 0)  # BGR
                thickness = 2
                lineType = 4
                cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)

            # 绘制一个红色矩形
            ptLeftTop = (int(pairs_data[i]['location'][0]), int(pairs_data[i]['location'][1]))
            ptRightBottom = (int(pairs_data[i]['location'][2]), int(pairs_data[i]['location'][3]))
            point_color = (0, 0, 255)  # BGR
            thickness = 2
            lineType = 8
            cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)

            cv2.namedWindow("pairs(green)_data(red)", 0)
            cv2.imshow('pairs(green)_data(red)', img)
            cv2.waitKey(0)  # 显示 10000 ms 即 10s 后消失
            cv2.destroyAllWindows()

def BGA_side_filter(side_ocr_data):
    # 1.side中有用的尺寸数字应该小于side_max_limate
    new_side_ocr_data = []
    side_max_limate = 15
    for i in range(len(side_ocr_data)):
        try:
            if (side_ocr_data[i]['max_medium_min'] < side_max_limate).all():
                new_side_ocr_data.append(side_ocr_data[i])
        except:
            a = 0
    return new_side_ocr_data

def find_serial_number_letter_QFP(serial_numbers, serial_letters, top_ocr_data, bottom_ocr_data):
    '''
    YOLO检测的PIN序号区域中存在PIN序号，寻找PIN序号方法：横向PIN序号区域在左右两端寻找PIN序号，竖向PIN序号区域在上下两端寻找PIN序号
    serial_numbers:np(,4)[x1,y1,x2,y2]
    serial_letters:np(,4)[x1,y1,x2,y2]
    bottom_dbnet_data:np(,4)[x1,y1,x2,y2]
    横向YOLO取左右
    竖向YOLO取上下
    '''
    # 提取serial中的数字序号
    ratio = 0.2
    top_dbnet_data_account = np.zeros((len(top_ocr_data)))  # 1 = 是serial的文本，需要剔除
    new_top_ocr_data = []
    top_serial = []
    bottom_dbnet_data_account = np.zeros((len(bottom_ocr_data)))  # 1 = 是serial的文本，需要剔除
    new_bottom_ocr_data = []
    bottom_serial = []
    for i in range(len(serial_numbers)):
        if abs(serial_numbers[i][0] - serial_numbers[i][2]) > abs(serial_numbers[i][1] - serial_numbers[i][3]):
            fangxiang = 'x'
        else:
            fangxiang = 'y'

        middle = []

        for k in range(len(top_ocr_data)):

            if not (top_ocr_data[k]['location'][0] > serial_numbers[i][2] or top_ocr_data[k]['location'][2] <
                    serial_numbers[i][
                        0]):  # 两矩形在x坐标上的长有重叠
                if not (top_ocr_data[k]['location'][1] > serial_numbers[i][3] or top_ocr_data[k]['location'][3] <
                        serial_numbers[i][
                            1]):  # 两矩形在y坐标上的高有重叠
                    # 测试用
                    # print("重叠的", top_ocr_data[k])

                    l = abs(top_ocr_data[k]['location'][2] - top_ocr_data[k]['location'][0]) + abs(serial_numbers[i][2] - \
                                                                               serial_numbers[i][0]) - (
                                max(serial_numbers[i][2], top_ocr_data[k]['location'][2]) - min(serial_numbers[i][0],
                                                                                      top_ocr_data[k]['location'][0]))

                    w = top_ocr_data[k]['location'][3] - top_ocr_data[k]['location'][1] + serial_numbers[i][3] - \
                        serial_numbers[i][1] - (
                                max(serial_numbers[i][3], top_ocr_data[k]['location'][3]) - min(serial_numbers[i][1],
                                                                                      top_ocr_data[k]['location'][1]))
                    if l * w / (top_ocr_data[k]['location'][2] - top_ocr_data[k]['location'][0]) * (
                            top_ocr_data[k]['location'][3] - top_ocr_data[k]['location'][1]) > ratio or l * w / (
                            serial_numbers[i][2] - serial_numbers[i][0]) * (
                            serial_numbers[i][3] - serial_numbers[i][1]) > ratio:

                        if fangxiang == 'x':
                            if serial_numbers[i][0] < (top_ocr_data[k]['location'][0] + top_ocr_data[k]['location'][2]) * 0.5 < serial_numbers[i][0] + abs(serial_numbers[i][2] - serial_numbers[i][0]) * 0.25 or serial_numbers[i][0]+ abs(serial_numbers[i][2] - serial_numbers[i][0]) * (1 - 0.25) < (top_ocr_data[k]['location'][0] + top_ocr_data[k]['location'][2]) * 0.5 < serial_numbers[i][2]:

                                # 确保标注是整数且无公差
                                if top_ocr_data[k]['max_medium_min'][0] == top_ocr_data[k]['max_medium_min'][1] and \
                                        top_ocr_data[k]['max_medium_min'][1] == top_ocr_data[k]['max_medium_min'][2]:
                                    if int(top_ocr_data[k]['max_medium_min'][0]) == top_ocr_data[k]['max_medium_min'][
                                        0]:

                                        top_dbnet_data_account[k] = 1
                                        middle.append({'s_n_location': top_ocr_data[k]['location'],
                                                       'fangxiang': [fangxiang],
                                                       'max_medium_min': top_ocr_data[k]['max_medium_min']})
                        if fangxiang == 'y':
                            if serial_numbers[i][1] < (top_ocr_data[k]['location'][1] + top_ocr_data[k]['location'][3]) * 0.5 < serial_numbers[i][1] + abs(serial_numbers[i][3] - serial_numbers[i][1]) * 0.25 or serial_numbers[i][1]+ abs(serial_numbers[i][3] - serial_numbers[i][1])* (1 - 0.25) < (top_ocr_data[k]['location'][1] + top_ocr_data[k]['location'][3]) * 0.5 < serial_numbers[i][3]:

                                # 确保标注是整数且无公差
                                if top_ocr_data[k]['max_medium_min'][0] == top_ocr_data[k]['max_medium_min'][1] and top_ocr_data[k]['max_medium_min'][1] == top_ocr_data[k]['max_medium_min'][2]:
                                    if int(top_ocr_data[k]['max_medium_min'][0]) == top_ocr_data[k]['max_medium_min'][0]:

                                        top_dbnet_data_account[k] = 1
                                        middle.append({'s_n_location': top_ocr_data[k]['location'],
                                                           'fangxiang': [fangxiang], 'max_medium_min': top_ocr_data[k]['max_medium_min']})
        if len(middle) != 0:
            top_serial.append(middle)
    # 测试用
    # print("top_ocr_data:\n", *top_ocr_data, sep='\n')
    # print("top_dbnet_data_account", top_dbnet_data_account)

    for i in range(len(top_dbnet_data_account)):
        if top_dbnet_data_account[i] != 1:
            new_top_ocr_data.append(top_ocr_data[i])

    for i in range(len(serial_letters)):
        if abs(serial_letters[i][0] - serial_letters[i][2]) > abs(serial_letters[i][1] - serial_letters[i][3]):
            fangxiang = 'x'
        else:
            fangxiang = 'y'

        middle = []

        for k in range(len(bottom_ocr_data)):

            if not (bottom_ocr_data[k]['location'][0] > serial_letters[i][2] or bottom_ocr_data[k]['location'][2] <
                    serial_letters[i][
                        0]):  # 两矩形在x坐标上的长有重叠
                if not (bottom_ocr_data[k]['location'][1] > serial_letters[i][3] or bottom_ocr_data[k]['location'][3] <
                        serial_letters[i][
                            1]):  # 两矩形在y坐标上的高有重叠

                    l = abs(bottom_ocr_data[k]['location'][2] - bottom_ocr_data[k]['location'][0]) + abs(
                        serial_letters[i][2] - \
                        serial_letters[i][0]) - (
                                max(serial_letters[i][2], bottom_ocr_data[k]['location'][2]) - min(serial_letters[i][0],
                                                                                                   bottom_ocr_data[k][
                                                                                                       'location'][
                                                                                                       0]))

                    w = bottom_ocr_data[k]['location'][3] - bottom_ocr_data[k]['location'][1] + serial_letters[i][3] - \
                        serial_letters[i][1] - (
                                max(serial_letters[i][3], bottom_ocr_data[k]['location'][3]) - min(serial_letters[i][1],
                                                                                                   bottom_ocr_data[k][
                                                                                                       'location'][
                                                                                                       1]))
                    if l * w / (bottom_ocr_data[k]['location'][2] - bottom_ocr_data[k]['location'][0]) * (
                            bottom_ocr_data[k]['location'][3] - bottom_ocr_data[k]['location'][1]) > ratio or l * w / (
                            serial_letters[i][2] - serial_letters[i][0]) * (
                            serial_letters[i][3] - serial_letters[i][1]) > ratio:
                        if fangxiang == 'x':
                            if serial_letters[i][0] < (bottom_ocr_data[k]['location'][0] + bottom_ocr_data[k]['location'][2]) * 0.5 < serial_letters[i][0] + abs(serial_letters[i][2] - serial_letters[i][0]) * 0.25 or serial_letters[i][0]+ abs(serial_letters[i][2] - serial_letters[i][0]) * (1 - 0.25) < (bottom_ocr_data[k]['location'][0] + bottom_ocr_data[k]['location'][2]) * 0.5 < serial_letters[i][2]:

                                if bottom_ocr_data[k]['max_medium_min'][0] == bottom_ocr_data[k]['max_medium_min'][
                                    1] and \
                                        bottom_ocr_data[k]['max_medium_min'][1] == bottom_ocr_data[k]['max_medium_min'][
                                    2]:
                                    if int(bottom_ocr_data[k]['max_medium_min'][0]) == \
                                            bottom_ocr_data[k]['max_medium_min'][0]:
                                        bottom_dbnet_data_account[k] = 1
                                        middle.append({'s_n_location': bottom_ocr_data[k]['location'],
                                                       'fangxiang': [fangxiang],
                                                       'max_medium_min': bottom_ocr_data[k]['max_medium_min']})
                        if fangxiang == 'y':
                            if serial_letters[i][1] < (bottom_ocr_data[k]['location'][1] + bottom_ocr_data[k]['location'][3]) * 0.5 < serial_letters[i][1] + abs(serial_letters[i][3] - serial_letters[i][1]) * 0.25 or serial_letters[i][1]+ abs(serial_letters[i][3] - serial_letters[i][1])* (1 - 0.25) < (bottom_ocr_data[k]['location'][1] + bottom_ocr_data[k]['location'][3]) * 0.5 < serial_letters[i][3]:

                                # 确保标注是整数且无公差
                                if bottom_ocr_data[k]['max_medium_min'][0] == bottom_ocr_data[k]['max_medium_min'][
                                    1] and \
                                        bottom_ocr_data[k]['max_medium_min'][1] == bottom_ocr_data[k]['max_medium_min'][
                                    2]:
                                    if int(bottom_ocr_data[k]['max_medium_min'][0]) == \
                                            bottom_ocr_data[k]['max_medium_min'][0]:
                                        bottom_dbnet_data_account[k] = 1
                                        middle.append({'s_n_location': bottom_ocr_data[k]['location'],
                                                       'fangxiang': [fangxiang],
                                                       'max_medium_min': bottom_ocr_data[k]['max_medium_min']})



        if len(middle) != 0:
            bottom_serial.append(middle)
    for i in range(len(bottom_dbnet_data_account)):
        if bottom_dbnet_data_account[i] != 1:
            new_bottom_ocr_data.append(bottom_ocr_data[i])
    # 测试用代码
    # print("top视图找到的PIN序号", *top_serial, sep='\n')
    # print("bottom视图找到的PIN序号", *bottom_serial, sep='\n')
    # print("top_ocr_data:\n", *new_top_ocr_data, sep='\n')
    # print("bottom_ocr_data:\n", *new_bottom_ocr_data, sep='\n')


    return top_serial, bottom_serial, new_top_ocr_data, new_bottom_ocr_data

# 标注的原始文本'ocr_strings': ocr_data[i],
# 标注的关键信息'key_info': [],
def convert_Dic(dbnet_data, ocr_data):
    '''
    将ocr识别出来的字符串与位置信息结合为字典类型
    dbnet_data:np(, 4)[x1,y1,x2,y2]
    ocr_data:list['string','str']
    '''
    # === [新增] 兼容性补丁：如果 ocr_data 已经是字典列表，说明是定向 OCR 的结果，直接返回 ===
    if ocr_data and isinstance(ocr_data, list) and len(ocr_data) > 0 and isinstance(ocr_data[0], dict):
        return ocr_data

    # === [新增] 容错处理：如果 ocr_data 为空（且不是上面的字典情况），返回空列表，防止索引越界 ===
    if not ocr_data or len(ocr_data) == 0:
        return []

    new_ocr_data = []
    min_len = min(len(dbnet_data), len(ocr_data))
    for i in range(min_len):
        dic = {'location': dbnet_data[i],
               'ocr_strings': ocr_data[i],
               'key_info': [],
               'matched_pairs_location': [],
               'matched_pairs_outside_or_inside': [],
               'matched_pairs_yinXian': [],
               'Absolutely': [],
               'max_medium_min': []}
        new_ocr_data.append(dic)
    return new_ocr_data

def filter_ocr_data_0(ocr_data):
    '''
    筛除‘’
    '''
    new_ocr_data = []
    for i in range(len(ocr_data)):
        if not (ocr_data[i]['ocr_strings'] == ''):
            new_ocr_data.append(ocr_data[i])
    return new_ocr_data

def filter_ocr_data__1(ocr_data):
    '''
    删除标注关键信息检测识别为空
    '''
    new_ocr_data = []
    for i in range(len(ocr_data)):
        if not (ocr_data[i]['key_info'] == []):
            new_ocr_data.append(ocr_data[i])
    return new_ocr_data

def filter_ocr_data__2(ocr_data):
    '''
    清理key_info中不含数字的数据
    '''
    new_ocr_data = []
    for i in range(len(ocr_data)):
        right_key = 0
        for j in range(len(ocr_data[i]['key_info'])):
            if right_key == 1:
                break
            for k in range(len(ocr_data[i]['key_info'][j])):
                try:
                    a = float(ocr_data[i]['key_info'][j][k])
                    right_key = 1
                    break
                except:
                    pass
        if right_key == 1:
            new_ocr_data.append(ocr_data[i])
    ocr_data = new_ocr_data

    return ocr_data

def filter_ocr_data_1(list):
    '''
    字符串中删除‘数字‘ + 'X’ (字符串中不存在'='时),'Absolutely'为'mb_pin_diameter'
    '''
    for i in range(len(list)):
        str_data1 = re.findall("=", list[i]['ocr_strings'])
        if len(str_data1) == 0:
            str_data2 = re.findall("(\d+\.?\d*)[Xx]", list[i]['ocr_strings'])
            str_data = re.sub("(\d+\.?\d*)[Xx]", '', list[i]['ocr_strings'])
            if len(str_data2) != 0:
                list[i]['Absolutely'] = 'mb_pin_diameter'
            list[i]['ocr_strings'] = str_data
    return list

def filter_ocr_data_2(list):
    '''
    字符串中删除'PIN1'和’PIN‘
    '''
    for i in range(len(list)):
        str_data = re.sub("[Pp][Ii][Nn]1*", '', list[i]['ocr_strings'])
        list[i]['ocr_strings'] = str_data
    return list

def filter_ocr_data_3(list):
    '''
    字符串中删除'A1'
    '''
    for i in range(len(list)):
        str_data = re.sub("[Aa]1", '', list[i]['ocr_strings'])
        list[i]['ocr_strings'] = str_data
    return list

def filter_ocr_data_11(list):
    '''
    删除'note' + '整数数字'
    '''
    for i in range(len(list)):
        str_data = re.sub("[Nn][Oo][Tt][Ee][23456789]", '', list[i]['ocr_strings'])
        list[i]['ocr_strings'] = str_data
    return list

def filter_ocr_data_4(list):
    '''
    ','改为'.'
    '''
    for i in range(len(list)):
        str_data = re.sub("[,，]", '.', list[i]['ocr_strings'])
        list[i]['ocr_strings'] = str_data
    return list

def filter_ocr_data_5(list):
    '''
    单个字符时，字符串中删除'A''B''C''D'
    '''
    for i in range(len(list)):
        if len(list[i]['ocr_strings']) == 1:
            str_data = re.sub("[AaBbCcDd]", '', list[i]['ocr_strings'])
            list[i]['ocr_strings'] = str_data
    return list

def filter_ocr_data_6(list):
    '''
    提取’数字‘’+‘’-‘’=‘’Φ‘’±‘’max‘’nom‘’min‘'x''°'
    如果检测到"±",仅保留"±"以及符号的前一位数字和后一位数字
    当出现'°'时，标记absolute为'angle'
    '''
    for i in range(len(list)):
        str_data_angle = re.findall("°", list[i]['ocr_strings'])
        if len(str_data_angle) != 0:
            list[i]['Absolutely'] = 'angle'
        str_data = re.findall("\d+(?:\.\d+)?|=|\+|-|Φ|±|[Mm][Aa][Xx]|[Nn][Oo][Mm]|[Mm][Ii][Nn]|[Xx*]",
                              list[i]['ocr_strings'])
        str_data = [x.strip() for x in str_data if x.strip() != '']  # 将字符串列表中空项删除
        list[i]['key_info'] = str_data

        for j in range(len(str_data)):
            if j > 0:
                if str_data[j] == '±':
                    try:
                        a = float(str_data[j - 1])
                        b = float(str_data[j + 1])
                        new_str_data = []
                        new_str_data.append(str_data[j - 1])
                        new_str_data.append(str_data[j])
                        new_str_data.append(str_data[j + 1])
                        for k in range(len(str_data)):
                            if str_data[k] == 'Φ':
                                new_str_data.append(str_data[k])
                        list[i]['key_info'] = new_str_data
                    except:
                        pass

    return list

def filter_ocr_data_7(list):
    '''
    删除key_info中的'.'
    当key_info中只有'x'时删除
    '''
    for i in range(len(list)):
        new_key_info = []
        for k in range(len(list[i]['key_info'])):
            if list[i]['key_info'][k] != '.':
                new_key_info.append(list[i]['key_info'][k])
        list[i]['key_info'] = new_key_info

        if len(list[i]['key_info']) == 1:
            if list[i]['key_info'][0] == 'X' or list[i]['key_info'][0] == 'x':
                list[i]['key_info'] = []
    return list

def filter_ocr_data_9(ocr_data):
    '''
    key_info中的数字如果以'0'开头而第二个字符却没有小数点，则添加小数点
    '''
    for i in range(len(ocr_data)):
        for j in range(len(ocr_data[i]['key_info'])):
            for k in range(len(ocr_data[i]['key_info'][j])):
                try:
                    a = float(ocr_data[i]['key_info'][j][k])
                    if ocr_data[i]['key_info'][j][k][0] == '0':
                        if ocr_data[i]['key_info'][j][k][1] != '.':
                            b = '.'
                            str_list = list(ocr_data[i]['key_info'][j][k])
                            str_list.insert(1, b)
                            a_b = ''.join(str_list)
                            ocr_data[i]['key_info'][j][k] = a_b
                except:
                    pass
    return ocr_data

def filter_ocr_data_10(ocr_data):
    '''
    # 删除key_info中的'0','0.','00'
    '''

    for i in range(len(ocr_data)):
        if ocr_data[i]['Absolutely'] != 'angle':
            new_key_info = []
            for j in range(len(ocr_data[i]['key_info'])):
                string = ocr_data[i]['key_info'][j]
                if not (string == '0' or string == '0.' or string == '00'):
                    new_key_info.append(string)
            ocr_data[i]['key_info'] = new_key_info
    return ocr_data

def filter_ocr_data_12(ocr_data):
    '''
    删除公差特别大的标注(不删除角度标注)
    '''
    new_ocr_data = []
    for i in range(len(ocr_data)):
        if abs(ocr_data[i]['max_medium_min'][0] - ocr_data[i]['max_medium_min'][1]) <= 1 and abs(
                ocr_data[i]['max_medium_min'][1] - ocr_data[i]['max_medium_min'][2]) <= 1 or ocr_data[i]['Absolutely'] == 'angle':
            new_ocr_data.append(ocr_data[i])
        else:
            print('删除公差特别大的标注:', ocr_data[i]['max_medium_min'])
    return new_ocr_data

def get_serial(top_serial_numbers_data, bottom_serial_numbers_data):
    '''
    尝试从yolo检测的serial标签中按照规则寻找nx和ny
    top_serial_numbers_data[[[], [], []], [], []]
    '''
    # 1.7
    nx = []
    ny = []

    for i in range(len(top_serial_numbers_data)):
        if len(top_serial_numbers_data[i]) > 1:
            middle = {'n': 0, 'from': [0, 0], 'count': 0}
            for j in range(len(top_serial_numbers_data[i])):
                for k in range(len(top_serial_numbers_data[i])):
                    if abs(top_serial_numbers_data[i][j]['max_medium_min'][0] - top_serial_numbers_data[i][k]['max_medium_min'][0]) > middle['n']:
                        middle['n'] = abs(top_serial_numbers_data[i][j]['max_medium_min'][0] - top_serial_numbers_data[i][k]['max_medium_min'][0])
                        middle['from'] = [j, k]
                        if top_serial_numbers_data[i][0]['fangxiang'] == 'x':
                            nx.append(middle)
                        else:
                            ny.append(middle)
    for i in range(len(bottom_serial_numbers_data)):
        if len(bottom_serial_numbers_data[i]) > 1:
            middle = {'n': 0, 'from': [0, 0], 'count': 0}
            for j in range(len(bottom_serial_numbers_data[i])):
                for k in range(len(bottom_serial_numbers_data[i])):
                    if abs(bottom_serial_numbers_data[i][j]['max_medium_min'][0] -
                           bottom_serial_numbers_data[i][k]['max_medium_min'][0]) > middle['n']:
                        middle['n'] = abs(bottom_serial_numbers_data[i][j]['max_medium_min'][0] -
                                          bottom_serial_numbers_data[i][k]['max_medium_min'][0])
                        middle['from'] = [j, k]
                        if bottom_serial_numbers_data[i][0]['fangxiang'] == 'x':
                            nx.append(middle)
                        else:
                            ny.append(middle)

    max_nx = 0
    max_count = 0
    for i in range(len(nx)):
        for j in range(len(nx)):
            if nx[i]['n'] == nx[j]['n']:
                nx[i]['count'] += 1
                nx[j]['count'] += 1
                if max_count < nx[j]['count']:
                    max_count = nx[j]['count']
                    max_nx = nx[j]['n']
    max_ny = 0
    max_count = 0
    for i in range(len(ny)):
        for j in range(len(ny)):
            if ny[i]['n'] == ny[j]['n']:
                ny[i]['count'] += 1
                ny[j]['count'] += 1
                if max_count < ny[j]['count']:
                    max_count = ny[j]['count']
                    max_ny = ny[j]['n']
    if max_nx == 0:
        max_nx = max_ny
    if max_ny == 0:
        max_ny = max_nx
    if max_nx != 0:
        max_nx += 1
    if max_ny != 0:
        max_ny += 1
    print('nx', max_nx, 'ny', max_ny)
    return max_nx, max_ny

def get_BGA_parameter_list(top_ocr_data, bottom_ocr_data, side_ocr_data, detailed_ocr_data, body_x, body_y):
    '''
    D/E 10~35
    D1/E1
    D2/E2
    A
    A1
    e
    Φ
    b
    θ
    L
    c:引脚厚度

    '''
    QFP_parameter_list = []
    dic = {'parameter_name': [], 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    dic_D = {'parameter_name': 'D', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    dic_E = {'parameter_name': 'E', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    D_max = 35
    D_min = 8.75
    E_max = D_max
    E_min = D_min

    # ===== 12.22新增：Φ（pin_diameter）=====
    dic_phi = {'parameter_name': 'Φ', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    phi_max = 0.8
    phi_min = 0.4

    # ===== 12.22新增：把 Φ 放到末尾（不影响原有索引）=====

    # index = len(QFP_parameter_list)-1 = 19
    phi_idx = len(QFP_parameter_list) - 1
    # ==============================================
    # ================================

    dic_D1 = {'parameter_name': 'D1', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    dic_E1 = {'parameter_name': 'E1', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    D1_max = 30
    D1_min = 6.9
    E1_max = D1_max
    E1_min = D1_min

    dic_L = {'parameter_name': 'L', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    L_max = 0.75
    L_min = 0.45


    dic_GAGE_PLANE = {'parameter_name': 'GAGE_PLANE', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    GAGE_PLANE_max = 0.25
    GAGE_PLANE_min = 0.25

    dic_c = {'parameter_name': 'c', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    c_max = 0.20
    c_min = 0.09

    dic_θ = {'parameter_name': 'θ', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    θ_max = 10
    θ_min = 0

    dic_θ1 = {'parameter_name': 'θ1', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    θ1_max = 14
    θ1_min = 0

    dic_θ2 = {'parameter_name': 'θ2', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    θ2_max = 16
    θ2_min = 11

    dic_θ3 = {'parameter_name': 'θ3', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    θ3_max = 16
    θ3_min = 11

    # dic_D2 = {'parameter_name': 'D2', 'maybe_data': [], 'possible': []}
    # dic_E2 = {'parameter_name': 'E2', 'maybe_data': [], 'possible': []}

    dic_A = {'parameter_name': 'A', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    A_max = 4.5
    A_min = 1.0
    dic_A1 = {'parameter_name': 'A1', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    A1_max = 0.4
    A1_min = 0
    dic_e = {'parameter_name': 'e', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    e_max = 1.3
    e_min = 0.30
    dic_b = {'parameter_name': 'b', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    b_max = 0.83
    b_min = 0.13
    dic_D2 = {'parameter_name': 'D2', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    dic_E2 = {'parameter_name': 'E2', 'maybe_data': [], 'maybe_data_num': 0, 'possible': [], 'OK': 0}
    D2_max = 7.2
    D2_min = 3.15
    E2_max = 7.2
    E2_min = 3.15
    QFP_parameter_list.append(dic_D)
    QFP_parameter_list.append(dic_E)
    QFP_parameter_list.append(dic_D1)
    QFP_parameter_list.append(dic_E1)
    # QFP_parameter_list.append(dic_D2)
    # QFP_parameter_list.append(dic_E2)
    QFP_parameter_list.append(dic_A)
    QFP_parameter_list.append(dic_A1)
    QFP_parameter_list.append(dic_e)
    QFP_parameter_list.append(dic_b)
    QFP_parameter_list.append(dic_D2)
    QFP_parameter_list.append(dic_E2)
    QFP_parameter_list.append(dic_L)
    QFP_parameter_list.append(dic_GAGE_PLANE)
    QFP_parameter_list.append(dic_c)
    QFP_parameter_list.append(dic_θ)
    QFP_parameter_list.append(dic_θ1)
    QFP_parameter_list.append(dic_θ2)
    QFP_parameter_list.append(dic_θ3)
    # 12.22新增：Φ（pin_diameter）=====

    QFP_parameter_list.append(dic_phi)




    for i in range(len(top_ocr_data)):
        if D_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= D_max:
            QFP_parameter_list[0]['maybe_data'].append(top_ocr_data[i])
            QFP_parameter_list[0]['maybe_data_num'] += 1
        if E_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= E_max:
            QFP_parameter_list[1]['maybe_data'].append(top_ocr_data[i])
            QFP_parameter_list[1]['maybe_data_num'] += 1
        if len(body_x) > 0:
            QFP_parameter_list[2]['maybe_data'] = body_x
        if D1_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= D1_max:
                QFP_parameter_list[2]['maybe_data'].append(top_ocr_data[i])
                QFP_parameter_list[2]['maybe_data_num'] += 1
        if len(body_y) > 0:
            QFP_parameter_list[3]['maybe_data'] = body_y
        if E1_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= E1_max:
                QFP_parameter_list[3]['maybe_data'].append(top_ocr_data[i])
                QFP_parameter_list[3]['maybe_data_num'] += 1
        if e_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= e_max:
            QFP_parameter_list[6]['maybe_data'].append(top_ocr_data[i])
            QFP_parameter_list[6]['maybe_data_num'] += 1
        if b_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= b_max:
            QFP_parameter_list[7]['maybe_data'].append(top_ocr_data[i])
            QFP_parameter_list[7]['maybe_data_num'] += 1
        if D2_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= D2_max:
            QFP_parameter_list[8]['maybe_data'].append(top_ocr_data[i])
            QFP_parameter_list[8]['maybe_data_num'] += 1
        if E2_min <= top_ocr_data[i]['max_medium_min'][2] and top_ocr_data[i]['max_medium_min'][0] <= E2_max:
            QFP_parameter_list[9]['maybe_data'].append(top_ocr_data[i])
            QFP_parameter_list[9]['maybe_data_num'] += 1
    for i in range(len(bottom_ocr_data)):
        if D_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= D_max  :
            QFP_parameter_list[0]['maybe_data'].append(bottom_ocr_data[i])
            QFP_parameter_list[0]['maybe_data_num'] += 1
        if E_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= E_max :
            QFP_parameter_list[1]['maybe_data'].append(bottom_ocr_data[i])
            QFP_parameter_list[1]['maybe_data_num'] += 1
        if D1_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= D1_max:
           if bottom_ocr_data[i].get('Absolutely') == 'Body_x':
                QFP_parameter_list[2]['maybe_data'].append(bottom_ocr_data[i])
                QFP_parameter_list[2]['maybe_data_num'] += 1
            # ===== 新增：Φ 只从 bottom_ocr_data 来，且 Absolutely == 'pin_diameter' =====
            # ===== 新增：Φ只从bottom来，且Absolutely == pin_diameter =====
        if bottom_ocr_data[i].get('Absolutely') == 'pin_diameter' or bottom_ocr_data[i].get(
                'Absolutely') == 'mb_pin_diameter' or bottom_ocr_data[i].get('Absolutely') == 'pin_diameter+':
            if phi_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][
                0] <= phi_max:
                QFP_parameter_list[phi_idx]['maybe_data'].append(bottom_ocr_data[i])  # 用-1永远指向最后一个（Φ）
                QFP_parameter_list[phi_idx]['maybe_data_num'] += 1
        # =====================================================================

        if E1_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= E1_max:
            if bottom_ocr_data[i].get('Absolutely') == 'Body_y':
                QFP_parameter_list[3]['maybe_data'].append(bottom_ocr_data[i])
                QFP_parameter_list[3]['maybe_data_num'] += 1
        #0111新增：将ocr中最大的数据填入D1与E1的候选列表中
        # Max_index_D = max_maybe_data_medium(QFP_parameter_list[0])  # 0为D，1为E，2为D1，3为E1
        # Max_index_E = max_maybe_data_medium(QFP_parameter_list[1])
        # QFP_parameter_list[2]['maybe_data'].append(bottom_ocr_data[Max_index_D])
        # QFP_parameter_list[2]['maybe_data_num'] += 1
        # QFP_parameter_list[3]['maybe_data'].append(bottom_ocr_data[Max_index_E])
        # QFP_parameter_list[3]['maybe_data_num'] += 1

        if bottom_ocr_data[i].get('Absolutely') != 'pin_diameter' and bottom_ocr_data[i].get(
                'Absolutely') != 'mb_pin_diameter' and bottom_ocr_data[i].get('Absolutely') != 'pin_diameter+':
            if e_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= e_max:
                QFP_parameter_list[6]['maybe_data'].append(bottom_ocr_data[i])
                QFP_parameter_list[6]['maybe_data_num'] += 1
        if b_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= b_max:
            QFP_parameter_list[7]['maybe_data'].append(bottom_ocr_data[i])
            QFP_parameter_list[7]['maybe_data_num'] += 1
        if D2_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= D2_max:
            QFP_parameter_list[8]['maybe_data'].append(bottom_ocr_data[i])
            QFP_parameter_list[8]['maybe_data_num'] += 1
        if E2_min <= bottom_ocr_data[i]['max_medium_min'][2] and bottom_ocr_data[i]['max_medium_min'][0] <= E2_max:
            QFP_parameter_list[9]['maybe_data'].append(bottom_ocr_data[i])
            QFP_parameter_list[9]['maybe_data_num'] += 1

    for i in range(len(side_ocr_data)):
        if D_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= D_max:
            QFP_parameter_list[0]['maybe_data'].append(side_ocr_data[i])
            QFP_parameter_list[0]['maybe_data_num'] += 1
        if E_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= E_max:
            QFP_parameter_list[1]['maybe_data'].append(side_ocr_data[i])
            QFP_parameter_list[1]['maybe_data_num'] += 1
        if D1_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= D1_max:
                QFP_parameter_list[2]['maybe_data'].append(side_ocr_data[i])
                QFP_parameter_list[2]['maybe_data_num'] += 1
        if E1_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= E1_max:
                QFP_parameter_list[3]['maybe_data'].append(side_ocr_data[i])
                QFP_parameter_list[3]['maybe_data_num'] += 1
        if A_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= A_max:
            QFP_parameter_list[4]['maybe_data'].append(side_ocr_data[i])
            QFP_parameter_list[4]['maybe_data_num'] += 1

        if A1_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= A1_max:
            QFP_parameter_list[5]['maybe_data'].append(side_ocr_data[i])
            QFP_parameter_list[5]['maybe_data_num'] += 1
        if e_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= e_max:
            QFP_parameter_list[6]['maybe_data'].append(side_ocr_data[i])
            QFP_parameter_list[6]['maybe_data_num'] += 1
        if b_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= b_max:
            QFP_parameter_list[7]['maybe_data'].append(side_ocr_data[i])
            QFP_parameter_list[7]['maybe_data_num'] += 1
        if side_ocr_data[i]['Absolutely'] == 'angle':
            if θ2_min <= side_ocr_data[i]['max_medium_min'][2] and side_ocr_data[i]['max_medium_min'][0] <= θ2_max:
                QFP_parameter_list[15]['maybe_data'].append(side_ocr_data[i])
                QFP_parameter_list[15]['maybe_data_num'] += 1
    for i in range(len(detailed_ocr_data)):
        if A_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= A_max:
            QFP_parameter_list[4]['maybe_data'].append(detailed_ocr_data[i])
            QFP_parameter_list[4]['maybe_data_num'] += 1
        if L_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= L_max:
            QFP_parameter_list[10]['maybe_data'].append(detailed_ocr_data[i])
            QFP_parameter_list[10]['maybe_data_num'] += 1
        if GAGE_PLANE_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= GAGE_PLANE_max:
            QFP_parameter_list[11]['maybe_data'].append(detailed_ocr_data[i])
            QFP_parameter_list[11]['maybe_data_num'] += 1
        if c_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= c_max:
            QFP_parameter_list[12]['maybe_data'].append(detailed_ocr_data[i])
            QFP_parameter_list[12]['maybe_data_num'] += 1

        if detailed_ocr_data[i]['Absolutely'] == 'angle':
            if θ_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= θ_max:
                QFP_parameter_list[13]['maybe_data'].append(detailed_ocr_data[i])
                QFP_parameter_list[13]['maybe_data_num'] += 1
            if θ1_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= θ1_max:
                QFP_parameter_list[14]['maybe_data'].append(detailed_ocr_data[i])
                QFP_parameter_list[14]['maybe_data_num'] += 1

            if θ2_min < detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= θ2_max:
                QFP_parameter_list[15]['maybe_data'].append(detailed_ocr_data[i])
                QFP_parameter_list[15]['maybe_data_num'] += 1
            if θ3_min <= detailed_ocr_data[i]['max_medium_min'][2] and detailed_ocr_data[i]['max_medium_min'][0] <= θ3_max:
                QFP_parameter_list[16]['maybe_data'].append(detailed_ocr_data[i])
                QFP_parameter_list[16]['maybe_data_num'] += 1

    for i in range(len(QFP_parameter_list)):
        maybe = QFP_parameter_list[i].get("maybe_data", [])
        seen = set()
        new_maybe = []

        for j in range(len(maybe)):
            mmm = maybe[j].get("max_medium_min")
            if mmm is None:
                continue  # 没有这个字段就跳过（你也可以选择保留）

            key = tuple(mmm)  # [15.1, 15, 14.9] -> (15.1, 15, 14.9)
            if key in seen:
                continue  # 重复，丢弃
            seen.add(key)
            new_maybe.append(maybe[j])

        QFP_parameter_list[i]["maybe_data"] = new_maybe
        QFP_parameter_list[i]["maybe_data_num"] = len(new_maybe)

    for i in range(len(QFP_parameter_list)):
        print("筛选完毕，以下是第一次填参结果：")
        print("***/", QFP_parameter_list[i]['parameter_name'],"/***")

        for j in range(len(QFP_parameter_list[i]['maybe_data'])):
            print(QFP_parameter_list[i]['maybe_data'][j]['max_medium_min'])

    for i in range(len(QFP_parameter_list)):
        print(QFP_parameter_list[i]['maybe_data_num'])

    return QFP_parameter_list

def resort_parameter_list_2(QFP_parameter_list):
    '''
    1.查看哪些参数已经只有一个标注了，将这个参数确定标注
    2.将其他参数的可能标注中去除这个确定标注
    3.返回1
    '''
    print("继续筛选参数，确定唯一值：")
    key = True
    while key:
        key = False
        for i in range(len(QFP_parameter_list)):

            if len(QFP_parameter_list[i]['maybe_data']) == 1:
                key_dic = QFP_parameter_list[i]['maybe_data'][0]
                QFP_parameter_list[i]['OK'] = 1

                for j in range(len(QFP_parameter_list)):
                    if j != i:
                        if len(QFP_parameter_list[j]['maybe_data']) > 0:
                            if any(np.array_equal(key_dic, item) for item in QFP_parameter_list[j]['maybe_data']):
                                # 执行相应的操作

                                QFP_parameter_list[j]['maybe_data'] = [d for d in QFP_parameter_list[j]['maybe_data'] if not np.array_equal(key_dic, d)]
                                key = True

    for i in range(len(QFP_parameter_list)):
        print("***/", QFP_parameter_list[i]['parameter_name'], "/***")
        print(QFP_parameter_list[i])
        for j in range(len(QFP_parameter_list[i]['maybe_data'])):
            print(QFP_parameter_list[i]['maybe_data'][j]['max_medium_min'])
    for i in range(len(QFP_parameter_list)):
        QFP_parameter_list[i]['maybe_data_num'] = len(QFP_parameter_list[i]['maybe_data'])

    for i in range(len(QFP_parameter_list)):
        print(QFP_parameter_list[i]['maybe_data_num'])
    return QFP_parameter_list

def empty_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
    except FileNotFoundError:
        print(f"文件夹 {folder_path} 不存在！")

def correct_serial_letters_data(serial_letters_data):
    for i in range(len(serial_letters_data)):
        j = -1
        for every_letter in serial_letters_data[i][4]:
            j += 1
            if every_letter == '8':
                every_letter = 'B'
                strings = list(serial_letters_data[i][4])
                strings[j] = 'B'
                serial_letters_data[i][4] = ''.join(strings)
    return serial_letters_data

def find_pin_num_pin_1(serial_numbers_data, serial_letters_data, serial_numbers, serial_letters):
    '''
    serial_numbers_data:np.(,4)['x1','y1','x2','y2','str']
    serial_letters_data:np.(,4)['x1','y1','x2','y2','str']
    serial_numbers:np.(,4)[x1,y1,x2,y2)
    serial_letters:np.(,4)[x1,y1,x2,y2)
    '''
    # 默认输出
    pin_num_x_serial = 0
    pin_num_y_serial = 0
    pin_num_serial_number = 0
    pin_num_serial_letter = 0
    pin_1_location = np.array([-1, -1])
    # pin_1_location = [X, Y],X = 0:横向用数字标记序号，纵向用字母标记序号；X= 1，横向用字母标记序号，纵向用数字标记序号
    # pin_1_location = [X, Y],Y = 0 = 左上角,1= 右上角，2 = 右下角，3 = 左下角
    if len(serial_numbers_data) > 0 or len(serial_numbers_data) > 0:

        # ocr识别serial_number,serial_letter
        # img_path = 'data/bottom.jpg'
        # serial_numbers_data = ocr_en_cn_onnx(img_path, serial_numbers_data)
        # print('serial_numbers_data', serial_numbers_data)
        # serial_letters_data = ocr_en_cn_onnx(img_path, serial_letters_data)
        # print('serial_letters_data', serial_letters_data)
        # 根据经验修改ocr识别的错误
        serial_letters_data = correct_serial_letters_data(serial_letters_data)
        print('修正之后的serial_letters_data', serial_letters_data)
        # 根据serial_number最大值找行列数
        serial_number = np.zeros((0))
        new_serial_numbers_data = np.array([['0', '0', '0', '0', '0']])
        for i in range(len(serial_numbers_data)):
            try:
                serial_number = np.append(serial_number, int(serial_numbers_data[i][4]))
                new_serial_numbers_data = np.r_[new_serial_numbers_data, [serial_numbers_data[i]]]
            except:
                print("在用数字标识的pin行列序号中ocr识别到非数字信息，删除")

        serial_numbers_data = new_serial_numbers_data
        print('修正之后的serial_numbers_data', serial_numbers_data)
        serial_number = -(np.sort(-serial_number))  # 从大到小排列
        pin_num_serial_number = 0
        for i in range(len(serial_number)):
            if len(serial_number) > 1 and i + 1 < len(serial_number):
                if serial_number[i] - serial_number[i + 1] < 3:
                    pin_num_serial_number = serial_number[i]
                    break

        # 根据serial_letter最大值找行列数
        letter_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'T', 'U', 'V', 'W',
                       'Y']
        letter_list_a = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 't', 'u', 'v', 'w',
                         'y']
        serial_letter = np.zeros((0))
        # 将字母序列转为数字序列
        for i in range(len(serial_letters_data)):
            letter_number = 0
            no = 0
            letters = serial_letters_data[i][4]
            letters = letters[::-1]  # 倒序
            for every_letter in letters:
                no += 1
                for j in range(len(letter_list)):
                    if letter_list[j] == every_letter or letter_list_a[j] == every_letter:
                        letter_number += 20 ** (no - 1) * (j + 1)
            serial_letter = np.append(serial_letter, letter_number)
            serial_letters_data[i][4] = str(letter_number)
        print("serial_letters_data", serial_letters_data)
        serial_letter = -(np.sort(-serial_letter))  # 从大到小排列
        pin_num_serial_letter = 0
        for i in range(len(serial_letter)):
            if len(serial_letter) > 1 and i + 1 < len(serial_letter):
                if serial_letter[i] - serial_letter[i + 1] < 3:
                    pin_num_serial_letter = serial_letter[i]
                    break
        print('pin_num_serial_number, pin_num_serial_letter', pin_num_serial_number, pin_num_serial_letter)
    if pin_num_serial_number != 0:
        if len(serial_numbers) > 0:
            if abs(serial_numbers[0][0] - serial_numbers[0][2]) > abs(serial_numbers[0][1] - serial_numbers[0][3]):
                pin_num_x_serial = pin_num_serial_number
            else:
                pin_num_y_serial = pin_num_serial_number
    if pin_num_serial_letter != 0:
        if len(serial_letters) > 0:
            if abs(serial_letters[0][0] - serial_letters[0][2]) > abs(serial_letters[0][1] - serial_letters[0][3]):
                pin_num_x_serial = pin_num_serial_letter
            else:
                pin_num_y_serial = pin_num_serial_letter
    print("pin_num_x_serial, pin_num_y_serial", pin_num_x_serial, pin_num_y_serial)
    # pin_1_location = [X, Y],X = 0:横向用数字标记序号，纵向用字母标记序号；X= 1，横向用字母标记序号，纵向用数字标记序号
    # pin_1_location = [X, Y],Y = 0 = 左上角,1= 右上角，2 = 右下角，3 = 左下角
    # pin1定位
    if len(serial_numbers_data) > 0 or len(serial_numbers_data) > 0:
        if len(serial_numbers) > 0:
            if abs(serial_numbers[0][0] - serial_numbers[0][2]) > abs(serial_numbers[0][1] - serial_numbers[0][3]):
                pin_1_location[0] = 0
            else:
                pin_1_location[0] = 1
        if len(serial_letters) > 0:
            if abs(serial_letters[0][0] - serial_letters[0][2]) > abs(serial_letters[0][1] - serial_letters[0][3]):
                pin_1_location[0] = 1
            else:
                pin_1_location[0] = 0

        heng_begain = -1
        shu_begain = -1
        serial_numbers_data = serial_numbers_data.astype(np.float32)
        serial_numbers_data = serial_numbers_data.astype(np.int32)
        # 删除0
        new_serial_numbers_data = np.zeros((0, 5))
        for i in range(len(serial_numbers_data)):
            if serial_numbers_data[i][4] != 0:
                new_serial_numbers_data = np.r_[new_serial_numbers_data, [serial_numbers_data[i]]]
        serial_numbers_data = new_serial_numbers_data

        if len(serial_numbers_data) > 0:
            if abs(serial_numbers[0][0] - serial_numbers[0][2]) > abs(serial_numbers[0][1] - serial_numbers[0][3]):
                if len(serial_numbers_data) >= 2:
                    if len(serial_numbers_data[0]) == 5:
                        serial_numbers_data = serial_numbers_data[np.argsort(serial_numbers_data[:, 4])]  # 按序号从小到大排序
                        if serial_numbers_data[0, 0] < serial_numbers_data[(len(serial_numbers_data) - 1), 0]:
                            heng_begain = 0
                        else:
                            heng_begain = 1
            if abs(serial_numbers[0][0] - serial_numbers[0][2]) < abs(serial_numbers[0][1] - serial_numbers[0][3]):
                if len(serial_numbers_data) >= 2:
                    if len(serial_numbers_data[0]) == 5:
                        serial_numbers_data = serial_numbers_data[np.argsort(serial_numbers_data[:, 4])]  # 按序号从小到大排序
                        if serial_numbers_data[0, 1] < serial_numbers_data[(len(serial_numbers_data) - 1), 1]:
                            shu_begain = 0
                        else:
                            shu_begain = 1
        if len(serial_letters_data) > 0:
            serial_letters_data = serial_letters_data.astype(np.float32)
            serial_letters_data = serial_letters_data.astype(np.int32)
            # 删除0
            new_serial_letters_data = np.zeros((0, 5))
            for i in range(len(serial_letters_data)):
                if serial_letters_data[i][4] != 0:
                    new_serial_letters_data = np.r_[new_serial_letters_data, [serial_letters_data[i]]]
            serial_letters_data = new_serial_letters_data
            if abs(serial_letters[0][0] - serial_letters[0][2]) > abs(serial_letters[0][1] - serial_letters[0][3]):
                if len(serial_letters_data) >= 2:
                    if len(serial_letters_data[0]) == 5:
                        serial_letters_data = serial_letters_data[np.argsort(serial_letters_data[:, 4])]  # 按序号从小到大排序
                        if serial_letters_data[0, 0] < serial_letters_data[(len(serial_letters_data) - 1), 0]:
                            heng_begain = 0
                        else:
                            heng_begain = 1
            if abs(serial_letters[0][0] - serial_letters[0][2]) < abs(serial_letters[0][1] - serial_letters[0][3]):
                if len(serial_letters_data) >= 2:
                    if len(serial_letters_data[0]) == 5:
                        serial_letters_data = serial_letters_data[np.argsort(serial_letters_data[:, 4])]  # 按序号从小到大排序
                        if serial_letters_data[0, 1] < serial_letters_data[(len(serial_letters_data) - 1), 1]:
                            shu_begain = 0
                        else:
                            shu_begain = 1
        if heng_begain == 0 and shu_begain == 0:
            pin_1_location[1] = 0
        if heng_begain == 1 and shu_begain == 0:
            pin_1_location[1] = 1
        if heng_begain == 1 and shu_begain == 1:
            pin_1_location[1] = 2
        if heng_begain == 0 and shu_begain == 1:
            pin_1_location[1] = 3
        if heng_begain == 0 and shu_begain == -1:
            pin_1_location[1] = 0
        if heng_begain == 1 and shu_begain == -1:
            pin_1_location[1] = 1
        if heng_begain == -1 and shu_begain == 0:
            pin_1_location[1] = 0
        if heng_begain == -1 and shu_begain == 1:
            pin_1_location[1] = 3

    return pin_num_x_serial, pin_num_y_serial, pin_1_location

def get_body(yolox_pairs_top, top_yolox_pairs_length, yolox_pairs_bottom, bottom_yolox_pairs_length, top_border, bottom_border, top_ocr_data, bottom_ocr_data):
    '''
    # yolox_pairs_top,np.二维数组（，11）[pairs_x1_y1_x2_y2,标注x1_y1_x2_y2，max,medium,min]
    # top_yolox_pairs_length,np.二维数组（，13）[pairs_x1_y1_x2_y2,引线1_x1_y1_x2_y2,引线2_x1_y1_x2_y2,两引线距离]
    '''
    print("---开始用引线方法寻找body---")
    # print("top_yolox_pairs_length, bottom_yolox_pairs_length\n", top_yolox_pairs_length, bottom_yolox_pairs_length)
    # (1)标注添加引线
    min_body = 1.28  # 设置最小的body长
    new_top_yolox_pairs_length = np.zeros((0, 16))
    middle = np.zeros(16)
    for i in range(len(top_yolox_pairs_length)):
        for j in range(len(yolox_pairs_top)):
            if (yolox_pairs_top[j, 0:3] == top_yolox_pairs_length[i, 0:3]).all():
                middle[0:13] = top_yolox_pairs_length[i]
                middle[13:16] = yolox_pairs_top[j, -3:]
                new_top_yolox_pairs_length = np.r_[new_top_yolox_pairs_length, [middle]]
    print('new_bottom_yolox_pairs_length:', new_top_yolox_pairs_length)

    new_bottom_yolox_pairs_length = np.zeros((0, 16))
    middle = np.zeros(16)
    for i in range(len(bottom_yolox_pairs_length)):
        for j in range(len(yolox_pairs_bottom)):
            if (yolox_pairs_bottom[j, 0:3] == bottom_yolox_pairs_length[i, 0:3]).all():
                middle[0:13] = bottom_yolox_pairs_length[i]
                middle[13:16] = yolox_pairs_bottom[j, -3:]
                new_bottom_yolox_pairs_length = np.r_[new_bottom_yolox_pairs_length, [middle]]
    # (2)opencv寻找body
    img_path = f"{DATA}/top.jpg"
    if not os.path.exists(img_path):
        top_body = []
    else:
        top_body = output_body(img_path, name='top')
    # 测试用
    print("top_body:", top_body)
    print("top_border:", top_border)
    if len(top_body) == 0:
        top_body = top_border  # top_body:np(1,4)[x1,y1,x2,y2]
        if len(top_body) == 0:
            top_body = np.zeros((0, 3))
    # 测试用
    print("top_body:", top_body)

    img_path = f"{DATA}/bottom.jpg"
    if not os.path.exists(img_path):
        bottom_body = []
    else:
        bottom_body = output_body(img_path, name='bottom')

    if len(bottom_body) == 0:
        bottom_body = bottom_border  # bottom_body:np(1,4)[x1,y1,x2,y2]
        if len(bottom_body) == 0:
            bottom_body = np.zeros((0, 3))
    # (3)通过引线位置和body的位置对比找到长和宽
    zer = np.zeros(3)
    gao = np.zeros(3)
    kuan = np.zeros(3)
    bottom_gao = np.zeros(3)
    bottom_kuan = np.zeros(3)
    ratio = 0.9
    for i in range(len(new_top_yolox_pairs_length)):

        ruler_1 = 0
        ruler_2 = 0
        if abs(new_top_yolox_pairs_length[i][0] - new_top_yolox_pairs_length[i][2]) > abs(
                new_top_yolox_pairs_length[i][1] - new_top_yolox_pairs_length[i][3]):
            ruler_1 = new_top_yolox_pairs_length[i][4]
            ruler_2 = new_top_yolox_pairs_length[i][8]
            ruler_3 = top_body[0][0]
            ruler_4 = top_body[0][2]
            if new_top_yolox_pairs_length[i][12] != 0:
                # 测试用
                print(abs(ruler_1 - ruler_3) / new_top_yolox_pairs_length[i][12], abs(ruler_2 - ruler_4) /
                      new_top_yolox_pairs_length[i][12], abs(ruler_1 - ruler_4) / new_top_yolox_pairs_length[i][12],
                      abs(ruler_1 - ruler_4) / new_top_yolox_pairs_length[i][12])
                if (abs(ruler_1 - ruler_3) / new_top_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_4) /
                    new_top_yolox_pairs_length[i][12] < ratio) or (
                        abs(ruler_1 - ruler_4) / new_top_yolox_pairs_length[i][12] < ratio and abs(
                    ruler_2 - ruler_3) /
                        new_top_yolox_pairs_length[i][12] < ratio):

                    if (new_top_yolox_pairs_length[i][-3:] > min_body).all():
                        kuan = new_top_yolox_pairs_length[i][-3:]
                        print("引线下引到实体top_body宽，找到top_body_x参数", kuan)
        if abs(new_top_yolox_pairs_length[i][0] - new_top_yolox_pairs_length[i][2]) < abs(
                new_top_yolox_pairs_length[i][1] - new_top_yolox_pairs_length[i][3]):
            ruler_1 = new_top_yolox_pairs_length[i][5]
            ruler_2 = new_top_yolox_pairs_length[i][9]
            ruler_3 = top_body[0][1]
            ruler_4 = top_body[0][3]
            if new_top_yolox_pairs_length[i][12] != 0:
                if (abs(ruler_1 - ruler_3) / new_top_yolox_pairs_length[i][12] < ratio and abs(ruler_2 - ruler_4) /
                    new_top_yolox_pairs_length[i][12] < ratio) or (
                        abs(ruler_1 - ruler_4) / new_top_yolox_pairs_length[i][12] < ratio and abs(
                    ruler_2 - ruler_3) /
                        new_top_yolox_pairs_length[i][12] < ratio):
                    if (new_top_yolox_pairs_length[i][-3:] > min_body).all():
                        gao = new_top_yolox_pairs_length[i][-3:]
                        print("引线下引到实体top_body高，找到top_body_y参数", gao)
    ratio = 0.03
    for i in range(len(new_bottom_yolox_pairs_length)):
        ruler_1 = 0
        ruler_2 = 0
        if abs(new_bottom_yolox_pairs_length[i][0] - new_bottom_yolox_pairs_length[i][2]) > abs(
                new_bottom_yolox_pairs_length[i][1] - new_bottom_yolox_pairs_length[i][3]):
            ruler_1 = new_bottom_yolox_pairs_length[i][4]
            ruler_2 = new_bottom_yolox_pairs_length[i][8]
            ruler_3 = bottom_body[0][0]
            ruler_4 = bottom_body[0][2]
            if new_bottom_yolox_pairs_length[i][12] != 0:
                if (abs(ruler_1 - ruler_3) / new_bottom_yolox_pairs_length[i][12] < ratio and abs(
                        ruler_2 - ruler_4) /
                    new_bottom_yolox_pairs_length[i][12] < ratio) or (
                        abs(ruler_1 - ruler_4) / new_bottom_yolox_pairs_length[i][12] < ratio and abs(
                    ruler_2 - ruler_3) /
                        new_bottom_yolox_pairs_length[i][12] < ratio):
                    if (new_bottom_yolox_pairs_length[i][-3:] > min_body).all():
                        bottom_kuan = new_bottom_yolox_pairs_length[i][-3:]
                        print("引线下引到实体bottom_body宽，找到bottom_body_x参数", bottom_kuan)
        if abs(new_bottom_yolox_pairs_length[i][0] - new_bottom_yolox_pairs_length[i][2]) < abs(
                new_bottom_yolox_pairs_length[i][1] - new_bottom_yolox_pairs_length[i][3]):
            ruler_1 = new_bottom_yolox_pairs_length[i][5]
            ruler_2 = new_bottom_yolox_pairs_length[i][9]
            ruler_3 = bottom_body[0][1]
            ruler_4 = bottom_body[0][3]
            if new_bottom_yolox_pairs_length[i][12] != 0:
                if (abs(ruler_1 - ruler_3) / new_bottom_yolox_pairs_length[i][12] < ratio and abs(
                        ruler_2 - ruler_4) /
                    new_bottom_yolox_pairs_length[i][12] < ratio) or (
                        abs(ruler_1 - ruler_4) / new_bottom_yolox_pairs_length[i][12] < ratio and abs(
                    ruler_2 - ruler_3) /
                        new_bottom_yolox_pairs_length[i][12] < ratio):
                    if (new_bottom_yolox_pairs_length[i][-3:] > min_body).all():
                        bottom_gao = new_bottom_yolox_pairs_length[i][-3:]
                        print("引线下引到实体bottom_body高，找到bottom_body_y参数", bottom_gao)

    # 2.1.计算top中body的长宽
    # 为没有匹配到引线的标尺线添加引线间距为标尺线的外框长度
    for i in range(len(new_top_yolox_pairs_length)):
        if new_top_yolox_pairs_length[i][12] == 0:
            new_top_yolox_pairs_length[i][12] = max((new_top_yolox_pairs_length[i][2] - new_top_yolox_pairs_length[i][0]), (new_top_yolox_pairs_length[i][3] - new_top_yolox_pairs_length[i][1]))

    for i in range(len(new_bottom_yolox_pairs_length)):
        if new_bottom_yolox_pairs_length[i][12] == 0:
            new_bottom_yolox_pairs_length[i][12] = max((new_bottom_yolox_pairs_length[i][2] - new_bottom_yolox_pairs_length[i][0]), (new_bottom_yolox_pairs_length[i][3] - new_bottom_yolox_pairs_length[i][1]))

    if len(top_body) > 0:
        top_body_heng = top_body[0][2] - top_body[0][0]
        top_body_shu = top_body[0][3] - top_body[0][1]

        # 2.在误差允许范围内，pairs对应长宽则认为是长宽
        ratio = 0.1
        for i in range(len(new_top_yolox_pairs_length)):
            if abs(new_top_yolox_pairs_length[i][12] - top_body_heng) / top_body_heng < ratio and (
                    kuan == zer).all():
                if (new_top_yolox_pairs_length[i][-3:] > min_body).all():
                    kuan = new_top_yolox_pairs_length[i, -3:]
                    print("引线之间距离近似kuan，找到kuan参数", kuan)
            if abs(new_top_yolox_pairs_length[i][12] - top_body_shu) / top_body_shu < ratio and (gao == zer).all():
                if (new_top_yolox_pairs_length[i][-3:] > min_body).all():
                    gao = new_top_yolox_pairs_length[i, -3:]
                    print("引线之间距离近似gao，找到gao参数", gao)
    # 3.计算bottom中body的长和宽

    if len(bottom_body) > 0:
        # print("bottom_body", bottom_body)
        bottom_body_heng = bottom_body[0][2] - bottom_body[0][0]
        bottom_body_shu = bottom_body[0][3] - bottom_body[0][1]

        # 4.在误差允许范围内，pairs对应长宽则认为是长宽
        ratio = 0.1
        for i in range(len(new_bottom_yolox_pairs_length)):
            if abs(new_bottom_yolox_pairs_length[i][12] - bottom_body_heng) / bottom_body_heng < ratio and (
                    bottom_kuan == zer).all():
                if (new_bottom_yolox_pairs_length[i][-3:] > min_body).all():
                    bottom_kuan = new_bottom_yolox_pairs_length[i, -3:]
                    print("引线之间距离近似bottom_kuan，找到bottom_kuan参数", bottom_kuan)
            if abs(new_bottom_yolox_pairs_length[i][12] - bottom_body_shu) / bottom_body_shu < ratio and (
                    bottom_gao == zer).all():
                if (new_bottom_yolox_pairs_length[i][-3:] > min_body).all():
                    bottom_gao = new_bottom_yolox_pairs_length[i, -3:]
                    print("引线之间距离近似bottom_gao，找到bottom_gao参数", bottom_gao)
    # top和bottom互补
    if (kuan == zer).all() and (bottom_kuan != zer).any():
        kuan = bottom_kuan
    if (bottom_kuan == zer).all() and (kuan != zer).any():
        bottom_kuan = kuan
    if (gao == zer).all() and (bottom_gao != zer).any():
        gao = bottom_gao
    if (bottom_gao == zer).all() and (gao != zer).any():
        bottom_gao = gao
    # top的x和y互补
    if (kuan == zer).all() and (gao != zer).any():
        kuan = gao
    if (gao == zer).all() and (kuan != zer).any():
        gao = kuan
    # bottom的x和y互补
    if (bottom_kuan == zer).all() and (bottom_gao != zer).any():
        bottom_kuan = bottom_gao
    if (bottom_gao == zer).all() and (bottom_kuan != zer).any():
        bottom_gao = bottom_kuan
    # bottom和top中找最合适的
    body_x = np.zeros(3)
    body_y = np.zeros(3)
    if (kuan != zer).any() and (bottom_kuan != zer).any():
        if (kuan > bottom_kuan).all():
            body_x = kuan
        if (kuan < bottom_kuan).all():
            body_x = bottom_kuan
    if (gao != zer).any() and (bottom_gao != zer).any():
        if (gao > bottom_gao).all():
            body_x = gao
        if (gao < bottom_gao).all():
            body_x = bottom_gao
    if (body_x == zer).all():
        body_x = kuan
    if (body_y == zer).all():
        body_y = gao
    # 将body_x和body_y还原为字典格式
    ocr_data = top_ocr_data + bottom_ocr_data
    if (body_x != zer).any():
        for i in range(len(ocr_data)):
            if ocr_data[i]['max_medium_min'][0] == body_x[0] and ocr_data[i]['max_medium_min'][1] == body_x[1] and ocr_data[i]['max_medium_min'][2] == body_x[2]:
                a = []
                a.append(ocr_data[i])
                body_x = a
    else:
        body_x = []
    if (body_y != zer).any():
        for i in range(len(ocr_data)):
            if ocr_data[i]['max_medium_min'][0] == body_y[0] and ocr_data[i]['max_medium_min'][1] == body_y[1] and ocr_data[i]['max_medium_min'][2] == body_y[2]:
                a = []
                a.append(ocr_data[i])
                body_y = a
    else:
        body_y = []
    print("---结束用引线方法寻找body---")
    print("body_x, body_y", body_x, body_y)
    return body_x, body_y


# 1218 新添加：收集各 pdf 的 yolox/dbnet 结果与最终参数匹配结果，并以 txt 保存
import sys
import re
from pathlib import Path
from contextlib import contextmanager


@contextmanager
def redirect_print_to_file(txt_path: Path, mode: str = "w"):
    """
    临时将 print 输出重定向到 txt 文件
    mode:
      - "w" 覆盖写
      - "a" 追加写
    """
    txt_path = Path(txt_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    old_stdout = sys.stdout
    with open(txt_path, mode, encoding="utf-8") as f:
        sys.stdout = f
        try:
            yield
        finally:
            sys.stdout = old_stdout


from typing import Union
from pathlib import Path

def get_first_pdf(pdf_dir: Union[str, Path] = "Result/temp") -> Path:
    """
    获取 pdf_dir 下的第一个 pdf（按文件名排序）
    """
    pdf_dir = Path(pdf_dir)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"{pdf_dir} 下没有 pdf 文件")
    return pdf_files[0]


def get_pdf_log_dir(pdf_path: Path, txt_root_dir: str | Path = "Result/log") -> Path:
    """
    为每个 pdf 创建独立日志子目录：
      Result/log/<pdf_stem>/
    """
    txt_root_dir = Path(txt_root_dir)
    log_dir = txt_root_dir / pdf_path.stem
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def build_numbered_txt_path(log_dir: Path, base_name: str, view_idx: int | None = None) -> Path:
    """
    在同一 PDF 的日志目录下，为同类日志生成带编号的 txt：
      base_name(1).txt, base_name(2).txt ...
    - 如果 view_idx 传入：直接生成 base_name(view_idx).txt
    - 如果不传：自动扫描已有文件，取下一个可用编号
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if view_idx is not None:
        return log_dir / f"{base_name}({view_idx}).txt"

    # 自动递增编号：找 base_name(n).txt 的最大 n
    pattern = re.compile(rf"^{re.escape(base_name)}\((\d+)\)\.txt$", re.IGNORECASE)
    max_n = 0
    for p in log_dir.glob(f"{base_name}(*).txt"):
        m = pattern.match(p.name)
        if m:
            max_n = max(max_n, int(m.group(1)))

    next_n = max_n + 1
    return log_dir / f"{base_name}({next_n}).txt"


def run_and_save_resort_log(
        QFP_parameter_list,
        temp_dir: str | Path = "Result/temp",
        txt_root_dir: str | Path = "BGA_pdf",
        base_name: str = "resort_log",
        view_idx: int | None = None,
        mode: str = "w"
) -> Path:
    """
    执行 resort_parameter_list_2，并将所有 print 输出保存为：
      Result/log/<pdf_stem>/resort_log(1).txt  / resort_log(2).txt ...
    """
    pdf_path = get_first_pdf(temp_dir)
    log_dir = get_pdf_log_dir(pdf_path, txt_root_dir)
    txt_path = build_numbered_txt_path(log_dir, base_name, view_idx=view_idx)

    with redirect_print_to_file(txt_path, mode=mode):
        print(f"===== PDF: {pdf_path.name} =====")
        print(f"===== VIEW: {view_idx if view_idx is not None else 'AUTO'} =====")
        print("===== RESORT PARAMETER LOG =====")
        resort_parameter_list_2(QFP_parameter_list)

    return txt_path


def run_and_save_resort_log2(
        top_ocr_data,
        bottom_ocr_data,
        side_ocr_data,
        detailed_ocr_data,
        temp_dir: str | Path = "Result/temp",
        txt_root_dir: str | Path = "BGA_pdf",
        base_name: str = "ocr_log",
        view_idx: int | None = None,
        mode: str = "w"
) -> Path:
    """
    将 OCR 相关数据保存为：
      Result/log/<pdf_stem>/ocr_log(1).txt / ocr_log(2).txt ...
    """
    pdf_path = get_first_pdf(temp_dir)
    log_dir = get_pdf_log_dir(pdf_path, txt_root_dir)
    txt_path = build_numbered_txt_path(log_dir, base_name, view_idx=view_idx)

    with redirect_print_to_file(txt_path, mode=mode):
        print(f"===== PDF: {pdf_path.name} =====")
        print(f"===== VIEW: {view_idx if view_idx is not None else 'AUTO'} =====")

        print("\n===== TOP OCR DATA =====")
        print(top_ocr_data)

        print("\n===== BOTTOM OCR DATA =====")
        print(bottom_ocr_data)

        print("\n===== SIDE OCR DATA =====")
        print(side_ocr_data)

        print("\n===== DETAILED OCR DATA =====")
        print(detailed_ocr_data)

    return txt_path

# 0111新增 用框的坐标计算坐标框的长度,并把横纵长度最长的bottom_ocr_data数据标记为长宽
def Body_x_y_length(bottom_ocr_data):
    if not bottom_ocr_data:
        return bottom_ocr_data
    max_width = -1
    max_height = -1
    width_index = None
    height_index = None
    for i, item in enumerate(bottom_ocr_data):
        location = item.get('matched_pairs_location', [])
        if len(location) < 4:
            continue
        width = abs(location[2] - location[0])
        height = abs(location[3] - location[1])
        if width > max_width:
            max_width = width
            width_index = i
        if height > max_height:
            max_height = height
            height_index = i
    if width_index is not None and not bottom_ocr_data[width_index]['Absolutely']:
        bottom_ocr_data[width_index]['Absolutely'] = 'Body_x'
        print("该元素拥有横向最大长度箭头对：",bottom_ocr_data[width_index])
    else :
        print("未找到横向最大长度箭头对")
    if height_index is not None and not bottom_ocr_data[height_index]['Absolutely']:
        bottom_ocr_data[height_index]['Absolutely'] = 'Body_y'
        print("该元素拥有纵向最大长度箭头对：",bottom_ocr_data[height_index])
    else :
        print("未找到纵向最大长度箭头对")
    return bottom_ocr_data

# 0111新增,找出候选参数maybe_data中medium值最大的元素
def max_maybe_data_medium(param):
        """
        输入: param = QFP_parameter_list[i]
        输出: j (使 param['maybe_data'][j]['max_medium_min'][1] 最大的索引)
             如果 maybe_data 为空或找不到有效值，返回 -1
        """
        best_j = -1
        best_medium = float("-inf")

        maybe = param.get("maybe_data", [])
        for j in range(len(maybe)):
            mmm = maybe[j].get("max_medium_min", None)
            if mmm is None or len(mmm) < 2:
                continue
            medium = float(mmm[1])
            if medium > best_medium:
                best_medium = medium
                best_j = j

        return best_j
