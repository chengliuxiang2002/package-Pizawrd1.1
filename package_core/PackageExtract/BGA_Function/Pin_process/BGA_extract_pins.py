# -*- coding: utf-8 -*-
"""
图像文本检测与坐标计算脚本（支持双簇合并+单行单列DETR逻辑+None值DETR强制触发）
核心功能：
1. OCR识别与文本框过滤
2. 水平/垂直聚类（返回前2个最优簇，支持合并距离≤30的簇）
3. 单行/单列簇（用户定义场景）调用DETR，X/Y取DETR与OCR最大值
4. 当X或Y为None时强制触发DETR，最终结果仅使用DETR值
5. 可视化与批量结果导出
"""

# 导入依赖库
import csv
import json
import os
import string
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from functools import cmp_to_key

from package_core.PackageExtract.BGA_Function.Pin_process.BGA_DETR_get_pins import detr_pin_XY  # DETR模型调用函数
from package_core.PackageExtract.BGA_Function.Pin_process.OCR import Run_onnx  # 自定义OCR模块
from PIL import Image
from typing import List, Tuple, Union, Optional

# 字母映射表（原逻辑保留）
LETTER_DICT: List[str] = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'T', 'U', 'V', 'W', 'Y',
    'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AJ', 'AK', 'AL', 'AM', 'AN', 'AP', 'AR', 'AT', 'AU', 'AV', 'AW',
    'AY',
    'BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG', 'BH', 'BJ', 'BK', 'BL', 'BM', 'BN', 'BP', 'BR', 'BT', 'BU', 'BV', 'BW',
    'BY'
]


def filter_boxes_by_aspect_ratio(
        boxes: List[List[Tuple[int, int]]],
        texts: List[str],
        aspect_ratio_threshold: float = 2.0
) -> Tuple[List[List[Tuple[int, int]]], List[str]]:
    """原逻辑保留：按长宽比过滤文本框"""
    if len(boxes) != len(texts):
        raise ValueError(f"boxes与texts长度不匹配！boxes长度：{len(boxes)}，texts长度：{len(texts)}")

    filtered_boxes = []
    filtered_texts = []
    for box, text in zip(boxes, texts):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)

        if width <= 0 or height <= 0:
            continue

        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio <= aspect_ratio_threshold:
            filtered_boxes.append(box)
            filtered_texts.append(text)

    print(f"长宽比过滤：原始{len(boxes)}个框 → 保留{len(filtered_boxes)}个框")
    return filtered_boxes, filtered_texts


def calculate_centers(boxes: List[List[Tuple[int, int]]]) -> np.ndarray:
    """原逻辑保留：计算文本框中心点"""
    centers = []
    for b in boxes:
        xs = [p[0] for p in b]
        ys = [p[1] for p in b]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        centers.append((cx, cy))
    return np.array(centers)


def get_letter_value(text: Union[str, List[str]], case_sensitive: bool = False) -> Union[int, List[int], None]:
    """原逻辑保留：字母文本转数值"""

    def _get_single_value(s: str) -> Optional[int]:
        if not case_sensitive:
            s = s.upper()
        if s in LETTER_DICT:
            return LETTER_DICT.index(s) + 1
        return None

    if isinstance(text, list):
        return [_get_single_value(s) for s in text]
    elif isinstance(text, str):
        return _get_single_value(text)
    else:
        raise TypeError("输入必须是字符串或字符串列表")


def cluster_comparison(cluster_a, cluster_b, size_diff_threshold=3):
    """簇比较函数：实现自定义排序逻辑（原逻辑保留）"""
    len_a = -cluster_a[0]
    len_b = -cluster_b[0]
    var_a = cluster_a[1]
    var_b = cluster_b[1]

    len_diff = abs(len_a - len_b)

    if len_diff > size_diff_threshold:
        return -1 if len_a > len_b else 1
    else:
        return -1 if var_a < var_b else 1 if var_a > var_b else 0


def find_vertical_clusters(boxes: List[List[Tuple[int, int]]], texts: List[str], centers: np.ndarray,
                           x_thresh: int = 15, min_len: int = 3) -> List[List[int]]:
    """返回前2个最优垂直簇（原逻辑保留）"""
    if len(centers) == 0:
        return []

    idx_sorted = np.argsort(centers[:, 0])
    clusters_with_metrics = []
    group = [idx_sorted[0]]

    for i in idx_sorted[1:]:
        x_diff = abs(centers[i, 0] - centers[group[-1], 0])
        if x_diff < x_thresh:
            group.append(i)
        else:
            if len(group) >= min_len:
                cluster_x = centers[group, 0]
                x_variance = np.var(cluster_x)
                clusters_with_metrics.append((-len(group), x_variance, group))
            group = [i]

    if len(group) >= min_len:
        cluster_x = centers[group, 0]
        x_variance = np.var(cluster_x)
        clusters_with_metrics.append((-len(group), x_variance, group))

    clusters_with_metrics.sort(key=cmp_to_key(cluster_comparison))
    top_clusters = [item[2] for item in clusters_with_metrics[:2]]

    # ============= 新增：空内容占比过滤（和水平簇逻辑完全一致） =============
    def is_cluster_valid(cluster: List[int]) -> bool:
        """判断簇是否有效：空内容占比 < 80%"""
        total_count = len(cluster)
        empty_count = 0
        for idx in cluster:
            # 复用清洗后的texts，直接判断是否为空（避免重复清洗）
            cleaned_text = texts[idx].strip()
            if len(cleaned_text) == 0:
                empty_count += 1
        empty_ratio = empty_count / total_count if total_count > 0 else 1.0
        print(f"垂直簇（{total_count}个点）空内容占比：{empty_ratio:.2%}")
        return empty_ratio < 0.8  # 空占比<80%才有效

    # 过滤无效簇（保留空占比<80%的簇）
    valid_top_clusters = []
    for cluster in top_clusters:
        if is_cluster_valid(cluster):
            valid_top_clusters.append(cluster)

    # --------------- 最终返回结果 ---------------
    return valid_top_clusters


def find_horizontal_clusters(boxes: List[List[Tuple[int, int]]], texts: List[str], centers: np.ndarray,
                             y_thresh: int = 15, min_len: int = 3,
                             x_variance_thresh: float = 100.0) -> List[List[int]]:
    """改进的水平聚类：增加x坐标分布检查"""
    if len(centers) == 0:
        return []

    idx_sorted = np.argsort(centers[:, 1])
    clusters_with_metrics = []
    current_group = [idx_sorted[0]]

    for i in idx_sorted[1:]:
        y_diff = abs(centers[i, 1] - centers[current_group[-1], 1])
        if y_diff < y_thresh:
            current_group.append(i)
        else:
            if len(current_group) >= min_len:
                # 检查x坐标的分布：水平排列应该有较大的x方差
                cluster_x = centers[current_group, 0]
                cluster_y = centers[current_group, 1]
                x_variance = np.var(cluster_x)
                y_variance = np.var(cluster_y)

                # 水平簇应该满足：x方向分布较广，y方向分布较集中
                if x_variance > x_variance_thresh and x_variance > y_variance:
                    clusters_with_metrics.append((-len(current_group), y_variance, current_group))
            current_group = [i]

    if len(current_group) >= min_len:
        cluster_x = centers[current_group, 0]
        cluster_y = centers[current_group, 1]
        x_variance = np.var(cluster_x)
        y_variance = np.var(cluster_y)

        if x_variance > x_variance_thresh and x_variance > y_variance:
            clusters_with_metrics.append((-len(current_group), y_variance, current_group))

    clusters_with_metrics.sort(key=cmp_to_key(cluster_comparison))
    top_clusters = [item[2] for item in clusters_with_metrics[:2]]

    # ============= 新增：空内容占比过滤（核心优化） =============
    def is_cluster_valid(cluster: List[int]) -> bool:
        """判断簇是否有效：空内容占比 < 80%"""
        total_count = len(cluster)
        empty_count = 0
        for idx in cluster:
            # texts是已被filter_boxes_texts清洗后的文本，直接判断是否为空
            cleaned_text = texts[idx].strip()
            if len(cleaned_text) == 0:
                empty_count += 1
        empty_ratio = empty_count / total_count if total_count > 0 else 1.0
        print(f"簇（{total_count}个点）空内容占比：{empty_ratio:.2%}")
        return empty_ratio < 0.8  # 空占比<80%才有效

    # 过滤无效簇（保留空占比<80%的簇）
    valid_top_clusters = []
    for cluster in top_clusters:
        if is_cluster_valid(cluster):
            valid_top_clusters.append(cluster)

    # --------------- 最终返回结果 ---------------
    # 若过滤后只剩1个有效簇，返回1个；若2个都有效，返回2个；若都无效，返回空
    return valid_top_clusters


def find_origin_and_directions(h_cluster: List[int], v_cluster: List[int], centers: np.ndarray) -> Tuple[
    Optional[Tuple[float, float]], Optional[np.ndarray], Optional[np.ndarray]]:
    """原逻辑保留：确定坐标原点与方向向量（主要用于可视化）"""
    h_empty = len(h_cluster) == 0
    v_empty = len(v_cluster) == 0

    if h_empty and v_empty:
        return None, None, None

    elif not h_empty and v_empty:
        origin = centers[h_cluster[0]].copy()
        return origin, np.array([0, 1], dtype=np.float64), np.array([1, 0], dtype=np.float64)

    elif h_empty and not v_empty:
        origin = centers[v_cluster[0]].copy()
        return origin, np.array([0, 1], dtype=np.float64), np.array([1, 0], dtype=np.float64)

    else:
        min_distance = float('inf')
        origin_h_idx = None
        origin_v_idx = None

        for h_idx in h_cluster:
            for v_idx in v_cluster:
                distance = np.linalg.norm(centers[h_idx] - centers[v_idx])
                if distance < min_distance:
                    min_distance = distance
                    origin_h_idx = h_idx
                    origin_v_idx = v_idx

        origin = (
            (centers[origin_h_idx][0] + centers[origin_v_idx][0]) / 2,
            (centers[origin_h_idx][1] + centers[origin_v_idx][1]) / 2
        )

        # 垂直簇方向向量（y轴）
        v_points = [centers[idx] for idx in v_cluster]
        v_mean = np.mean(v_points, axis=0)
        v_cov = np.cov(np.array(v_points).T)
        v_eigenvalues, v_eigenvectors = np.linalg.eig(v_cov)
        v_direction = v_eigenvectors[:, np.argmax(v_eigenvalues)]

        # 水平簇方向向量（x轴）
        h_points = [centers[idx] for idx in h_cluster]
        h_mean = np.mean(h_points, axis=0)
        h_cov = np.cov(np.array(h_points).T)
        h_eigenvalues, h_eigenvectors = np.linalg.eig(h_cov)
        h_direction = h_eigenvectors[:, np.argmax(h_eigenvalues)]

        return origin, v_direction, h_direction


def sort_cluster(cluster: List[int], centers: np.ndarray, is_horizontal: bool) -> List[int]:
    """按坐标升序排序（原逻辑保留）"""
    if not cluster:
        return []

    if is_horizontal:
        sorted_indices = sorted(cluster, key=lambda idx: centers[idx, 0])
    else:
        sorted_indices = sorted(cluster, key=lambda idx: centers[idx, 1])

    return sorted_indices


def assign_coordinates(sorted_h: List[int], sorted_v: List[int], origin: Optional[Tuple[float, float]],
                       x_direction: Optional[np.ndarray], y_direction: Optional[np.ndarray],
                       centers: np.ndarray) -> Tuple[dict, dict]:
    """原逻辑保留：为簇分配坐标标签"""
    h_coords = {}
    v_coords = {}
    valid_x_dir = x_direction is not None and len(x_direction) > 0
    valid_y_dir = y_direction is not None and len(y_direction) > 0

    if sorted_h and origin is not None and valid_x_dir and valid_y_dir:
        for i, idx in enumerate(sorted_h):
            rel_x = np.dot(centers[idx] - origin, x_direction)
            rel_y = np.dot(centers[idx] - origin, y_direction)
            coord = f"X{i + 1}"
            h_coords[idx] = (coord, rel_x, rel_y)

    if sorted_v and origin is not None and valid_x_dir and valid_y_dir:
        for i, idx in enumerate(sorted_v):
            rel_x = np.dot(centers[idx] - origin, x_direction)
            rel_y = np.dot(centers[idx] - origin, y_direction)
            coord = f"Y{i + 1}"
            v_coords[idx] = (coord, rel_x, rel_y)

    return h_coords, v_coords


def filter_boxes_texts(
        boxes: List[List[Tuple[int, int]]],
        texts: List[str],
        substrings_to_remove: List[str] = ['00'],
        chars_to_remove: str = ',;!o$-.?'
) -> Tuple[List[List[Tuple[int, int]]], List[str]]:
    """原逻辑保留：文本内容清洗"""
    if len(boxes) != len(texts):
        raise ValueError(f"boxes与texts长度不匹配！boxes长度：{len(boxes)}，texts长度：{len(texts)}")

    if len(texts) == 0:
        print("警告：输入texts为空，直接返回空列表")
        return [], []

    filtered_boxes = []
    filtered_texts = []
    char_translation = str.maketrans('', '', chars_to_remove)

    err_text = ['回','+']

    for box, text in zip(boxes, texts):
        if text is None or text in err_text:
            continue
        if text == '0':
            text = ''

        cleaned = text
        for substr in substrings_to_remove:
            cleaned = cleaned.replace(substr, '')
        cleaned = cleaned.translate(char_translation).strip()

        if len(cleaned) == 0 or len(cleaned) <= 3:
            if len(cleaned) == 3:
                cleaned = cleaned[:2]
            filtered_boxes.append(box)
            filtered_texts.append(cleaned)

    return filtered_boxes, filtered_texts


def get_cluster_avg_coord(cluster: List[int], centers: np.ndarray, is_horizontal: bool) -> float:
    """计算簇的平均坐标（原逻辑保留）"""
    if not cluster:
        return float('inf')
    target_coords = centers[cluster, 1] if is_horizontal else centers[cluster, 0]
    return np.mean(target_coords)


def combine_close_clusters(clusters: List[List[int]], centers: np.ndarray, is_horizontal: bool,
                           close_thresh: int = 50) -> Tuple[List[int], bool]:
    """返回合并后的簇 + 是否合并的标记（用于单行单列判断）"""
    merged = False
    if len(clusters) < 2:
        return clusters[0] if clusters else [], merged

    cluster1, cluster2 = clusters[0], clusters[1]
    avg1 = get_cluster_avg_coord(cluster1, centers, is_horizontal)
    avg2 = get_cluster_avg_coord(cluster2, centers, is_horizontal)
    cluster_distance = abs(avg1 - avg2)

    if cluster_distance <= close_thresh:
        combined_cluster = list(set(cluster1 + cluster2))
        merged = True
        print(
            f"✅ 合并{'水平' if is_horizontal else '垂直'}簇：簇1（{len(cluster1)}个点）与簇2（{len(cluster2)}个点）距离{cluster_distance:.1f}≤{close_thresh}，合并后共{len(combined_cluster)}个点")
    else:
        combined_cluster = cluster1  # 不合并，取最优簇
        print(
            f"❌ {'水平' if is_horizontal else '垂直'}簇不合并：簇1与簇2距离{cluster_distance:.1f}>{close_thresh}，保留最优簇（{len(cluster1)}个点）")

    return combined_cluster, merged


def visualize_with_sorting(image_path: str, boxes: List[List[Tuple[int, int]]], texts: List[str],
                           h_cluster: List[int], v_cluster: List[int], centers: np.ndarray,
                           h_clusters_original: List[List[int]], v_clusters_original: List[List[int]]) -> Tuple[
    List[int], List[str], List[int], List[str]]:
    """可视化函数（原逻辑保留）"""
    img = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(14, 14))
    ax = plt.gca()
    ax.imshow(img)
    ax.axis('on')

    origin, y_direction, x_direction = find_origin_and_directions(h_cluster, v_cluster, centers)

    sorted_h = sort_cluster(h_cluster, centers, is_horizontal=True) if h_cluster else []
    sorted_v = sort_cluster(v_cluster, centers, is_horizontal=False) if v_cluster else []

    sorted_h_text = [texts[idx] for idx in sorted_h] if sorted_h else []
    sorted_v_text = [texts[idx] for idx in sorted_v] if sorted_v else []

    h_coords, v_coords = assign_coordinates(sorted_h, sorted_v, origin, x_direction, y_direction, centers)

    # 绘制水平簇
    if sorted_h:
        is_h_merged = len(h_clusters_original) >= 2 and \
                      abs(get_cluster_avg_coord(h_clusters_original[0], centers, True) -
                          get_cluster_avg_coord(h_clusters_original[1], centers, True)) <= 30
        styles_h = [('red', 'solid'), ('darkred', 'dashed')] if is_h_merged else [('red', 'solid')]
        idx_to_h_cluster = {}
        if is_h_merged:
            for cluster_idx, orig_cluster in enumerate(h_clusters_original[:2]):
                for idx in orig_cluster:
                    idx_to_h_cluster[idx] = cluster_idx

        for idx in sorted_h:
            box = boxes[idx]
            poly = np.array(box + [box[0]])
            if is_h_merged:
                cluster_idx = idx_to_h_cluster.get(idx, 0)
                color, linestyle = styles_h[cluster_idx]
                label = 'Horizontal Cluster 1' if (idx == sorted_h[0] and cluster_idx == 0) else (
                    'Horizontal Cluster 2 (Merged)' if (idx == sorted_h[0] and cluster_idx == 1) else "")
            else:
                color, linestyle = styles_h[0]
                label = 'Horizontal Cluster' if idx == sorted_h[0] else ""

            ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=2.0, linestyle=linestyle, label=label)
            text_label = f"{texts[idx]}"
            ax.text(centers[idx][0] + 5, centers[idx][1] + 5, text_label, fontsize=10, color='red', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

    # 绘制垂直簇
    if sorted_v:
        is_v_merged = len(v_clusters_original) >= 2 and \
                      abs(get_cluster_avg_coord(v_clusters_original[0], centers, False) -
                          get_cluster_avg_coord(v_clusters_original[1], centers, False)) <= 30
        styles_v = [('blue', 'solid'), ('darkblue', 'dashed')] if is_v_merged else [('blue', 'solid')]
        idx_to_v_cluster = {}
        if is_v_merged:
            for cluster_idx, orig_cluster in enumerate(v_clusters_original[:2]):
                for idx in orig_cluster:
                    idx_to_v_cluster[idx] = cluster_idx

        for idx in sorted_v:
            box = boxes[idx]
            poly = np.array(box + [box[0]])
            if is_v_merged:
                cluster_idx = idx_to_v_cluster.get(idx, 0)
                color, linestyle = styles_v[cluster_idx]
                label = 'Vertical Cluster 1' if (idx == sorted_v[0] and cluster_idx == 0) else (
                    'Vertical Cluster 2 (Merged)' if (idx == sorted_v[0] and cluster_idx == 1) else "")
            else:
                color, linestyle = styles_v[0]
                label = 'Vertical Cluster' if idx == sorted_v[0] else ""

            ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=2.0, linestyle=linestyle, label=label)
            text_label = f"{texts[idx]}"
            ax.text(centers[idx][0] + 5, centers[idx][1] + 5, text_label, fontsize=10, color='blue', weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

    # 绘制原点与网格线
    if origin is not None:
        origin_x, origin_y = origin
        ax.scatter(origin_x, origin_y, color='green', s=100, marker='*', label='Origin')
        ax.text(origin_x + 10, origin_y + 10, 'Origin', fontsize=12, color='green', weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=12)
    ax.set_title('Clusters (Merged + Coordinate Sorting)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

    return sorted_h, sorted_h_text, sorted_v, sorted_v_text


def is_letter_list(str_list: List[str]) -> bool:
    """原逻辑保留：判断文本列表是否以字母为主"""
    letters = set(string.ascii_letters)
    digits = set(string.digits)
    letter_feature_count = 0
    non_letter_feature_count = 0

    for s in str_list:
        s = s.strip()
        if len(s) > 3 or len(s) == 0:
            continue

        if len(s) == 1:
            if s[0] in letters:
                letter_feature_count += 1
            elif s[0] in digits:
                non_letter_feature_count += 1
            continue

        if len(s) == 2:
            if s[0] in letters:
                letter_feature_count += 1
            elif s[0] in digits:
                non_letter_feature_count += 1
            continue

        if len(s) == 3:
            first_char = s[0]
            last_char = s[-1]
            if first_char in letters and last_char in letters:
                letter_feature_count += 1
            elif first_char in digits and last_char in digits:
                non_letter_feature_count += 1
            else:
                inner_letter = 0
                inner_digit = 0
                for c in s:
                    if c in letters:
                        inner_letter += 1
                    elif c in digits:
                        inner_digit += 1
                if inner_letter > inner_digit:
                    letter_feature_count += 1
                else:
                    non_letter_feature_count += 1

    total = letter_feature_count + non_letter_feature_count
    if total == 0:
        return False
    return letter_feature_count > non_letter_feature_count


def has_valid_digit_feature(str_list: List[str]) -> bool:
    """原逻辑保留：判断是否包含有效数字"""
    if not str_list:
        return False
    for s in str_list:
        cleaned_text = s.strip()
        if len(cleaned_text) <= 3 and cleaned_text.isdigit():
            return True
    return False


def get_valid_with_positions(str_list: List[str], is_digit: bool) -> List[Tuple[int, int]]:
    """原逻辑保留：提取有效元素及其索引"""
    valid_list = []
    for idx, text in enumerate(str_list):
        cleaned_text = text.strip()
        if is_digit:
            if len(cleaned_text) <= 3 and cleaned_text.isdigit():
                valid_list.append((int(cleaned_text), idx))
        else:
            val = get_letter_value(cleaned_text)
            if val is not None:
                valid_list.append((val, idx))
    return valid_list


def get_sequence_trend(valid_values: List[int]) -> str:
    """原逻辑保留：分析序列趋势"""
    if len(valid_values) < 2:
        return "flat"

    steps = [valid_values[i] - valid_values[i - 1] for i in range(1, len(valid_values))]
    positive_steps = sum(1 for s in steps if s > 0)
    negative_steps = sum(1 for s in steps if s < 0)

    if positive_steps > negative_steps:
        return "positive"
    elif negative_steps > positive_steps:
        return "negative"
    else:
        return "flat"


def is_ordered_sequence(values: List[int], trend: str) -> bool:
    """原逻辑保留：判断序列是否有序"""
    if len(values) < 2:
        return True
    if trend == "positive":
        return all(values[i] < values[i + 1] for i in range(len(values) - 1))
    elif trend == "negative":
        return all(values[i] > values[i + 1] for i in range(len(values) - 1))
    else:
        return True


def find_two_consecutive_ordered(values: List[int], trend: str) -> Tuple[bool, List[int]]:
    """原逻辑保留：寻找符合趋势的连续序列"""
    if len(values) < 2:
        return False, []

    expected_step = 1 if trend == "positive" else -1

    if trend == "positive":
        for i in range(len(values) - 2, -1, -1):
            a, b = values[i], values[i + 1]
            if b - a == expected_step:
                return True, [a, b]
    else:
        for i in range(len(values) - 1):
            a, b = values[i], values[i + 1]
            if b - a == expected_step:
                return True, [a, b]

    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            a, b = values[i], values[j]
            if b - a == expected_step:
                return True, [a, b]

    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            a, b = values[i], values[j]
            if (trend == "positive" and b > a) or (trend == "negative" and b < a):
                return True, [a, b]

    return False, []


def calculate_adjusted_value(str_list: List[str], is_digit: bool) -> Optional[int]:
    """原逻辑保留：计算OCR调整后的值"""
    valid_with_pos = get_valid_with_positions(str_list, is_digit)
    if not valid_with_pos:
        return None

    valid_values = [val for val, idx in valid_with_pos]
    valid_indices = [idx for val, idx in valid_with_pos]
    total_length = len(str_list)
    trend = get_sequence_trend(valid_values)
    cluster_size = len(valid_values)

    adjusted = None
    last_val, last_idx = valid_with_pos[-1] if trend == "positive" else (None, None)
    first_val, first_idx = valid_with_pos[0] if trend == "negative" else (None, None)

    if trend == "positive":
        invalid_steps = total_length - last_idx - 1
        adjusted = last_val + invalid_steps
        print(f"📈 正增长趋势：有效值{valid_values} → 初始调整后{adjusted}")
    elif trend == "negative":
        invalid_steps = first_idx
        adjusted = first_val + invalid_steps
        print(f"📉 负增长趋势：有效值{valid_values} → 初始调整后{adjusted}")
    else:
        adjusted = max(valid_values)
        print(f"⚖️  平稳趋势：有效值{valid_values} → 最大值{adjusted}")

    need_repair = False
    check_values = []
    if trend in ["positive", "negative"] and adjusted < cluster_size:
        if len(valid_values) >= 3:
            check_values = valid_values[-3:] if trend == "positive" else valid_values[:3]
            is_ordered = is_ordered_sequence(check_values, trend)
            if not is_ordered:
                need_repair = True
                print(f"⚠️  调整值{adjusted} < 簇大小{cluster_size}，关联值{check_values}无序，尝试修复...")

    if need_repair:
        found, seq_values = find_two_consecutive_ordered(valid_values, trend)
        if found and len(seq_values) == 2:
            step = seq_values[1] - seq_values[0]

            if trend == "negative":
                corrected_first = seq_values[0] + abs(step)
                adjusted = corrected_first + invalid_steps
                print(f"✅ 负增长修复：第一个值{valid_values[0]}→{corrected_first}，调整后{adjusted}")

            elif trend == "positive":
                last_valid_idx = valid_values.index(seq_values[1])
                correct_last_val = seq_values[1] + step * (len(valid_values) - 1 - last_valid_idx)
                adjusted = correct_last_val + invalid_steps
                print(
                    f"✅ 正增长修复：参考序列{seq_values}（步长{step}），最后值{last_val}→{correct_last_val}，调整后{adjusted}")

    if not is_digit and adjusted is not None:
        if adjusted < 1 or adjusted > len(LETTER_DICT):
            print(f"❌ 调整值{adjusted}不在字母映射范围内（1~{len(LETTER_DICT)}）")
            return None
        else:
            print(f"✅ 有效字母值：{adjusted}（对应{LETTER_DICT[adjusted - 1]}）")

    return adjusted


# -------------------------- 簇大小触发DETR的判定逻辑（核心修改） --------------------------
def is_small_clusters(
        h_cluster: List[int],
        v_cluster: List[int],
        size_threshold: int = 20
) -> bool:
    """判断水平簇和垂直簇的文本框数量是否都小于阈值（默认20）"""
    h_size = len(h_cluster)
    v_size = len(v_cluster)
    print(f"🔍 簇大小判定：水平簇{h_size}个点，垂直簇{v_size}个点（阈值{size_threshold}）")
    is_small = h_size < size_threshold and v_size < size_threshold
    print(f"✅ 是否触发DETR：{is_small}（水平簇<{size_threshold}且垂直簇<{size_threshold}）")
    return is_small


# -------------------------- DETR与OCR结果处理函数 --------------------------
def get_max_coord(ocr_val: Optional[int], detr_val: Optional[int], coord_type: str) -> Optional[int]:
    """簇大小触发场景：取OCR与DETR最大值（原逻辑保留）"""
    print(f"📊 {coord_type}值对比：OCR={ocr_val}, DETR={detr_val}")

    if ocr_val is None and detr_val is None:
        print(f"❌ {coord_type}值：OCR与DETR均无有效数据")
        return None
    elif ocr_val is None:
        print(f"✅ {coord_type}值：OCR无数据，取DETR值{detr_val}")
        return detr_val
    elif detr_val is None:
        print(f"✅ {coord_type}值：DETR无数据，取OCR值{ocr_val}")
        return ocr_val
    else:
        max_val = max(ocr_val, detr_val)
        print(f"✅ {coord_type}值：取OCR与DETR最大值{max_val}")
        return max_val


def convert_quad_to_rect(
        boxes: List[List[List[Union[int, float]]]]
) -> List[List[Union[int, float]]]:
    """
    将四边形框格式 [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ...]
    转换为轴对齐矩形框格式 [[x1,y1,x2,y2], ...]

    转换规则：
    x1 = 四边形所有x坐标的最小值
    y1 = 四边形所有y坐标的最小值
    x2 = 四边形所有x坐标的最大值
    y2 = 四边形所有y坐标的最大值

    参数:
        boxes: 四边形框列表（支持int/float类型坐标）

    返回:
        轴对齐矩形框列表
    """
    rect_boxes = []
    for idx, quad in enumerate(boxes):
        try:
            # 验证四边形格式（必须是4个(x,y)坐标）
            if not isinstance(quad, list) or len(quad) != 4:
                raise ValueError(f"格式错误：需4个坐标点，实际为{len(quad)}个")

            # 转换为numpy数组，方便计算极值
            quad_np = np.array(quad, dtype=np.float32)
            if quad_np.shape != (4, 2):
                raise ValueError(f"坐标维度错误：需(4,2)，实际为{quad_np.shape}")

            # 计算x/y的极值，得到轴对齐矩形
            x1 = float(np.min(quad_np[:, 0]))
            y1 = float(np.min(quad_np[:, 1]))
            x2 = float(np.max(quad_np[:, 0]))
            y2 = float(np.max(quad_np[:, 1]))

            # 保持与原坐标类型一致（int/float）
            if all(isinstance(coord, int) for point in quad for coord in point):
                rect_boxes.append([int(x1), int(y1), int(x2), int(y2)])
            else:
                rect_boxes.append([x1, y1, x2, y2])

        except Exception as e:
            print(f"⚠️  第{idx + 1}个四边形转换失败：{str(e)}，跳过该框")
            continue

    return rect_boxes


def write_boxes_to_json(
        new_boxes: List[List[Union[int, float]]],
        json_path: str,
        mode: str = "w",
        indent: int = 2,
        encoding: str = "utf-8",
        overwrite_confirm: bool = False
) -> bool:
    """
    将转换后的矩形框列表写入JSON文件

    参数:
        new_boxes: 矩形框列表，格式为[[x1,y1,x2,y2], ...]
        json_path: JSON文件保存路径（绝对路径/相对路径）
        mode: 写入模式（"w"=覆盖，"a"=追加，默认"w"）
        indent: JSON格式化缩进（默认2，0为紧凑格式）
        encoding: 文件编码（默认utf-8）
        overwrite_confirm: 覆盖模式下是否询问确认（默认False，直接覆盖）

    返回:
        bool: 写入成功返回True，失败返回False
    """
    # -------------------------- 1. 验证输入参数 --------------------------
    # 验证new_boxes格式
    if not isinstance(new_boxes, list):
        print("❌ 错误：new_boxes必须是列表格式")
        return False

    # 验证每个框的格式（可选，确保写入的数据合法）
    for idx, box in enumerate(new_boxes):
        if not isinstance(box, list) or len(box) != 4:
            print(f"⚠️  第{idx + 1}个框格式错误（需4个数值），已跳过该框")
            new_boxes.pop(idx)

    if len(new_boxes) == 0:
        print("❌ 错误：无有效矩形框数据，取消写入")
        return False

    # 验证写入模式
    if mode not in ["w", "a"]:
        print("❌ 错误：mode只能是'w'（覆盖）或'a'（追加）")
        return False

    # -------------------------- 2. 处理文件路径 --------------------------
    json_path = Path(json_path)
    # 自动创建父目录（如果不存在）
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # 覆盖模式下的确认（如果开启）
    if mode == "w" and json_path.exists() and overwrite_confirm:
        confirm = input(f"⚠️  文件已存在：{json_path}\n是否覆盖？（y/n，默认y）：").strip().lower()
        if confirm not in ["", "y", "yes"]:
            print("✅ 已取消覆盖，写入终止")
            return False

    # -------------------------- 3. 写入JSON文件 --------------------------
    try:
        if mode == "a" and json_path.exists():
            # 追加模式：读取已有数据，合并后写入
            with open(json_path, 'r', encoding=encoding) as f:
                existing_data = json.load(f)
            # 验证已有数据格式（必须是列表）
            if not isinstance(existing_data, list):
                print(f"⚠️  已有文件格式错误（需列表），将覆盖原有内容")
                existing_data = []
            # 合并新数据
            existing_data.extend(new_boxes)
            write_data = existing_data
        else:
            # 覆盖模式：直接写入新数据
            write_data = new_boxes

        # 写入JSON文件（格式化输出，方便查看）
        with open(json_path, 'w', encoding=encoding) as f:
            json.dump(write_data, f, ensure_ascii=False, indent=indent)
        return True

    except PermissionError:
        print(f"❌ 错误：无权限写入文件 → {json_path}（可能被占用）")
        return False
    except Exception as e:
        print(f"❌ 错误：写入JSON失败 → {str(e)}")
        return False

def BGA_get_PIN(image_path: str, visualize: bool = False, min_cluster_len: int = 3) -> Tuple[
    List[str], List[str], Optional[int], Optional[int]]:
    """
    核心调整：
    1. 触发DETR的条件改为：最终水平簇和垂直簇文本框数量都小于20
    2. 若X或Y为None：强制触发DETR，最终结果仅使用DETR值
    """
    try:
        # 步骤1：OCR识别与过滤（原逻辑保留）
        boxes, texts = Run_onnx(image_path, 't')
        image_name = os.path.basename(image_path)
        new_boxes = convert_quad_to_rect(boxes)
        # 写入JSON文件
        JSON_SAVE_PATH = r"./BGA_bottom_DBNet_boxes.json"  # 保存路径
        write_boxes_to_json(
            new_boxes=new_boxes,
            json_path=JSON_SAVE_PATH,
            mode="w",  # "w"=覆盖，"a"=追加
            indent=2,  # 缩进2空格，增强可读性
            encoding="utf-8",
            overwrite_confirm=False  # 是否需要覆盖确认
        )


        print(f"\n{'=' * 50} 处理图像: {image_name} {'=' * 50}")
        print(f"OCR原始结果：{len(boxes)}个文本框，{len(texts)}个文本")

        boxes, texts = filter_boxes_by_aspect_ratio(boxes, texts)
        if len(boxes) == 0:
            print("❌ 长宽比过滤后无有效文本框，直接触发DETR")
            x, y = detr_pin_XY(image_path)
            return [], [], x, y

        boxes, texts = filter_boxes_texts(boxes, texts)
        print(f"内容过滤后：{len(texts)}个文本 → {texts}")
        if len(boxes) == 0:
            print("❌ 内容过滤后无有效文本框，直接触发DETR")
            x, y = detr_pin_XY(image_path)
            return [], [], x, y

        # 步骤2：聚类与合并（原逻辑保留）
        centers = calculate_centers(boxes)
        print(f"\n[动态聚类] 第一步：用默认最小长度（{min_cluster_len}）判断簇存在性")
        h_clusters_init = find_horizontal_clusters(
            boxes, texts, centers, min_len=min_cluster_len
        )
        v_clusters_init = find_vertical_clusters(
            boxes, texts, centers, min_len=min_cluster_len
        )

        # 判断两个方向是否有有效簇（存在=至少有1个簇，且簇内点数量≥默认min_len）
        h_has_cluster = len(h_clusters_init) > 0 and any(len(c) >= min_cluster_len for c in h_clusters_init)
        v_has_cluster = len(v_clusters_init) > 0 and any(len(c) >= min_cluster_len for c in v_clusters_init)
        print(f"[动态聚类] 水平簇存在：{h_has_cluster}，垂直簇存在：{v_has_cluster}")

        # 3.2 根据存在性调整最小长度，重新执行聚类
        h_clusters = []
        v_clusters = []
        if h_has_cluster and not v_has_cluster:
            # 水平簇存在，垂直簇不存在→垂直聚类最小长度改为2
            print(f"[动态聚类] 水平簇存在，垂直簇不存在→垂直聚类最小长度调整为2")
            h_clusters = h_clusters_init  # 水平簇保持初始结果
            v_clusters = find_vertical_clusters(
                boxes, texts, centers, min_len=2  # 调整为2
            )
        elif v_has_cluster and not h_has_cluster:
            # 垂直簇存在，水平簇不存在→水平聚类最小长度改为2
            print(f"[动态聚类] 垂直簇存在，水平簇不存在→水平聚类最小长度调整为2")
            v_clusters = v_clusters_init  # 垂直簇保持初始结果
            h_clusters = find_horizontal_clusters(
                boxes, texts, centers, min_len=2  # 调整为2
            )
        else:
            # 两者都存在/都不存在→保持默认最小长度
            print(f"[动态聚类] 无需调整最小长度（两者都存在/都不存在）")
            h_clusters = h_clusters_init
            v_clusters = v_clusters_init

        print(f"聚类结果：水平方向{len(h_clusters)}个有效簇，垂直方向{len(v_clusters)}个有效簇")

        h_cluster_combined, h_merged = combine_close_clusters(h_clusters, centers, is_horizontal=True)
        v_cluster_combined, v_merged = combine_close_clusters(v_clusters, centers, is_horizontal=False)
        print(f"合并后：水平簇{len(h_cluster_combined)}个点，垂直簇{len(v_cluster_combined)}个点")

        # 步骤3：排序与可视化（原逻辑保留）
        if visualize:
            sorted_h_idx, sorted_h_text, sorted_v_idx, sorted_v_text = visualize_with_sorting(
                image_path, boxes, texts, h_cluster_combined, v_cluster_combined, centers,
                h_clusters, v_clusters
            )
        else:
            sorted_h = sort_cluster(h_cluster_combined, centers, True) if h_cluster_combined else []
            sorted_v = sort_cluster(v_cluster_combined, centers, False) if v_cluster_combined else []
            sorted_h_text = [texts[idx] for idx in sorted_h] if sorted_h else []
            sorted_v_text = [texts[idx] for idx in sorted_v] if sorted_v else []

        print(f"\n排序后：水平簇文本{len(sorted_h_text)}个 → {sorted_h_text}")
        print(f"排序后：垂直簇文本{len(sorted_v_text)}个 → {sorted_v_text}")

        # 步骤4：计算OCR值（基础逻辑保留）
        X, Y = None, None
        ocr_x, ocr_y = None, None
        detr_x, detr_y = None, None
        print("\n=== 数值计算 ===")

        # 情况1：仅垂直簇（原逻辑保留）
        if v_cluster_combined and not h_cluster_combined:
            print("→ 仅检测到垂直簇，计算Y值")
            is_v_letter = is_letter_list(sorted_v_text)
            if is_v_letter:
                Y = calculate_adjusted_value(sorted_v_text, False)
            else:
                if has_valid_digit_feature(sorted_v_text):
                    Y = calculate_adjusted_value(sorted_v_text, True)
            print(f"→ OCR Y值：{Y if Y else '无有效数据'}")

        # 情况2：仅水平簇（原逻辑保留）
        elif h_cluster_combined and not v_cluster_combined:
            print("→ 仅检测到水平簇，计算X值")
            is_h_letter = is_letter_list(sorted_h_text)
            if is_h_letter:
                X = calculate_adjusted_value(sorted_h_text, False)
            else:
                if has_valid_digit_feature(sorted_h_text):
                    X = calculate_adjusted_value(sorted_h_text, True)
            print(f"→ OCR X值：{X if X else '无有效数据'}")

        # 情况3：水平+垂直簇（核心修改：簇大小触发DETR）
        else:
            print("→ 检测到水平+垂直簇，计算OCR值")
            is_v_letter = is_letter_list(sorted_v_text)
            if is_v_letter:
                ocr_y = calculate_adjusted_value(sorted_v_text, False)
                print(f"→ 垂直簇（字母）OCR Y值：{ocr_y if ocr_y else '无有效字母'}")
                if has_valid_digit_feature(sorted_h_text):
                    ocr_x = calculate_adjusted_value(sorted_h_text, True)
                    print(f"→ 水平簇（数字）OCR X值：{ocr_x if ocr_x else '无有效数字'}")
            else:
                if has_valid_digit_feature(sorted_v_text):
                    ocr_y = calculate_adjusted_value(sorted_v_text, True)
                    print(f"→ 垂直簇（数字）OCR Y值：{ocr_y if ocr_y else '无有效数字'}")
                if is_letter_list(sorted_h_text):
                    ocr_x = calculate_adjusted_value(sorted_h_text, False)
                    print(f"→ 水平簇（字母）OCR X值：{ocr_x if ocr_x else '无有效字母'}")

            # 核心修改：簇大小触发DETR（水平簇和垂直簇都小于阈值）
            print("\n=== 簇大小触发DETR判定 ===")
            size_threshold = 60
            is_small_clusters_flag = is_small_clusters(
                h_cluster_combined,
                v_cluster_combined,
                size_threshold=size_threshold
            )

            if is_small_clusters_flag:
                print(f"✅ 触发簇大小DETR调用（均<{size_threshold}）")
                detr_x, detr_y = detr_pin_XY(image_path)
                print(f"→ DETR识别结果：X={detr_x}, Y={detr_y}")
                X = get_max_coord(ocr_x, detr_x, "X")
                Y = get_max_coord(ocr_y, detr_y, "Y")
            else:
                print(f"❌ 簇大小不满足触发条件，使用OCR值")
                X = ocr_x
                Y = ocr_y

        # 核心新增：X或Y为None时强制触发DETR（原逻辑保留）
        if X is None or Y is None:
            print(f"\n⚠️ 检测到X={X}或Y={Y}为None，强制触发DETR并以其结果为准")
            detr_x_final, detr_y_final = detr_pin_XY(image_path)
            print(f"→ 最终DETR结果：X={detr_x_final}, Y={detr_y_final}")
            X, Y = detr_x_final, detr_y_final  # 完全覆盖原有值

        # 最终结果
        print(f"\n最终结果：X={X}, Y={Y}")
        return sorted_h_text, sorted_v_text, X, Y

    except Exception as e:
        print(f"处理图像 {image_name} 出错: {str(e)}")
        import traceback
        traceback.print_exc()
        # 异常情况也触发DETR
        x, y = detr_pin_XY(image_path)
        return [], [], x, y


# def process_batch_images(folder_path: str, output_csv: str = "results_final.csv", visualize: bool = False,
#                          cluster_size_threshold: int = 20):
#     """批量处理：更新DETR触发原因标记（适配簇大小条件）"""
#     image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
#     image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
#                    if any(f.lower().endswith(ext) for ext in image_extensions)]
#     image_files = natsorted(image_files)
#     print(f"找到 {len(image_files)} 张图像，开始批量处理...")
#
#     with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
#         fieldnames = ['Image', 'Horizontal_Texts', 'Vertical_Texts', 'X', 'Y', 'DETR_Trigger']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#
#         for image_path in image_files:
#             sorted_h_text, sorted_v_text, X, Y =  BGA_get_PIN(
#                 image_path, visualize, cluster_size_threshold
#             )
#
#             # 判定DETR触发原因（核心修改：适配簇大小条件）
#             detr_trigger = "No"
#             # 1. 无有效文本框（水平和垂直文本都为空）
#             if not sorted_h_text and not sorted_v_text:
#                 detr_trigger = "No Valid Boxes"
#
#             # 2. 簇大小触发（用文本数量近似簇大小）
#             else:
#                 h_size = len(sorted_h_text)
#                 v_size = len(sorted_v_text)
#                 # 若存在水平和垂直簇，且都小于阈值 → 簇大小触发
#                 if h_size > 0 and v_size > 0 and h_size < cluster_size_threshold and v_size < cluster_size_threshold:
#                     detr_trigger = f"Clusters < {cluster_size_threshold}"
#
#                 # 3. X/Y为None触发（结合文本有效性判断）
#                 else:
#                     # 若文本存在但X/Y为None → 强制DETR
#                     has_valid_text = (h_size > 0) or (v_size > 0)
#                     if has_valid_text and (X is None or Y is None):
#                         detr_trigger = "X/Y is None"
#
#             writer.writerow({
#                 'Image': os.path.basename(image_path),
#                 'Horizontal_Texts': ';'.join(sorted_h_text),
#                 'Vertical_Texts': ';'.join(sorted_v_text),
#                 'X': X if X is not None else '',
#                 'Y': Y if Y is not None else '',
#                 'DETR_Trigger': detr_trigger  # 标记DETR触发原因（新增簇大小条件）
#             })
#             print("-" * 100)
#
#     print(f"\n批量处理完成！结果已保存到 {output_csv}")


# 主程序入口
if __name__ == "__main__":
    # 示例1：单张图像处理
    image_path = r"D:\workspace\PackageWizard1.1\Result\Package_extract\data\bottom.jpg"
    # 可通过cluster_size_threshold参数调整触发阈值（默认20）
    sorted_h_text, sorted_v_text, X, Y =  BGA_get_PIN(image_path, visualize=True)
    print(f"最终X={X}, Y={Y}")

    # 示例2：批量处理
    # folder_path = "imgs/test/400_less"
    # process_batch_images(
    #     folder_path,
    #     output_csv="results_400_less.csv",
    #     visualize=False,
    #     cluster_size_threshold=20  # 簇大小触发阈值
    # )