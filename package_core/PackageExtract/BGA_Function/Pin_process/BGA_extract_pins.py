# -*- coding: utf-8 -*-
"""
图像文本检测与坐标计算脚本（重构完整版）
核心功能：
1. OCR识别与文本框过滤
2. 动态水平/垂直聚类（支持双簇合并）
3. YOLO与OCR结果融合（支持小簇触发、None值强制触发）
4. Matplotlib可视化（包含簇连线、原点、方向向量）
5. BGA标准行列标签生成（支持A1角判断）
"""

# ===================== 1. 导入依赖库 =====================
import csv
import json
import os
import string
import time
from collections import Counter
from pathlib import Path
from functools import cmp_to_key
from typing import List, Tuple, Union, Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# === 自定义模块导入 ===
# 请确保这些路径在您的项目中是正确的
from package_core.PackageExtract.BGA_Function.Pin_process.BGA_DETR_get_pins import detr_pin_XY
from package_core.PackageExtract.BGA_Function.Pin_process.OCR import Run_onnx
from package_core.PackageExtract.BGA_Function.orientation_classifier import OrientationClassifier
from package_core.PackageExtract.yolox_onnx_py.model_paths import model_path, result_path

# ===================== 2. 全局常量 =====================
LETTER_DICT: List[str] = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'T', 'U', 'V', 'W', 'Y',
    'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AJ', 'AK', 'AL', 'AM', 'AN', 'AP', 'AR', 'AT', 'AU', 'AV', 'AW',
    'AY',
    'BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG', 'BH', 'BJ', 'BK', 'BL', 'BM', 'BN', 'BP', 'BR', 'BT', 'BU', 'BV', 'BW',
    'BY'
]

# 全局单例模型
_ORIENTATION_MODEL = None


# ===================== 3. 基础工具函数 =====================

def convert_quad_to_rect(boxes: List[List[List[Union[int, float]]]]) -> List[List[Union[int, float]]]:
    """将四边形框转换为轴对齐矩形框 [x1, y1, x2, y2]"""
    rect_boxes = []
    for idx, quad in enumerate(boxes):
        try:
            quad_np = np.array(quad, dtype=np.float32)
            if quad_np.shape != (4, 2):
                continue
            x1, y1 = np.min(quad_np, axis=0)
            x2, y2 = np.max(quad_np, axis=0)

            # 保持数据类型一致性
            is_int = all(isinstance(c, int) for p in quad for c in p)
            rect_boxes.append([int(x1), int(y1), int(x2), int(y2)] if is_int else [x1, y1, x2, y2])
        except Exception as e:
            print(f"⚠️ 框转换失败 (索引{idx}): {e}")
    return rect_boxes


def write_boxes_to_json(new_boxes: List[List[Any]], json_path: str, mode: str = "w") -> bool:
    """保存文本框坐标到JSON文件"""
    try:
        path_obj = Path(json_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        write_data = new_boxes
        if mode == "a" and path_obj.exists():
            with open(path_obj, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            if isinstance(existing, list):
                existing.extend(new_boxes)
                write_data = existing

        with open(path_obj, 'w', encoding='utf-8') as f:
            json.dump(write_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"❌ JSON写入失败: {e}")
        return False


def mask_clusters_in_image(image_path: str, boxes: List[List[Tuple[int, int]]], indices: List[int]):
    """对指定的文本框区域进行涂白掩膜处理"""
    try:
        img = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        for idx in indices:
            if idx < len(boxes):
                # Flatten box to [x1, y1, x2, y2, ...]
                xy = [coord for point in boxes[idx] for coord in point]
                draw.polygon(xy, fill='white', outline='white')
        save_path = result_path("Package_extract","clean_bottom","bottom.jpg")

        # 提取保存路径的目录部分，不存在则创建
        save_dir = os.path.dirname(save_path)  # 提取目录路径（比如"xxx/Package_extract/clean_bottom"）
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)  # 递归创建目录，exist_ok=True避免已存在时报错
        img.save(save_path)
        print(f"✅ [预处理] 原图掩膜完成：覆盖了 {len(indices)} 个文本框")
    except Exception as e:
        print(f"❌ 掩膜处理失败：{str(e)}")


def get_letter_value(text: Union[str, List[str]], case_sensitive: bool = False) -> Union[int, List[int], None]:
    """字母转数值 (A->1, B->2 ...)"""

    def _single(s: str):
        if not case_sensitive: s = s.upper()
        return LETTER_DICT.index(s) + 1 if s in LETTER_DICT else None

    if isinstance(text, list): return [_single(s) for s in text]
    return _single(text)


# ===================== 4. 趋势分析与数据清洗 =====================

def get_sequence_trend(valid_values: List[int]) -> str:
    """分析数值序列趋势：positive(增), negative(减), flat(平)"""
    if len(valid_values) < 2: return "flat"
    steps = [valid_values[i] - valid_values[i - 1] for i in range(1, len(valid_values))]
    pos = sum(1 for s in steps if s > 0)
    neg = sum(1 for s in steps if s < 0)
    if pos > neg: return "positive"
    if neg > pos: return "negative"
    return "flat"


def get_valid_with_positions(str_list: List[str], is_digit: bool) -> List[Tuple[int, int]]:
    """提取有效的数字或字母及其索引"""
    valid_list = []
    for idx, text in enumerate(str_list):
        cleaned = text.strip()
        if is_digit:
            if len(cleaned) <= 3 and cleaned.isdigit():
                valid_list.append((int(cleaned), idx))
        else:
            val = get_letter_value(cleaned)
            if val is not None:
                valid_list.append((val, idx))
    return valid_list


def determine_A1_corner(sorted_h_text: List[str], sorted_v_text: List[str]) -> str:
    """根据行列趋势判断A1引脚位置"""
    h_data = get_valid_with_positions(sorted_h_text, True) or get_valid_with_positions(sorted_h_text, False)
    h_vals = [v[0] for v in h_data]
    h_trend = get_sequence_trend(h_vals)

    v_data = get_valid_with_positions(sorted_v_text, False) or get_valid_with_positions(sorted_v_text, True)
    v_vals = [v[0] for v in v_data]
    v_trend = get_sequence_trend(v_vals)

    print(f"🧭 趋势检测: 水平={h_trend}, 垂直={v_trend}")

    if h_trend == "flat" or v_trend == "flat":
        return "Top-Left"

    if v_trend == "positive":
        return "Top-Left" if h_trend == "positive" else "Top-Right"
    else:  # v_trend == "negative"
        return "Bottom-Left" if h_trend == "positive" else "Bottom-Right"


def filter_boxes_by_aspect_ratio(boxes, texts, threshold=2.0):
    """按长宽比过滤"""
    res_boxes, res_texts = [], []
    for box, text in zip(boxes, texts):
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        if w > 0 and h > 0 and (max(w, h) / min(w, h) <= threshold):
            res_boxes.append(box)
            res_texts.append(text)
    return res_boxes, res_texts


def filter_boxes_texts(boxes, texts, sub_remove=['00'], char_remove=',;!o$-.?'):
    """清洗文本内容"""
    res_boxes, res_texts = [], []
    trans = str.maketrans('', '', char_remove)
    err_text = ['回', '+', '0']

    for box, text in zip(boxes, texts):
        if not text or text in err_text: continue
        cleaned = text
        for s in sub_remove: cleaned = cleaned.replace(s, '')
        cleaned = cleaned.translate(trans).strip()

        if 0 < len(cleaned) <= 3:
            if len(cleaned) == 3: cleaned = cleaned[:2]
            res_boxes.append(box)
            res_texts.append(cleaned)
    return res_boxes, res_texts


def is_letter_list(str_list: List[str]) -> bool:
    """判断列表是否主要由字母组成"""
    letters, digits = set(string.ascii_letters), set(string.digits)
    l_cnt, d_cnt = 0, 0
    for s in str_list:
        s = s.strip()
        if not s or len(s) > 3: continue
        chars = [c for c in s if c in letters or c in digits]
        l_inner = sum(1 for c in chars if c in letters)
        d_inner = sum(1 for c in chars if c in digits)
        if l_inner > d_inner:
            l_cnt += 1
        else:
            d_cnt += 1
    return l_cnt > d_cnt if (l_cnt + d_cnt) > 0 else False


def has_valid_digit_feature(str_list: List[str]) -> bool:
    """判断是否包含有效数字特征"""
    if not str_list: return False
    return any(s.strip().isdigit() and len(s.strip()) <= 3 for s in str_list)


# ===================== 5. 聚类算法核心 =====================

def calculate_centers(boxes) -> np.ndarray:
    """计算文本框中心点"""
    return np.array([((min(p[0] for p in b) + max(p[0] for p in b)) / 2,
                      (min(p[1] for p in b) + max(p[1] for p in b)) / 2) for b in boxes])


def cluster_comparison(cluster_a, cluster_b):
    """排序：优先长度大，其次方差小"""
    len_a, var_a = -cluster_a[0], cluster_a[1]
    len_b, var_b = -cluster_b[0], cluster_b[1]
    if abs(len_a - len_b) > 3: return -1 if len_a > len_b else 1
    return -1 if var_a < var_b else 1 if var_a > var_b else 0


def _get_valid_clusters(top_clusters, texts):
    """过滤空内容占比过高的簇"""
    valid = []
    for cl in top_clusters:
        empty = sum(1 for idx in cl if not texts[idx].strip())
        if len(cl) > 0 and (empty / len(cl)) < 0.8:
            valid.append(cl)
    return valid


def find_vertical_clusters(boxes, texts, centers, x_thresh=15, min_len=3):
    if len(centers) == 0: return []
    idx_sorted = np.argsort(centers[:, 0])
    metrics = []
    group = [idx_sorted[0]]

    for i in idx_sorted[1:]:
        if abs(centers[i, 0] - centers[group[-1], 0]) < x_thresh:
            group.append(i)
        else:
            if len(group) >= min_len:
                metrics.append((-len(group), np.var(centers[group, 0]), group))
            group = [i]
    if len(group) >= min_len:
        metrics.append((-len(group), np.var(centers[group, 0]), group))

    metrics.sort(key=cmp_to_key(cluster_comparison))
    return _get_valid_clusters([m[2] for m in metrics[:2]], texts)


def find_horizontal_clusters(boxes, texts, centers, y_thresh=15, min_len=3, x_var_thresh=100.0):
    if len(centers) == 0: return []
    idx_sorted = np.argsort(centers[:, 1])
    metrics = []
    group = [idx_sorted[0]]

    def _add_if_valid(grp):
        if len(grp) >= min_len:
            x_var = np.var(centers[grp, 0])
            y_var = np.var(centers[grp, 1])
            if x_var > x_var_thresh and x_var > y_var:
                metrics.append((-len(grp), y_var, grp))

    for i in idx_sorted[1:]:
        if abs(centers[i, 1] - centers[group[-1], 1]) < y_thresh:
            group.append(i)
        else:
            _add_if_valid(group)
            group = [i]
    _add_if_valid(group)

    metrics.sort(key=cmp_to_key(cluster_comparison))
    return _get_valid_clusters([m[2] for m in metrics[:2]], texts)


def combine_close_clusters(clusters, centers, is_horizontal, close_thresh=50):
    """合并距离过近的簇"""
    if len(clusters) < 2: return (clusters[0] if clusters else []), False

    idx = 1 if is_horizontal else 0
    avg1 = np.mean(centers[clusters[0], idx])
    avg2 = np.mean(centers[clusters[1], idx])

    if abs(avg1 - avg2) <= close_thresh:
        merged = list(set(clusters[0] + clusters[1]))
        print(f"✅ 合并{'水平' if is_horizontal else '垂直'}簇，共{len(merged)}点")
        return merged, True
    else:
        print(f"❌ 未合并{'水平' if is_horizontal else '垂直'}簇，保留最优{len(clusters[0])}点")
        return clusters[0], False


def sort_cluster(cluster, centers, is_horizontal):
    """对簇内索引按坐标排序"""
    if not cluster: return []
    key_idx = 0 if is_horizontal else 1
    return sorted(cluster, key=lambda idx: centers[idx, key_idx])


# ===================== 6. 可视化功能模块 (核心补全) =====================

def find_origin_and_directions(h_cluster: List[int], v_cluster: List[int], centers: np.ndarray):
    """确定坐标原点与方向向量（用于可视化）"""
    h_empty, v_empty = not h_cluster, not v_cluster

    if h_empty and v_empty:
        return None, None, None
    elif not h_empty and v_empty:
        return centers[h_cluster[0]].copy(), np.array([0, 1]), np.array([1, 0])
    elif h_empty and not v_empty:
        return centers[v_cluster[0]].copy(), np.array([0, 1]), np.array([1, 0])
    else:
        # 双向簇：寻找交点（最近点对中点）
        min_dist = float('inf')
        h_idx, v_idx = -1, -1
        for h in h_cluster:
            for v in v_cluster:
                dist = np.linalg.norm(centers[h] - centers[v])
                if dist < min_dist:
                    min_dist, h_idx, v_idx = dist, h, v

        origin = (centers[h_idx] + centers[v_idx]) / 2

        # 计算方向向量 (PCA)
        def _get_direction(indices):
            if len(indices) < 2: return np.array([1, 0])
            pts = centers[indices]
            cov = np.cov(pts.T)
            vals, vecs = np.linalg.eig(cov)
            return vecs[:, np.argmax(vals)]

        return origin, _get_direction(v_cluster), _get_direction(h_cluster)


def visualize_with_sorting(image_path: str, boxes: List, texts: List,
                           h_cluster: List, v_cluster: List, centers: np.ndarray,
                           h_clusters_orig: List, v_clusters_orig: List):
    """Matplotlib 可视化绘图"""
    try:
        img = Image.open(image_path).convert('RGB')
        plt.figure(figsize=(14, 14))
        ax = plt.gca()
        ax.imshow(img)
        ax.axis('on')

        origin, y_dir, x_dir = find_origin_and_directions(h_cluster, v_cluster, centers)

        # 辅助绘制函数
        def _plot_cluster(indices, clusters_orig, color, label_pre, is_merged):
            idx_map = {}
            if is_merged:
                for cid, cl in enumerate(clusters_orig[:2]):
                    for idx in cl: idx_map[idx] = cid

            styles = [(color, 'solid'), (f'dark{color}', 'dashed')]
            sorted_indices = sort_cluster(indices, centers, label_pre.startswith('H'))

            for i, idx in enumerate(sorted_indices):
                box = boxes[idx]
                poly = np.array(box + [box[0]])

                cid = idx_map.get(idx, 0) if is_merged else 0
                c, ls = styles[cid % 2]
                lbl = f"{label_pre} {cid + 1}" if (i == 0 and is_merged) else (label_pre if i == 0 else "")

                ax.plot(poly[:, 0], poly[:, 1], color=c, lw=2, ls=ls, label=lbl)
                ax.text(centers[idx][0], centers[idx][1], texts[idx], fontsize=10, color=c, weight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", fc='white', alpha=0.7))

        is_h_merged = len(h_clusters_orig) >= 2 and len(h_cluster) > len(h_clusters_orig[0])
        is_v_merged = len(v_clusters_orig) >= 2 and len(v_cluster) > len(v_clusters_orig[0])

        if h_cluster: _plot_cluster(h_cluster, h_clusters_orig, 'red', 'Horizontal', is_h_merged)
        if v_cluster: _plot_cluster(v_cluster, v_clusters_orig, 'blue', 'Vertical', is_v_merged)

        if origin is not None:
            ax.scatter(origin[0], origin[1], c='green', s=150, marker='*', label='Origin', zorder=10)

        ax.grid(True, alpha=0.3, ls='--')
        ax.legend(loc='upper right')
        ax.set_title(f'Analysis: {os.path.basename(image_path)}')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"⚠️ 可视化失败: {e}")


# ===================== 7. 业务逻辑与计算 =====================

def get_orientation_model():
    """懒加载方向分类模型；加载失败则返回 None（降级为不旋转）"""
    global _ORIENTATION_MODEL
    if _ORIENTATION_MODEL is not None:
        return _ORIENTATION_MODEL

    onnx_path = model_path("orientation_model", "resnet_orientation.onnx")
    try:
        _ORIENTATION_MODEL = OrientationClassifier(onnx_path)
    except Exception as e:
        _ORIENTATION_MODEL = None
        print(f"⚠️ [方向模型] 加载失败，已降级为 rot=0。模型路径: {onnx_path}")
        print(f"⚠️ [方向模型] 失败原因: {e}")
    return _ORIENTATION_MODEL


def check_orientation_by_clusters(image_path, boxes, cluster_indices):
    """基于聚类文本框的方向投票；模型不可用时返回 0"""
    if not cluster_indices:
        return 0

    classifier = get_orientation_model()
    if classifier is None:
        return 0

    try:
        full_img = Image.open(image_path).convert('RGB')
    except Exception:
        return 0

    votes = []
    print(f"🧭 [方向检测] 采样 {len(cluster_indices)} 个框...")
    for idx in cluster_indices:
        quad = boxes[idx]
        xs, ys = [p[0] for p in quad], [p[1] for p in quad]
        try:
            crop = full_img.crop((min(xs), min(ys), max(xs), max(ys)))
            orient, conf, _ = classifier.predict(crop)
            if conf > 0.7:
                votes.append(orient)
        except Exception:
            continue

    if not votes:
        return 0
    top_ori = Counter(votes).most_common(1)[0][0]
    mapping = {"0°": 0, "90°": 90, "180°": 180, "270°": 270}
    return mapping.get(top_ori, 0)


def calculate_adjusted_value(str_list: List[str], is_digit: bool) -> Optional[int]:
    """计算OCR修正值"""
    valid = get_valid_with_positions(str_list, is_digit)
    if not valid: return None

    valid_vals = [v[0] for v in valid]
    trend = get_sequence_trend(valid_vals)

    if trend == "positive":
        adjusted = valid[-1][0]
        print(f"📈 正趋势: {valid_vals} -> {adjusted}")
    elif trend == "negative":
        adjusted = valid[0][0]
        print(f"📉 负趋势: {valid_vals} -> {adjusted}")
    else:
        return None

    if not is_digit and adjusted:
        if not (1 <= adjusted <= len(LETTER_DICT)): return None
    return adjusted


def is_small_clusters(h_cluster, v_cluster, size_threshold=20) -> bool:
    h_sz, v_sz = len(h_cluster), len(v_cluster)
    is_small = h_sz < size_threshold and v_sz < size_threshold
    print(f"🔍 簇大小检查: H={h_sz}, V={v_sz} (阈值{size_threshold}) -> {'触发' if is_small else '未触发'}")
    return is_small


# ===================== 8. 流程封装 (Steps) =====================

def _step_1_ocr_process(image_path: str) -> Tuple[List, List]:
    boxes, texts = Run_onnx(image_path, 't')
    image_name = os.path.basename(image_path)

    rect_boxes = convert_quad_to_rect(boxes)
    write_boxes_to_json(rect_boxes, r"./BGA_bottom_DBNet_boxes.json", mode="w")

    print(f"\n{'=' * 50} 处理图像: {image_name} {'=' * 50}")

    boxes, texts = filter_boxes_by_aspect_ratio(boxes, texts)
    boxes, texts = filter_boxes_texts(boxes, texts)
    return boxes, texts


def _step_2_dynamic_clustering(boxes, texts, centers, min_len=3):
    print(f"\n[动态聚类] 初始检测 (min_len={min_len})")
    h_clusters = find_horizontal_clusters(boxes, texts, centers, min_len=min_len)
    v_clusters = find_vertical_clusters(boxes, texts, centers, min_len=min_len)

    h_exists = any(len(c) >= min_len for c in h_clusters)
    v_exists = any(len(c) >= min_len for c in v_clusters)

    if h_exists and not v_exists:
        v_clusters = find_vertical_clusters(boxes, texts, centers, min_len=2)
    elif v_exists and not h_exists:
        h_clusters = find_horizontal_clusters(boxes, texts, centers, min_len=2)

    return h_clusters, v_clusters


def _step_3_calc_xy_logic(image_path, h_sorted_text, v_sorted_text, h_cluster, v_cluster, boxes):
    print("\n=== 数值计算 ===")
    ocr_x, ocr_y = None, None
    X, Y = None, None

    # 1. OCR 计算
    if v_cluster and not h_cluster:
        if is_letter_list(v_sorted_text):
            ocr_y = calculate_adjusted_value(v_sorted_text, False)
        elif has_valid_digit_feature(v_sorted_text):
            ocr_y = calculate_adjusted_value(v_sorted_text, True)
    elif h_cluster and not v_cluster:
        if is_letter_list(h_sorted_text):
            ocr_x = calculate_adjusted_value(h_sorted_text, False)
        elif has_valid_digit_feature(h_sorted_text):
            ocr_x = calculate_adjusted_value(h_sorted_text, True)
    else:
        if is_letter_list(v_sorted_text):
            ocr_y = calculate_adjusted_value(v_sorted_text, False)
            if has_valid_digit_feature(h_sorted_text): ocr_x = calculate_adjusted_value(h_sorted_text, True)
        else:
            if has_valid_digit_feature(v_sorted_text): ocr_y = calculate_adjusted_value(v_sorted_text, True)
            if is_letter_list(h_sorted_text): ocr_x = calculate_adjusted_value(h_sorted_text, False)

        # 掩膜处理：双向有趋势则涂白
        if ocr_x is not None and ocr_y is not None:
            mask_clusters_in_image(image_path, boxes, list(set(h_cluster + v_cluster)))

    # 2. YOLO 触发逻辑
    trigger_detr = False
    if is_small_clusters(h_cluster, v_cluster, size_threshold=60):
        trigger_detr = True

    if trigger_detr:
        print("✅ 触发YOLO调用")
        detr_x, detr_y = detr_pin_XY(image_path)

        def _merge(o, d):
            if o is None: return d
            if d is None: return o
            return max(o, d)

        X = _merge(ocr_x, detr_x)
        Y = _merge(ocr_y, detr_y)
    else:
        X, Y = ocr_x, ocr_y

    # 3. 强制兜底
    if X is None or Y is None:
        print(f"\n⚠️ 结果不完整 (X={X}, Y={Y})，强制调用yolo...")
        X, Y = detr_pin_XY(image_path)

    return X, Y


def _step_4_generate_labels(X, Y, a1_corner):
    final_col, final_row = [], []
    if X is not None and Y is not None:
        base_cols = [str(i) for i in range(1, int(X) + 1)]
        tgt_y = int(Y)
        base_rows = LETTER_DICT[:tgt_y] if tgt_y <= len(LETTER_DICT) else LETTER_DICT[:]
        for i in range(len(LETTER_DICT), tgt_y):
            base_rows.append(f"Row{i + 1}")

        if a1_corner == "Top-Left":
            final_col, final_row = base_cols, base_rows
        elif a1_corner == "Bottom-Right":
            final_col, final_row = base_cols[::-1], base_rows[::-1]
        elif a1_corner == "Top-Right":
            final_col, final_row = base_cols[::-1], base_rows
        elif a1_corner == "Bottom-Left":
            final_col, final_row = base_cols, base_rows[::-1]

    return final_col, final_row


# ===================== 9. 主程序入口 =====================

def BGA_get_PIN(image_path: str, visualize: bool = False, min_cluster_len: int = 3) -> Tuple[
    List[str], List[str], Optional[int], Optional[int], List[str], List[str], str, int
]:
    """主调度函数
    返回: 水平文本, 垂直文本, X, Y, Col标签, Row标签, A1角, rotation(旋转角度)
    """
    rot = 0  # 新增: 默认旋转角

    # Step 1: OCR
    boxes, texts = _step_1_ocr_process(image_path)
    if not boxes:
        x, y = detr_pin_XY(image_path)
        return [], [], x, y, [], [], "Top-Left", rot

    # Step 2: Clustering
    centers = calculate_centers(boxes)
    h_clusters_all, v_clusters_all = _step_2_dynamic_clustering(boxes, texts, centers)

    # Step 3: Merging
    h_comb, _ = combine_close_clusters(h_clusters_all, centers, True)
    v_comb, _ = combine_close_clusters(v_clusters_all, centers, False)

    # Step 4: Check Rotation
    target_indices = list(set(h_comb + v_comb))
    rot = check_orientation_by_clusters(image_path, boxes, target_indices)
    if rot != 0:
        print(f"🚨 检测到需要旋转角度 {rot}°，建议外层旋转图片后重试")

    # Step 5: Visualization
    if visualize:
        print("🎨 生成可视化图表...")
        visualize_with_sorting(image_path, boxes, texts, h_comb, v_comb, centers, h_clusters_all, v_clusters_all)

    # Step 6: Sorting & Data Prep
    sorted_h = sort_cluster(h_comb, centers, True)
    sorted_v = sort_cluster(v_comb, centers, False)
    sorted_h_text = [texts[i] for i in sorted_h]
    sorted_v_text = [texts[i] for i in sorted_v]

    print(f"水平文本: {sorted_h_text}")
    print(f"垂直文本: {sorted_v_text}")

    # Step 7: Calc X, Y
    X, Y = _step_3_calc_xy_logic(image_path, sorted_h_text, sorted_v_text, h_comb, v_comb, boxes)
    print(f"\n最终结果: X={X}, Y={Y}")

    # Step 8: Labels
    a1_corner = determine_A1_corner(sorted_h_text, sorted_v_text)
    final_col, final_row = _step_4_generate_labels(X, Y, a1_corner)

    print("\n=== 最终封装参数 ===")
    print(f"A1: {a1_corner}, Cols: {final_col}, Rows: {final_row}")

    return sorted_h_text, sorted_v_text, X, Y, final_col, final_row, a1_corner, rot


if __name__ == "__main__":
    test_img = r"D:\workspace\PackageWizard1.1\Result/Package_view/page/bottom.jpg"
    try:
        h_text, v_text, X, Y, cols, rows, a1, rot = BGA_get_PIN(test_img, visualize=True)
        print(f"✅ 执行完成, rot={rot}")
    except Exception as e:
        print(f"❌ 执行出错: {e}")
