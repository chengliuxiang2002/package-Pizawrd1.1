import onnxruntime as rt
from typing import Optional, List, Dict, Tuple
import csv
import re
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from package_core.PackageExtract.BGA_Function.Pin_process.predict import process_single_image, \
    extract_pin_coords, extract_border_coords

try:
    from package_core.PackageExtract.yolox_onnx_py.model_paths import model_path, result_path
except ModuleNotFoundError:  # pragma: no cover - 兼容脚本直接运行
    from pathlib import Path
    def model_path(*parts):
        return str(Path(__file__).resolve().parents[3] / 'model' / Path(*parts))

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """计算两个边界框的交并比（IoU）"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def remove_overlapping_pins(pins: List[List[int]], iou_threshold: float = 0.5) -> List[List[int]]:
    """
    去除重叠的BGA_PIN（不考虑置信度，仅基于IOU去重）
    逻辑：遍历所有框，与后续框计算IOU，重叠超过阈值则去除后出现的框
    Args:
        pins: PIN坐标列表，格式 [[x1, y1, x2, y2], ...]（x1,y1=左上角，x2,y2=右下角）
        iou_threshold: IOU阈值（默认0.5，超过此值判定为重叠）
    Returns: 去重后的PIN坐标列表
    """
    if not pins:
        print("⚠️  输入PIN列表为空，直接返回")
        return []

    # 标记是否保留每个框（默认全部保留）
    keep = [True] * len(pins)
    removed_count = 0

    # 遍历每个框，与后续框比较IOU
    for i in range(len(pins)):
        if not keep[i]:
            continue  # 跳过已标记为去除的框
        # 与当前框之后的所有框比较
        for j in range(i + 1, len(pins)):
            if not keep[j]:
                continue  # 跳过已标记为去除的框
            # 计算IOU
            iou = calculate_iou(pins[i], pins[j])
            if iou > iou_threshold:
                keep[j] = False  # 标记为去除（保留先出现的框）
                removed_count += 1
                print(f"检测到重叠PIN（IoU={iou:.3f}），去除后续重叠框")

    # 筛选出保留的框
    kept_pins = [pins[i] for i in range(len(pins)) if keep[i]]

    # 输出统计信息
    if removed_count > 0:
        print(f"重叠PIN处理完成：去除{removed_count}个，保留{len(kept_pins)}个")
    else:
        print("未检测到重叠PIN，全部保留")

    return kept_pins



def group_bga_by_rows(coordinates, y_threshold=5):
    """BGA按行分组（Y轴方向）"""
    if not coordinates:
        return []
    y_coords = [coord[1] for coord in coordinates]
    sorted_ys = sorted(y_coords)
    row_ys = []

    if sorted_ys:
        current_group = [sorted_ys[0]]
        for y in sorted_ys[1:]:
            if y - current_group[-1] <= y_threshold:
                current_group.append(y)
            else:
                row_ys.append(sum(current_group) / len(current_group))
                current_group = [y]
        row_ys.append(sum(current_group) / len(current_group))

    row_ys.sort()
    rows = [[] for _ in row_ys]
    for coord in coordinates:
        closest_row = min(range(len(row_ys)), key=lambda i: abs(row_ys[i] - coord[1]))
        rows[closest_row].append(coord)

    for row in rows:
        row.sort(key=lambda x: x[0])
    return rows


def group_bga_by_cols(coordinates, x_threshold=5):
    """BGA按列分组（X轴方向）"""
    if not coordinates:
        return []
    x_coords = [coord[0] for coord in coordinates]
    sorted_xs = sorted(x_coords)
    col_xs = []

    if sorted_xs:
        current_group = [sorted_xs[0]]
        for x in sorted_xs[1:]:
            if x - current_group[-1] <= x_threshold:
                current_group.append(x)
            else:
                col_xs.append(sum(current_group) / len(current_group))
                current_group = [x]
        col_xs.append(sum(current_group) / len(current_group))

    col_xs.sort()
    cols = [[] for _ in col_xs]
    for coord in coordinates:
        closest_col = min(range(len(col_xs)), key=lambda i: abs(col_xs[i] - coord[0]))
        cols[closest_col].append(coord)

    for col in cols:
        col.sort(key=lambda y: y[1])
    return cols


def calculate_average_height(coordinates):
    """计算BGA框平均高度"""
    if not coordinates:
        return 0
    heights = [coord[3] - coord[1] for coord in coordinates]
    return sum(heights) / len(heights)


def calculate_average_width(coordinates):
    """计算BGA框平均宽度"""
    if not coordinates:
        return 0
    widths = [coord[2] - coord[0] for coord in coordinates]
    return sum(widths) / len(widths)


def visualize_bga_rows(image_path, bga_rows):
    """可视化BGA行分组结果"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法找到图片: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(image)
    ax.set_title("BGA ball row grouping visualization")

    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    for row_idx, row in enumerate(bga_rows):
        color = colors[row_idx % 3]
        for coord in row:
            x1, y1, x2, y2 = coord
            rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f'R{row_idx + 1}', color=color, fontsize=8)

    ax.legend([f'Row {i + 1}' for i in range(len(bga_rows))], loc='upper right', bbox_to_anchor=(1.2, 1))
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def filter_valid_pins(pin_coords, border_coords):
    """
    筛选有效PIN（仅保留Border内部的PIN，外部为无效）
    核心特征：引脚位于元件本体（Border框内），外部无有效引脚
    :param pin_coords: 所有PIN坐标列表 [[x1,y1,x2,y2], ...]
    :param border_coords: Border坐标 [left, top, right, bottom]
    :return: 有效PIN坐标列表
    """
    left, top, right, bottom = border_coords
    valid_pins = []

    for pin in pin_coords:
        x1, y1, x2, y2 = pin
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 有效PIN判定：中心点在Border内部（包含边界）
        if left <= center_x <= right and top <= center_y <= bottom:
            valid_pins.append(pin)
    return valid_pins


# ===================== 核心优化：提取相邻PIN对（适配缺PIN场景） =====================
def get_bga_adjacent_pins(bga_rows, bga_cols) -> Tuple[List[List[int]], List[List[int]]]:
    """
    优化版：提取X/Y方向相邻PIN对，适配缺PIN场景（基于间距筛选，非固定索引）
    :param bga_rows: 按行分组的PIN列表
    :param bga_cols: 按列分组的PIN列表
    :return: (x_adjacent_pairs, y_adjacent_pairs)
    """

    # 辅助函数：计算两个PIN中心点的水平/垂直距离
    def get_center_distance(pin1, pin2, is_horizontal=True):
        c1_x = (pin1[0] + pin1[2]) / 2
        c1_y = (pin1[1] + pin1[3]) / 2
        c2_x = (pin2[0] + pin2[2]) / 2
        c2_y = (pin2[1] + pin2[3]) / 2
        return abs(c2_x - c1_x) if is_horizontal else abs(c2_y - c1_y)

    # 辅助函数：在目标列表中找间距最小的相邻PIN对
    def find_min_distance_pair(pin_list, is_horizontal=True):
        if len(pin_list) < 2:
            return []
        min_dist = float('inf')
        best_pair = []
        # 遍历所有连续PIN对，计算间距
        for i in range(len(pin_list) - 1):
            pin_a = pin_list[i]
            pin_b = pin_list[i + 1]
            dist = get_center_distance(pin_a, pin_b, is_horizontal)
            if dist < min_dist:
                min_dist = dist
                best_pair = [pin_a, pin_b]
        print(f"找到最小间距相邻对，间距：{min_dist:.2f}")
        return best_pair

    # ========== X方向：中间行 → 找最小水平间距对 ==========
    x_adjacent_pairs = []
    if len(bga_rows) >= 1:
        mid_row_idx = len(bga_rows) // 2
        target_row = bga_rows[mid_row_idx]
        # 关键：找目标行中水平间距最小的连续PIN对
        best_x_pair = find_min_distance_pair(target_row, is_horizontal=True)
        if best_x_pair:
            x_adjacent_pairs.append(best_x_pair)
            print(f"✅ 提取X方向相邻PIN对（第{mid_row_idx + 1}行，最小水平间距）")
        else:
            print(f"⚠️  第{mid_row_idx + 1}行无有效相邻PIN对")

    # ========== Y方向：中间列 → 找最小垂直间距对 ==========
    y_adjacent_pairs = []
    if len(bga_cols) >= 1:
        mid_col_idx = len(bga_cols) // 2
        target_col = bga_cols[mid_col_idx]
        # 关键：找目标列中垂直间距最小的连续PIN对
        best_y_pair = find_min_distance_pair(target_col, is_horizontal=False)
        if best_y_pair:
            y_adjacent_pairs.append(best_y_pair)
            print(f"✅ 提取Y方向相邻PIN对（第{mid_col_idx + 1}列，最小垂直间距）")
        else:
            print(f"⚠️  第{mid_col_idx + 1}列无有效相邻PIN对")

    return x_adjacent_pairs, y_adjacent_pairs


# ===================== 保存函数（保持不变） =====================
def save_bga_adjacent_pins(x_pairs: List[List[int]], y_pairs: List[List[int]], output_dir: str = None):
    if output_dir is None:
        output_path = result_path("Package_view", "pin", "BGA_adjacent_pins.txt")
    else:
        output_path = os.path.join(output_dir, "BGA_adjacent_pins.txt")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def fmt_pin(pin):
        return f"[{pin[0]:.2f}, {pin[1]:.2f}, {pin[2]:.2f}, {pin[3]:.2f}]"

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            if x_pairs and len(x_pairs[0]) == 2:
                pin1, pin2 = x_pairs[0]
                f.write(f"X: {fmt_pin(pin1)},{fmt_pin(pin2)}\n")
            else:
                f.write("X: None\n")

            if y_pairs and len(y_pairs[0]) == 2:
                pin1, pin2 = y_pairs[0]
                f.write(f"Y: {fmt_pin(pin1)},{fmt_pin(pin2)}\n")
            else:
                f.write("Y: None\n")

        print(f"✅ BGA相邻PIN对已保存至：{output_path}")
    except Exception as e:
        print(f"❌ 保存失败：{str(e)}")


# ===================== 新增：可视化相邻PIN对 =====================
def visualize_bga_adjacent_pins(image_path: str, pin_boxes: List[List[int]],
                               x_adjacent_pairs: List[List[int]], y_adjacent_pairs: List[List[int]]):
    """
    可视化BGA相邻PIN对（高亮X/Y方向选中的相邻PIN）
    :param image_path: 图片路径
    :param pin_boxes: 所有PIN坐标列表 [[x1,y1,x2,y2], ...]
    :param x_adjacent_pairs: X方向相邻PIN对列表 [[pin1, pin2], ...]
    :param y_adjacent_pairs: Y方向相邻PIN对列表 [[pin1, pin2], ...]
    """
    # 加载图片
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法加载图片：{image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 创建画布
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(image)
    ax.set_title("BGA Adjacent Pins Visualization (X:Red, Y:Blue)", fontsize=14, fontweight='bold')
    ax.axis('off')

    # 辅助函数：计算PIN中心点
    def get_pin_center(pin):
        x1, y1, x2, y2 = pin
        return (x1 + x2) / 2, (y1 + y2) / 2

    # 1. 绘制所有PIN（浅灰色，透明填充）
    for idx, pin in enumerate(pin_boxes):
        x1, y1, x2, y2 = pin
        # 基础PIN框（浅灰，透明）
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                        linewidth=1, edgecolor='#cccccc', facecolor='none', alpha=0.5)
        ax.add_patch(rect)
        # 标注PIN序号（小号字体）
        ax.text(x1, y1 - 3, f'P{idx+1}', fontsize=6, color='gray', ha='center')

    # 2. 绘制X方向相邻PIN对（红色高亮）
    if x_adjacent_pairs and len(x_adjacent_pairs[0]) == 2:
        pin1, pin2 = x_adjacent_pairs[0]
        # 绘制PIN1（红框+半透红填充）
        x1_1, y1_1, x2_1, y2_1 = pin1
        rect1 = Rectangle((x1_1, y1_1), x2_1 - x1_1, y2_1 - y1_1,
                         linewidth=3, edgecolor='red', facecolor='red', alpha=0.3)
        ax.add_patch(rect1)
        # 绘制PIN2（红框+半透红填充）
        x1_2, y1_2, x2_2, y2_2 = pin2
        rect2 = Rectangle((x1_2, y1_2), x2_2 - x1_2, y2_2 - y1_2,
                         linewidth=3, edgecolor='red', facecolor='red', alpha=0.3)
        ax.add_patch(rect2)
        # 绘制相邻连线（红色实线）
        c1_x, c1_y = get_pin_center(pin1)
        c2_x, c2_y = get_pin_center(pin2)
        ax.plot([c1_x, c2_x], [c1_y, c2_y], color='red', linewidth=2, marker='o', markersize=4, label='X Direction (Adjacent)')
        # 标注X方向
        mid_x = (c1_x + c2_x) / 2
        mid_y = (c1_y + c2_y) / 2 - 10
        ax.text(mid_x, mid_y, 'X →', fontsize=10, color='red', fontweight='bold', ha='center')

    # 3. 绘制Y方向相邻PIN对（蓝色高亮）
    if y_adjacent_pairs and len(y_adjacent_pairs[0]) == 2:
        pin1, pin2 = y_adjacent_pairs[0]
        # 绘制PIN1（蓝框+半透蓝填充）
        x1_1, y1_1, x2_1, y2_1 = pin1
        rect1 = Rectangle((x1_1, y1_1), x2_1 - x1_1, y2_1 - y1_1,
                         linewidth=3, edgecolor='blue', facecolor='blue', alpha=0.3)
        ax.add_patch(rect1)
        # 绘制PIN2（蓝框+半透蓝填充）
        x1_2, y1_2, x2_2, y2_2 = pin2
        rect2 = Rectangle((x1_2, y1_2), x2_2 - x1_2, y2_2 - y1_2,
                         linewidth=3, edgecolor='blue', facecolor='blue', alpha=0.3)
        ax.add_patch(rect2)
        # 绘制相邻连线（蓝色实线）
        c1_x, c1_y = get_pin_center(pin1)
        c2_x, c2_y = get_pin_center(pin2)
        ax.plot([c1_x, c2_x], [c1_y, c2_y], color='blue', linewidth=2, marker='o', markersize=4, label='Y Direction (Adjacent)')
        # 标注Y方向
        mid_x = (c1_x + c2_x) / 2 + 10
        mid_y = (c1_y + c2_y) / 2
        ax.text(mid_x, mid_y, 'Y ↓', fontsize=10, color='blue', fontweight='bold', ha='center')

    # 显示图例
    ax.legend(loc='upper right', fontsize=10)
    # 调整布局并显示
    plt.tight_layout()
    plt.show()


def detr_pin_XY(image_path):
    # 使用统一的路径管理加载模型
    ONNX_MODEL_PATH = model_path("yolo_model","pin_detect", "BGA_pin_detect.onnx")
    TARGET_CLASS_ID = 1  # 目标类别ID（默认1：BGA_PIN）
    # 基础配置参数（不变）
    std_h, std_w = 640, 640  # 标准输入尺寸
    conf_thres = 0.6  # 置信度阈值
    iou_thres = 0.3  # IOU阈值
    class_config = ['BGA_Border', 'BGA_PIN', 'BGA_serial_letter', 'BGA_serial_number']  # 类别配置

    # 加载ONNX模型（不变）
    try:
        sess = rt.InferenceSession(ONNX_MODEL_PATH)
        print(f"成功加载模型：{ONNX_MODEL_PATH}")
    except Exception as e:
        print(f"错误：无法加载模型 {ONNX_MODEL_PATH} - {str(e)}")
        exit(1)
    res = process_single_image(
        img_path=image_path,
        output_dir="",
        sess=sess,
        std_h=std_h,
        std_w=std_w,
        class_config=class_config,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        show_image=False
    )
    old_pin_boxes = extract_pin_coords(res)
    border = extract_border_coords(res)
    if border:
        old_pin_boxes = filter_valid_pins(old_pin_boxes, border)
    pin_boxes = remove_overlapping_pins(old_pin_boxes)
    X = None
    Y = None
    if pin_boxes:
        avg_height = calculate_average_height(pin_boxes)
        avg_width = calculate_average_width(pin_boxes)
        # print(f"平均高度：{avg_height:.2f}，平均宽度：{avg_width:.2f}")

        y_threshold = int(avg_height / 3)
        x_threshold = int(avg_width / 3)
        bga_rows = group_bga_by_rows(pin_boxes, y_threshold=y_threshold)
        bga_cols = group_bga_by_cols(pin_boxes, x_threshold=x_threshold)

        # 提取并保存相邻PIN对
        x_pairs, y_pairs = get_bga_adjacent_pins(bga_rows, bga_cols)
        save_bga_adjacent_pins(x_pairs, y_pairs)
        # visualize_bga_adjacent_pins(image_path, pin_boxes, x_pairs, y_pairs)

        X = len(bga_cols)
        Y = len(bga_rows)
        # visualize_bga_rows(image_path, bga_rows)
    return X,Y

# ==============================================================================
# 主函数（支持单张/批量处理切换）
# ==============================================================================
if __name__ == "__main__":
    # 单张图片处理（用于测试）
    image_path = r"D:\workspace\PackageWizard1.1\Result\Package_view\page\bottom.jpg"

    X, Y = detr_pin_XY(image_path=image_path)
    print(X,Y)
