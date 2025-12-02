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
    extract_pin_coords

try:
    from package_core.PackageExtract.yolox_onnx_py.model_paths import model_path
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


def detr_pin_XY(image_path):
    # 使用统一的路径管理加载模型
    ONNX_MODEL_PATH = model_path("yolo_model","pin_detect", "BGA_pin_detect.onnx")
    TARGET_CLASS_ID = 1  # 目标类别ID（默认1：BGA_PIN）
    # 基础配置参数（不变）
    std_h, std_w = 640, 640  # 标准输入尺寸
    conf_thres = 0.6  # 置信度阈值
    iou_thres = 0.5  # IOU阈值
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
        X = len(bga_cols)
        Y = len(bga_rows)
        # visualize_bga_rows(image_path, bga_rows)
    return X,Y

# ==============================================================================
# 主函数（支持单张/批量处理切换）
# ==============================================================================
if __name__ == "__main__":
    # 单张图片处理（用于测试）
    image_path = r"D:\workspace\PackageWizard1.1\Result\Package_extract\data\bottom.jpg"

    X, Y = detr_pin_XY(image_path=image_path)
    print(X,Y)
