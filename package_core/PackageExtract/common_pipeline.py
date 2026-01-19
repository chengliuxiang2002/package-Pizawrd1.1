"""封装提取流程中共用的辅助函数集合。"""

from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from package_core.PackageExtract.yolox_onnx_py.model_paths import result_path, model_path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# 全局路径 - 使用统一的路径管理函数
DATA = result_path('Package_extract', 'data')
DATA_BOTTOM_CROP = result_path('Package_extract', 'data_bottom_crop')
DATA_COPY = result_path('Package_extract', 'data_copy')
ONNX_OUTPUT = result_path('Package_extract', 'onnx_output')
OPENCV_OUTPUT = result_path('Package_extract', 'opencv_output')
OPENCV_OUTPUT_LINE = result_path('Package_extract', 'opencv_output_yinXian')
YOLO_DATA = result_path('Package_extract', 'yolox_data')
from package_core.PackageExtract.BGA_Function.DETR_BGA import DETR_BGA
from typing import Iterable, Tuple
import package_core.PackageExtract.get_pairs_data_present5_test as _pairs_module

from package_core.PackageExtract.function_tool import (
    empty_folder,
    find_list,
    recite_data,
    set_Image_size,
)
from package_core.PackageExtract.get_pairs_data_present5_test import *

# 默认需要处理的视图顺序，保持与原流程一致。
DEFAULT_VIEWS: Tuple[str, ...] = ("top", "bottom", "side", "detailed")


def prepare_workspace(
    data_dir: str,
    data_copy_dir: str,
    data_bottom_crop_dir: str,
    onnx_output_dir: str,
    opencv_output_dir: str,
    image_views: Iterable[str] = DEFAULT_VIEWS,
) -> None:
    """初始化提取流程所需的临时目录，并统一输入图片尺寸。

    该函数完整复刻了旧版 ``front_loading_work`` 的处理步骤：
    1. 清空上一次推理的中间产物目录；
    2. 遍历多个视图，确保图片尺寸符合推理要求；
    3. 将视图图像备份到 ``data_copy``，再还原到 ``data``，保证后续步骤在干净的副本上运行。
    """

    # 重置存放检测结果的临时目录。
    empty_folder(onnx_output_dir)
    os.makedirs(onnx_output_dir, exist_ok=True)

    empty_folder(data_bottom_crop_dir)
    os.makedirs(data_bottom_crop_dir, exist_ok=True)

    # 逐个视图调整图片尺寸，缺失图片时保留提示信息。
    for view_name in image_views:
        filein = os.path.join(data_dir, f"{view_name}.jpg")
        fileout = filein
        try:
            set_Image_size(filein, fileout)
        except Exception:
            print("文件", filein, "不存在")

    # 备份视图图片，保留当前状态。
    empty_folder(data_copy_dir)
    os.makedirs(data_copy_dir, exist_ok=True)
    if os.path.isdir(data_dir):
        for file_name in os.listdir(data_dir):
            shutil.copy(os.path.join(data_dir, file_name), os.path.join(data_copy_dir, file_name))

    # 清空 OpenCV 的输出目录。
    empty_folder(opencv_output_dir)
    os.makedirs(opencv_output_dir, exist_ok=True)

    # 使用备份重新构建 ``data`` 目录，确保后续步骤在一致的数据上运行。
    empty_folder(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    if os.path.isdir(data_copy_dir):
        for file_name in os.listdir(data_copy_dir):
            shutil.copy(os.path.join(data_copy_dir, file_name), os.path.join(data_dir, file_name))


def dbnet_get_text_box(img_path: str) -> np.ndarray:
    """运行 DBNet，获取指定图片的文本框坐标。"""

    location_cool = Run_onnx_det(img_path)
    dbnet_data = np.empty((len(location_cool), 4))  # [x1,x2,x3,x4]
    for i in range(len(location_cool)):
        dbnet_data[i][0] = min(location_cool[i][2], location_cool[i][0])
        dbnet_data[i][1] = min(location_cool[i][3], location_cool[i][1])
        dbnet_data[i][2] = max(location_cool[i][2], location_cool[i][0])
        dbnet_data[i][3] = max(location_cool[i][3], location_cool[i][1])

    dbnet_data = np.around(dbnet_data, decimals=2)
    return dbnet_data

from ultralytics import YOLO
import os
import cv2
import numpy as np


def bind_pairs(cls,bboxes):
   
    pairs_num = 0

    '''
    # 根据你的 YAML 顺序和最新要求整理如下：
    # 1: pairs_outside_row
    # 2: pairs_outside_col
    # 3: pairs_inside_row
    # 4: pairs_inside_col
    # 24: angle (你补充的)
    # 25: qfn_pairs_arrow
    # 26: qfn_pairs_inside_oblique
    # 29: pairs_inSide_thickness
    arrow_indices = [1, 2, 3, 4, 24, 25, 26, 29]
    '''


    for i in range(len(cls)):
        if cls[i] == 1 or cls[i] == 2 or cls[i] == 3 or cls[i] == 4 or cls[i] == 24 or cls[i] == 25 or cls[i] == 26 or cls[i] == 29:
            pairs_num += 1


        
    yolox_pairs = np.empty((pairs_num, 5))

    p = 0

    for i in range(len(cls)):
        if cls[i] == 1 or cls[i] == 2 or cls[i] == 3 or cls[i] == 4 or cls[i] == 24 or cls[i] == 25 or cls[i] == 26 or cls[i] == 29:
            yolox_pairs[p][0] = bboxes[i][0]
            yolox_pairs[p][1] = bboxes[i][1]
            yolox_pairs[p][2] = bboxes[i][2]
            yolox_pairs[p][3] = bboxes[i][3]
            if cls[i] == 1 or cls[i] == 2 or cls[i] == 25:
                yolox_pairs[p][4] = 0
            else:
                yolox_pairs[p][4] = 1
            p = p + 1
            
            
    return yolox_pairs
       

def yolo_find_pair(img_path,weight,CONF_THRESHOLD = 0.1):
    model = YOLO(weight)
    # 进行推理
    
    results = model.predict(
        source=img_path,
        conf=CONF_THRESHOLD,
        save=False,
    )
    # 读取图片
    img_ori = cv2.imread(img_path)

    # 正确提取检测结果
    if results and len(results) > 0:
        # 获取第一个结果（因为只处理单张图片）
        result = results[0]
        
        # 检查是否有检测到的目标
        if result.boxes is not None and len(result.boxes) > 0:
            # 提取边界框 (xyxy格式)
            boxes = result.boxes.xyxy.cpu().numpy()
            # 提取置信度
            scores = result.boxes.conf.cpu().numpy()
            # 提取类别索引
            cls_inds = result.boxes.cls.cpu().numpy().astype(int)
            
            final_boxes = boxes
            final_scores = scores
            final_cls_inds = cls_inds
            
            # # 可视化结果
            # origin_img = vis(img_ori, final_boxes, final_scores, final_cls_inds,
            #                 conf=0.1, class_names=VOC_CLASSES)
        else:
            # 没有检测到任何目标
            final_boxes = np.zeros((0, 4))
            final_scores = np.zeros(0)
            final_cls_inds = np.zeros(0)
            origin_img = img_ori
    else:
        # 没有结果
        final_boxes = np.zeros((0, 4))
        final_scores = np.zeros(0)
        final_cls_inds = np.zeros(0)
        origin_img = img_ori

    print("final_boxes", final_boxes)

    pairs = bind_pairs(np.array(final_cls_inds), np.array(final_boxes))  # 将yolox检测的pairs和data进行匹配输入到txt文本中
    
    
    # output_dir = "D:\\BaiduNetdiskDownload\\post0\\codepackage\\PackageWizard20250807\\Result\\Package_extract\\onnx_output"

    # # 确保输出目录存在
    # os.makedirs(output_dir, exist_ok=True)

    # # 基于原文件名生成输出路径
    # filename = os.path.basename(img_path)  # 获取原文件名
    # output_path = os.path.join(output_dir, filename)

    # # 保存图像
    # cv2.imwrite(output_path, origin_img)
    # print(f"图像已保存到: {output_path}")
    # '''
    # final_boxes:记录yolox检测的坐标位置np(, 4)[x1,y1,x2,y2]
    # final_cls_inds:记录每个yolox检测的种类np(, )[1,2,3,]
    # final_scores:记录yolox每个检测的分数np(, )[80.9,90.1,50.2,]
    # '''
    return pairs
def yolo_classify(img_path: str, package_classes: str):
    """调用 YOLO 系列检测器，返回图像元素的坐标信息。"""

    if package_classes == "BGA":
        # BGA 封装需要额外合并 DETR 结果，强化 PIN 及边框的检测质量。
        (
            _,
            yolox_num,
            yolox_serial_num,
            pin,
            other,
            pad,
            border,
            angle_pairs,
            BGA_serial_num,
            BGA_serial_letter,
        ) = begain_output_pairs_data_location(img_path, package_classes)
        # weight = "model\yolo_model\package_model\yolov13_arrow_pairs2.onnx"
        weight =model_path("yolo_model","package_model","yolov13_arrow_pairs2.onnx")
        yolox_pairs = yolo_find_pair(img_path,weight)
        (
            _,
            _,
            _,
            pin,
            _,
            _,
            border,
            _,
            BGA_serial_num,
            BGA_serial_letter,
        ) = DETR_BGA(img_path, package_classes)
        print("yolox_pairs", yolox_pairs)
        print("yolox_num", yolox_num)
        print("yolox_serial_num", yolox_serial_num)
        print("pin", pin)
        print("other", other)
        print("pad", pad)
        print("border", border)
        print("angle_pairs", angle_pairs)
        print("BGA_serial_num", BGA_serial_num)
        print("BGA_serial_letter", BGA_serial_letter)
    else:
        (
            _,
            yolox_num,
            yolox_serial_num,
            pin,
            other,
            pad,
            border,
            angle_pairs,
            BGA_serial_num,
            BGA_serial_letter,
        ) = begain_output_pairs_data_location(img_path, package_classes)
        weight = "model\yolo_model\package_model\yolov13_arrow_pairs2.onnx"
        yolox_pairs = yolo_find_pair(img_path,weight)

        yolox_pairs = np.around(yolox_pairs, decimals=2)
        yolox_num = np.around(yolox_num, decimals=2)
        angle_pairs = np.around(angle_pairs, decimals=2)

    return (
        yolox_pairs,
        yolox_num,
        yolox_serial_num,
        pin,
        other,
        pad,
        border,
        angle_pairs,
        BGA_serial_num,
        BGA_serial_letter,
    )


def _process_single_view(view: str, package_path: str, package_classes: str):
    """处理单个视图的检测任务（供并行执行）"""
    empty_data = np.empty((0, 4))
    img_path = package_path + '/' + f"{view}.jpg"
    print(f'具体图片路径{img_path}')

    if os.path.exists(img_path):
        dbnet_data = dbnet_get_text_box(img_path)
        (
            yolox_pairs,
            yolox_num,
            yolox_serial_num,
            pin,
            other,
            pad,
            border,
            angle_pairs,
            BGA_serial_num,
            BGA_serial_letter,
        ) = yolo_classify(img_path, package_classes)
        
        print(f'{view} yolo箭头数据:{yolox_pairs}')
    else:
        print(f"未找到视图 {view},返回空值")
        dbnet_data = empty_data
        yolox_pairs = empty_data
        yolox_num = empty_data
        yolox_serial_num = empty_data
        pin = empty_data
        other = empty_data
        pad = empty_data
        border = empty_data
        angle_pairs = empty_data
        BGA_serial_num = empty_data
        BGA_serial_letter = empty_data

    return view, {
        "dbnet_data": dbnet_data,
        "yolox_pairs": yolox_pairs,
        "yolox_num": yolox_num,
        "yolox_serial_num": yolox_serial_num,
        "pin": pin,
        "other": other,
        "pad": pad,
        "border": border,
        "angle_pairs": angle_pairs,
        "BGA_serial_num": BGA_serial_num,
        "BGA_serial_letter": BGA_serial_letter,
    }


def get_data_location_by_yolo_dbnet(
    package_path: str, package_classes: str, view_names: Iterable[str] = DEFAULT_VIEWS,
    parallel: bool = True, max_workers: int = 4
):
    """ 结合 YOLO 与 DBNet 的结果，汇总指定视图的检测数据。

    Args:
        package_path: 封装图片所在目录
        package_classes: 封装类型
        view_names: 视图名称列表
        parallel: 是否启用并行处理（默认启用）
        max_workers: 并行处理的最大线程数
    """

    L3 = []
    view_names_list = list(view_names)
    view_results = {}

    if parallel and len(view_names_list) > 1:
        # 并行处理多个视图（加速2-3倍）
        print(f"🚀 启用并行处理，{len(view_names_list)}个视图")
        with ThreadPoolExecutor(max_workers=min(max_workers, len(view_names_list))) as executor:
            futures = {
                executor.submit(_process_single_view, view, package_path, package_classes): view
                for view in view_names_list
            }
            for future in as_completed(futures):
                view, result = future.result()
                view_results[view] = result
    else:
        # 串行处理（保持向后兼容）
        for view in view_names_list:
            _, result = _process_single_view(view, package_path, package_classes)
            view_results[view] = result

    for view in view_names_list:
        results = view_results[view]
        for key in ("dbnet_data", "yolox_pairs", "yolox_num", "yolox_serial_num", "pin", "other", "pad", "border", "angle_pairs"):
            L3.append({"list_name": f"{view}_{key}", "list": results[key]})
        if view == "bottom":
            L3.append({"list_name": "bottom_BGA_serial_letter", "list": results["BGA_serial_letter"]})
            L3.append({"list_name": "bottom_BGA_serial_num", "list": results["BGA_serial_num"]})

    # 返回与旧流程一致的 L3 数据结构，方便直接替换原有实现。
    print(f'********:{L3}***********')
    return L3


def remove_other_annotations(L3):
    """F4.6：剔除 YOLO/DBNet 输出中的 OTHER 类型框。"""

    for view in ("top", "bottom", "side", "detailed"):
        yolox_key = f"{view}_yolox_num"
        dbnet_key = f"{view}_dbnet_data"
        other_key = f"{view}_other"

        yolox_num = find_list(L3, yolox_key)
        dbnet_data = find_list(L3, dbnet_key)
        other_data = find_list(L3, other_key)

        filtered_yolox = _pairs_module.delete_other(other_data, yolox_num)
        filtered_dbnet = _pairs_module.delete_other(other_data, dbnet_data)

        recite_data(L3, yolox_key, filtered_yolox)
        recite_data(L3, dbnet_key, filtered_dbnet)

    return L3


def enrich_pairs_with_lines(L3, image_root: str, test_mode: int):
    """F4.6：为尺寸线补齐对应的标尺界限。"""

    empty_data = np.empty((0, 13))
    for view in ("top", "bottom", "side", "detailed"):
        print(f'{view}方向为尺寸线补齐对应的标尺界限')
        yolox_pairs = find_list(L3, f"{view}_yolox_pairs")
        print(f'原先箭头数据:{yolox_pairs}')
        img_path = os.path.join(image_root, f"{view}.jpg")

        if os.path.exists(img_path):
            pairs_length = _pairs_module.find_pairs_length(img_path, yolox_pairs, test_mode)
        else:
            pairs_length = empty_data

        print(f'箭头数据:{pairs_length}')
        recite_data(L3, f"{view}_yolox_pairs_length", pairs_length)

    return L3


def preprocess_pairs_and_text(L3, key: int):
    """F4.7：整理尺寸线与文本，生成初始配对候选。"""

    top_yolox_pairs = find_list(L3, "top_yolox_pairs")
    bottom_yolox_pairs = find_list(L3, "bottom_yolox_pairs")
    side_yolox_pairs = find_list(L3, "side_yolox_pairs")
    detailed_yolox_pairs = find_list(L3, "detailed_yolox_pairs")
    top_dbnet_data = find_list(L3, "top_dbnet_data")
    bottom_dbnet_data = find_list(L3, "bottom_dbnet_data")
    side_dbnet_data = find_list(L3, "side_dbnet_data")
    detailed_dbnet_data = find_list(L3, "detailed_dbnet_data")

    (
        top_yolox_pairs,
        bottom_yolox_pairs,
        side_yolox_pairs,
        detailed_yolox_pairs,
        top_yolox_pairs_copy,
        bottom_yolox_pairs_copy,
        side_yolox_pairs_copy,
        detailed_yolox_pairs_copy,
        top_dbnet_data_all,
        bottom_dbnet_data_all,
    ) = _pairs_module.get_better_data_1(
        top_yolox_pairs,
        bottom_yolox_pairs,
        side_yolox_pairs,
        detailed_yolox_pairs,
        key,
        top_dbnet_data,
        bottom_dbnet_data,
        side_dbnet_data,
        detailed_dbnet_data,
    )

    recite_data(L3, "top_yolox_pairs", top_yolox_pairs)
    recite_data(L3, "bottom_yolox_pairs", bottom_yolox_pairs)
    recite_data(L3, "side_yolox_pairs", side_yolox_pairs)
    recite_data(L3, "detailed_yolox_pairs", detailed_yolox_pairs)
    recite_data(L3, "top_dbnet_data", top_dbnet_data)
    recite_data(L3, "bottom_dbnet_data", bottom_dbnet_data)
    recite_data(L3, "side_dbnet_data", side_dbnet_data)
    recite_data(L3, "detailed_dbnet_data", detailed_dbnet_data)
    recite_data(L3, "top_yolox_pairs_copy", top_yolox_pairs_copy)
    recite_data(L3, "bottom_yolox_pairs_copy", bottom_yolox_pairs_copy)
    recite_data(L3, "side_yolox_pairs_copy", side_yolox_pairs_copy)
    recite_data(L3, "detailed_yolox_pairs_copy", detailed_yolox_pairs_copy)
    recite_data(L3, "top_dbnet_data_all", top_dbnet_data_all)
    recite_data(L3, "bottom_dbnet_data_all", bottom_dbnet_data_all)

    return L3

def compute_overlap_ratio(box1, box2):
    """计算两个框的重叠面积与最小框面积的比例"""
    # 计算交集
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集面积
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算两个框各自的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算最小面积
    min_area = min(area1, area2)

    # 返回重叠面积与最小面积的比例
    return inter_area / min_area if min_area > 0 else 0


def merge_overlapping_boxes(boxes, ratio_threshold=0.5):
    """
    基于重叠面积与最小框面积的比例合并框 (Vectorized + Graph Theory)
    :param boxes: np.array or list, shape (N, 4) [x1, y1, x2, y2]
    :param ratio_threshold: float, 重叠阈值
    :return: np.array, 合并后的框
    """
    if len(boxes) == 0:
        return np.array([])

    boxes = np.array(boxes).astype(float)
    N = len(boxes)

    # 1. 向量化计算所有框的面积
    # area = (x2 - x1) * (y2 - y1)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # 2. 利用广播机制一次性计算所有框对的交集面积 (Intersection)
    # shape 变为 (N, N, 2)，分别对比左上角和右下角
    # lt = max(box_i_x1, box_j_x1), ...
    lt = np.maximum(boxes[:, None, :2], boxes[None, :, :2])
    rb = np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])

    # 交集宽高，负数置为0
    wh = np.maximum(rb - lt, 0)
    inter_areas = wh[:, :, 0] * wh[:, :, 1]

    # 3. 计算分母：题目逻辑是 "最小框面积" (Intersection / min(Area_A, Area_B))
    # 如果是标准的 IoU，分母则是 (Area_A + Area_B - Inter)
    min_areas = np.minimum(areas[:, None], areas[None, :])

    # 防止除以0
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = inter_areas / min_areas
        ratios[min_areas == 0] = 0

    # 4. 构建邻接矩阵：重叠率大于阈值则相连
    # 对角线置为 False (自己和自己不需要连，虽然连了也没事)
    adj_matrix = ratios > ratio_threshold
    np.fill_diagonal(adj_matrix, False)

    # 5. 查找连通分量 (核心优化点)
    # scipy 的 connected_components 极快
    graph = csr_matrix(adj_matrix)
    n_components, labels = connected_components(graph, directed=False)

    # 6. 根据连通分量合并框
    merged_boxes = []
    for i in range(n_components):
        # 获取属于当前分量的所有框的索引
        idxs = np.where(labels == i)[0]
        component_boxes = boxes[idxs]

        # 计算外接矩形
        min_xy = np.min(component_boxes[:, :2], axis=0)
        max_xy = np.max(component_boxes[:, 2:], axis=0)

        merged_boxes.append(np.concatenate([min_xy, max_xy]))

    return np.array(merged_boxes)

def run_svtr_ocr(L3):
    """F4.7：执行 SVTR OCR 推理，将文本候选加入 L3。"""

    top_dbnet_data_all = find_list(L3, "top_dbnet_data")
    bottom_dbnet_data_all = find_list(L3, "bottom_dbnet_data")
    side_dbnet_data = find_list(L3, "side_dbnet_data")
    detailed_dbnet_data = find_list(L3, "detailed_dbnet_data")

    top_dbnet_data_all = merge_overlapping_boxes(top_dbnet_data_all)
    bottom_dbnet_data_all = merge_overlapping_boxes(bottom_dbnet_data_all)
    side_dbnet_data = merge_overlapping_boxes(side_dbnet_data)
    detailed_dbnet_data = merge_overlapping_boxes(detailed_dbnet_data)

    _, _, top_ocr_data, bottom_ocr_data, side_ocr_data, detailed_ocr_data = _pairs_module.SVTR(
        top_dbnet_data_all,
        bottom_dbnet_data_all,
        side_dbnet_data,
        detailed_dbnet_data,
    )

    recite_data(L3, "top_ocr_data", top_ocr_data)
    recite_data(L3, "bottom_ocr_data", bottom_ocr_data)
    recite_data(L3, "side_ocr_data", side_ocr_data)
    recite_data(L3, "detailed_ocr_data", detailed_ocr_data)


    return L3


def normalize_ocr_candidates(L3, key: int):
    """F4.7：OCR 文本后处理，规整最大/中值/最小候选。"""

    top_dbnet_data = find_list(L3, "top_dbnet_data")
    bottom_dbnet_data = find_list(L3, "bottom_dbnet_data")
    side_dbnet_data = find_list(L3, "side_dbnet_data")
    detailed_dbnet_data = find_list(L3, "detailed_dbnet_data")
    top_ocr_data = find_list(L3, "top_ocr_data")
    bottom_ocr_data = find_list(L3, "bottom_ocr_data")
    side_ocr_data = find_list(L3, "side_ocr_data")
    detailed_ocr_data = find_list(L3, "detailed_ocr_data")
    top_yolox_num = find_list(L3, "top_yolox_num")
    bottom_yolox_num = find_list(L3, "bottom_yolox_num")
    side_yolox_num = find_list(L3, "side_yolox_num")
    detailed_yolox_num = find_list(L3, "detailed_yolox_num")

    (
        top_ocr_data,
        bottom_ocr_data,
        side_ocr_data,
        detailed_ocr_data,
    ) = _pairs_module.data_wrangling_optimized(
        key,
        top_dbnet_data,
        bottom_dbnet_data,
        side_dbnet_data,
        detailed_dbnet_data,
        top_ocr_data,
        bottom_ocr_data,
        side_ocr_data,
        detailed_ocr_data,
        top_yolox_num,
        bottom_yolox_num,
        side_yolox_num,
        detailed_yolox_num,
    )

    recite_data(L3, "top_ocr_data", top_ocr_data)
    recite_data(L3, "bottom_ocr_data", bottom_ocr_data)
    recite_data(L3, "side_ocr_data", side_ocr_data)
    recite_data(L3, "detailed_ocr_data", detailed_ocr_data)

    #1218新添加
    # run_and_save_resort_log2(
    #     top_ocr_data,
    #     bottom_ocr_data,
    #     side_ocr_data,
    #     detailed_ocr_data, )
    return L3


#############################QFN的side处理#########################
def extract_sorted_dimensions(side_ocr_data_list, side_yolox_num):
    """
    处理多个OCR数据，每个YOLO框可能对应不同的OCR数据
    
    参数:
    side_ocr_data_list: OCR检测数据列表
    side_yolox_num: YOLO检测框数据，维度为[n, 4]
    
    返回:
    side_A, side_A3, side_A1: 排序后的前3个max_medium_min数组（仅处理中间值<2的）
    """
    # 存储所有匹配的尺寸数组和对应的中间值
    matched_dimensions = []
    
    if side_yolox_num is None or len(side_yolox_num) == 0:
        return [0, 0, 0], [0, 0, 0], [0, 0, 0]
    
    for yolo_box in side_yolox_num:
        best_match = None
        best_match_score = float('inf')
        
        # 为每个YOLO框找到最匹配的OCR数据
        for ocr_data in side_ocr_data_list:
            ocr_location = ocr_data.get('location', None)
            if ocr_location is not None and len(ocr_location) == 4:
                # 计算两个框的距离（中心点距离）
                yolo_center = [(yolo_box[0] + yolo_box[2])/2, (yolo_box[1] + yolo_box[3])/2]
                ocr_center = [(ocr_location[0] + ocr_location[2])/2, (ocr_location[1] + ocr_location[3])/2]
                distance = np.sqrt((yolo_center[0] - ocr_center[0])**2 + (yolo_center[1] - ocr_center[1])**2)
                
                if distance < best_match_score:
                    best_match_score = distance
                    best_match = ocr_data
        
        # 如果找到匹配的OCR数据，提取其尺寸数组
        if best_match is not None and best_match_score < 10:  # 设置一个阈值
            dimensions = best_match.get('max_medium_min', [])
            if len(dimensions) == 3:
                middle_value = dimensions[1]
                # 只处理中间值小于2的尺寸数组
                if middle_value < 2:
                    matched_dimensions.append((dimensions, middle_value))
    
    # 初始化输出值
    side_A = [0, 0, 0]
    side_A3 = [0, 0, 0]
    side_A1 = [0, 0, 0]
    
    # 按中间值从大到小排序并返回前3个
    if matched_dimensions:
        # 按中间值从大到小排序
        sorted_dims = sorted(matched_dimensions, key=lambda x: x[1], reverse=True)
        
        # 只取前3个完整的尺寸数组
        top_dims = [dim_array for dim_array, _ in sorted_dims[:3]]
        
        # 如果不足3个，补充[0,0,0]
        while len(top_dims) < 3:
            top_dims.append([0, 0, 0])
        
        # 分配给输出变量
        if len(top_dims) > 2:
            # 长度大于2时，保持原逻辑
            side_A = list(top_dims[0])
            side_A1 = list(top_dims[2])
            side_A3 = list(top_dims[1])
        else:
            # 长度小于等于2时，按顺序填充
            side_A = list(top_dims[0]) if len(top_dims) > 0 else [0, 0, 0]
            side_A1 = list(top_dims[1]) if len(top_dims) > 1 else [0, 0, 0]
            side_A3 = [0, 0, 0]
    
    return side_A, side_A3, side_A1

def extract_side_A_A1_A3(L3):
    side_yolox_num = find_list(L3, "side_yolox_num")
    side_ocr_data = find_list(L3, "side_ocr_data")
    side_dbnet_data = find_list(L3, "side_dbnet_data")
    print("调试信息如下")
    print(f'side_ocr_data:{side_ocr_data}')
    print(f'side_dbnet_data:{side_dbnet_data}')
    side_A, side_A3, side_A1 = extract_sorted_dimensions(side_ocr_data,side_yolox_num)
    return side_A, side_A3, side_A1






def extract_top_dimensions(border, top_ocr_data_list, triple_factor, key):
    """
    从top视图提取尺寸数据，处理多个OCR数据元素
    
    参数:
    border: 边界框，格式为[[x1, y1, x2, y2]]
    top_ocr_data_list: OCR检测数据列表，每个元素包含location和max_medium_min
    triple_factor: 嵌套的视图数据
    key: 控制提取'top'还是'bottom'元素
    
    返回:
    top_D: 水平方向尺寸数组 [最大, 标准, 最小]
    top_E: 竖直方向尺寸数组 [最大, 标准, 最小]
    """
    
    def extract_top_elements(data):
        """递归提取view_name为'top'或'bottom'的元素"""
        top_elements = []
        
        if isinstance(data, dict):
            if(key == 0):
                if data.get('view_name') == 'top':
                    top_elements.append(data)
                for value in data.values():
                    if isinstance(value, (dict, list)):
                        top_elements.extend(extract_top_elements(value))
            else:
                if data.get('view_name') == 'bottom':
                    top_elements.append(data)
                for value in data.values():
                    if isinstance(value, (dict, list)):
                        top_elements.extend(extract_top_elements(value))
        elif isinstance(data, list):
            for item in data:
                top_elements.extend(extract_top_elements(item))
        
        return top_elements
    
    print("=== extract_top_dimensions 开始执行 ===")
    
    # 初始化输出值
    top_D = [0, 0, 0]
    top_E = [0, 0, 0]
    
    # 检查输入数据
    if not top_ocr_data_list or len(top_ocr_data_list) == 0:
        print("警告: top_ocr_data_list为空，返回默认值")
        return top_D, top_E
    
    print(f"收到 {len(top_ocr_data_list)} 个OCR数据")
    
    # 提取triple_factor中的所有top元素
    top_elements = extract_top_elements(triple_factor)
    
    print(f"找到 {len(top_elements)} 个top元素")
    
    if not top_elements:
        print("警告: 没有找到top元素，使用OCR数据中的标准值排序")
        # 如果没有top元素，从OCR数据中按标准值排序取最大的
        all_max_medium_min = []
        for ocr_data in top_ocr_data_list:
            max_medium_min = ocr_data.get('max_medium_min', [])
            if len(max_medium_min) == 3:
                all_max_medium_min.append(max_medium_min)
        
        if all_max_medium_min:
            print(f"从 {len(all_max_medium_min)} 个OCR数据中提取max_medium_min")
            # 按标准值(中间值)排序
            all_max_medium_min.sort(key=lambda x: x[1], reverse=True)
            top_D = all_max_medium_min[0].copy()
            top_E = all_max_medium_min[0].copy()
            print(f"使用标准值排序结果: top_D={top_D}, top_E={top_E}")
        else:
            print("没有找到有效的max_medium_min数据")
        
        return top_D, top_E
    
    # 将top元素分为两类：有arrow_pairs和没有arrow_pairs的
    top_with_arrow = []
    top_without_arrow = []
    
    for element in top_elements:
        if element.get('arrow_pairs') is not None:
            top_with_arrow.append(element)
        else:
            top_without_arrow.append(element)
    
    print(f"有arrow_pairs的top元素: {len(top_with_arrow)} 个")
    print(f"无arrow_pairs的top元素: {len(top_without_arrow)} 个")
    
    # 为每个OCR数据找到匹配的top元素，创建融合结构B
    all_b_elements = []
    
    print(f"开始匹配OCR数据和top元素...")
    matched_count = 0
    
    # 使用更宽松的匹配阈值
    position_tolerance = 2.0  # 位置容差从0.001放宽到2.0
    
    for ocr_data in top_ocr_data_list:
        ocr_location = ocr_data.get('location', None)
        max_medium_min = ocr_data.get('max_medium_min', [])
        
        if ocr_location is None or len(ocr_location) != 4:
            continue
        
        # 确保max_medium_min是列表格式
        if isinstance(max_medium_min, np.ndarray):
            max_medium_min = max_medium_min.tolist()
        
        # 优先匹配有arrow_pairs的元素
        matched = False
        matched_element = None
        
        # 首先尝试匹配有arrow_pairs的元素
        for top_element in top_with_arrow:
            element_location = top_element.get('location', None)
            if element_location is not None and len(element_location) == 4:
                # 使用放宽的阈值比较location
                if (abs(ocr_location[0] - element_location[0]) < position_tolerance and
                    abs(ocr_location[1] - element_location[1]) < position_tolerance and
                    abs(ocr_location[2] - element_location[2]) < position_tolerance and
                    abs(ocr_location[3] - element_location[3]) < position_tolerance):
                    
                    matched = True
                    matched_element = top_element
                    print(f"匹配成功(有箭头): OCR位置{ocr_location} 与 top位置{element_location}")
                    break
        
        # 如果没有匹配到有arrow_pairs的元素，再尝试匹配没有arrow_pairs的元素
        if not matched:
            for top_element in top_without_arrow:
                element_location = top_element.get('location', None)
                if element_location is not None and len(element_location) == 4:
                    # 使用放宽的阈值比较location
                    if (abs(ocr_location[0] - element_location[0]) < position_tolerance and
                        abs(ocr_location[1] - element_location[1]) < position_tolerance and
                        abs(ocr_location[2] - element_location[2]) < position_tolerance and
                        abs(ocr_location[3] - element_location[3]) < position_tolerance):
                        
                        matched = True
                        matched_element = top_element
                        print(f"匹配成功(无箭头): OCR位置{ocr_location} 与 top位置{element_location}")
                        break
        
        # 如果匹配成功，创建融合结构B
        if matched and matched_element is not None:
            b_element = {
                'location': matched_element['location'],
                'direction': matched_element.get('direction', ''),
                'arrow_pairs': matched_element.get('arrow_pairs', None),
                'max_medium_min': max_medium_min  # 使用OCR的max_medium_min
            }
            all_b_elements.append(b_element)
            matched_count += 1
            
            # 从原始列表中移除已匹配的元素，避免重复匹配
            if matched_element in top_with_arrow:
                top_with_arrow.remove(matched_element)
            elif matched_element in top_without_arrow:
                top_without_arrow.remove(matched_element)
    
    print(f"匹配完成，共找到 {matched_count} 个匹配项")
    
    if not all_b_elements:
        print("警告: 没有找到匹配的B元素，使用OCR数据中的标准值排序")
        # 如果没有匹配的B元素，从OCR数据中按标准值排序取最大的
        all_max_medium_min = []
        for ocr_data in top_ocr_data_list:
            max_medium_min = ocr_data.get('max_medium_min', [])
            if len(max_medium_min) == 3:
                all_max_medium_min.append(max_medium_min)
        
        if all_max_medium_min:
            print(f"从 {len(all_max_medium_min)} 个OCR数据中提取max_medium_min")
            # 按标准值(中间值)排序
            all_max_medium_min.sort(key=lambda x: x[1], reverse=True)
            top_D = all_max_medium_min[0].copy()
            top_E = all_max_medium_min[0].copy()
            print(f"使用标准值排序结果: top_D={top_D}, top_E={top_E}")
        else:
            print("没有找到有效的max_medium_min数据")
        
        return top_D, top_E
    
    # 计算border的长宽
    border_width = 0
    border_height = 0
    if border is not None and len(border) > 0:
        try:
            border_box = border[0]
            border_width = abs(float(border_box[2]) - float(border_box[0]))  # x2 - x1
            border_height = abs(float(border_box[3]) - float(border_box[1]))  # y2 - y1
            print(f"border尺寸: 宽度={border_width:.2f}, 高度={border_height:.2f}")
        except Exception as e:
            print(f"错误: 计算border尺寸时出错: {e}")
            border_width = 0
            border_height = 0
    else:
        print("警告: border为空或无效")
    
    # 按照标准值(中间值)对all_b_elements排序
    all_b_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0, reverse=True)
    print(f"按标准值排序后，前3个B元素的max_medium_min: {[b['max_medium_min'] for b in all_b_elements[:3]]}")
    
    # 如果没有border或border尺寸无效，使用标准值排序方法
    if border_width == 0 or border_height == 0:
        print("警告: border尺寸无效，使用标准值排序方法")
        # 分别收集水平和竖直方向的元素
        horizontal_elements = []
        vertical_elements = []
        
        for element in all_b_elements:
            direction = element.get('direction', '').lower()
            
            # 根据direction判断方向
            if direction in ['horizontal', 'up', 'down']:  # 水平方向：up和down
                horizontal_elements.append(element)
            elif direction in ['vertical', 'left', 'right']:  # 竖直方向：left和right
                vertical_elements.append(element)
            else:
                # 方向未知，两个方向都考虑
                horizontal_elements.append(element)
                vertical_elements.append(element)
        
        print(f"水平方向元素: {len(horizontal_elements)} 个")
        print(f"竖直方向元素: {len(vertical_elements)} 个")
        
        # 获取每个方向的最大标准值元素
        if horizontal_elements:
            horizontal_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0, reverse=True)
            top_D = horizontal_elements[0]['max_medium_min'].copy()
            print(f"水平方向选择: max_medium_min={top_D}")
        else:
            top_D = all_b_elements[0]['max_medium_min'].copy()
            print(f"水平方向无指定元素，使用第一个: max_medium_min={top_D}")
        
        if vertical_elements:
            vertical_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0, reverse=True)
            top_E = vertical_elements[0]['max_medium_min'].copy()
            print(f"竖直方向选择: max_medium_min={top_E}")
        else:
            top_E = all_b_elements[0]['max_medium_min'].copy()
            print(f"竖直方向无指定元素，使用第一个: max_medium_min={top_E}")
        
        return top_D, top_E
    
    # 有有效的border，进行比对
    print("开始与border尺寸进行比对...")
    best_horizontal_match = None
    best_vertical_match = None
    min_horizontal_diff = float('inf')
    min_vertical_diff = float('inf')
    
    # 优先考虑有arrow_pairs的元素进行border匹配
    for idx, element in enumerate(all_b_elements):
        direction = element.get('direction', '').lower()
        arrow_pairs = element.get('arrow_pairs', None)
        
        # 对于没有arrow_pairs的元素，跳过border匹配
        if arrow_pairs is None or len(arrow_pairs) == 0:
            continue
        
        # 获取最后一位（引线之间距离）
        try:
            arrow_distance = float(arrow_pairs[-1])
        except Exception as e:
            continue
        
        # 计算与border尺寸的差异
        horizontal_diff = abs(arrow_distance - border_width)
        vertical_diff = abs(arrow_distance - border_height)
        
        print(f"元素{idx}(有箭头): 方向={direction}, 箭头距离={arrow_distance:.2f}, "
              f"水平差异={horizontal_diff:.2f}, 垂直差异={vertical_diff:.2f}")
        
        # 根据direction确定主要方向
        if direction in ['horizontal', 'up', 'down']:  # 水平方向
            if horizontal_diff < min_horizontal_diff:
                min_horizontal_diff = horizontal_diff
                best_horizontal_match = element
                print(f"  更新水平最佳匹配: 差异={horizontal_diff:.2f}")
        elif direction in ['vertical', 'left', 'right']:  # 竖直方向
            if vertical_diff < min_vertical_diff:
                min_vertical_diff = vertical_diff
                best_vertical_match = element
                print(f"  更新竖直最佳匹配: 差异={vertical_diff:.2f}")
        else:
            # 方向未知，根据差异最小值决定方向
            if horizontal_diff < vertical_diff and horizontal_diff < min_horizontal_diff:
                min_horizontal_diff = horizontal_diff
                best_horizontal_match = element
                print(f"  更新水平最佳匹配(自动判断): 差异={horizontal_diff:.2f}")
            elif vertical_diff < horizontal_diff and vertical_diff < min_vertical_diff:
                min_vertical_diff = vertical_diff
                best_vertical_match = element
                print(f"  更新竖直最佳匹配(自动判断): 差异={vertical_diff:.2f}")
    
    # 如果通过有arrow_pairs的元素没有找到匹配，再考虑没有arrow_pairs的元素
    if best_horizontal_match is None or best_vertical_match is None:
        print("通过有arrow_pairs的元素未找到足够匹配，考虑无arrow_pairs的元素...")
        for idx, element in enumerate(all_b_elements):
            # 跳过已经有arrow_pairs的元素（已经处理过）
            if element.get('arrow_pairs') is not None:
                continue
                
            direction = element.get('direction', '').lower()
            max_medium_min = element.get('max_medium_min', [])
            
            if len(max_medium_min) < 2:
                continue
            
            std_value = max_medium_min[1]  # 标准值
            
            # 计算与border尺寸的差异
            horizontal_diff = abs(std_value - border_width)
            vertical_diff = abs(std_value - border_height)
            
            print(f"元素{idx}(无箭头): 方向={direction}, 标准值={std_value:.2f}, "
                  f"水平差异={horizontal_diff:.2f}, 垂直差异={vertical_diff:.2f}")
            
            # 根据direction确定主要方向
            if direction in ['horizontal', 'up', 'down']:  # 水平方向
                if horizontal_diff < min_horizontal_diff:
                    min_horizontal_diff = horizontal_diff
                    best_horizontal_match = element
                    print(f"  更新水平最佳匹配: 差异={horizontal_diff:.2f}")
            elif direction in ['vertical', 'left', 'right']:  # 竖直方向
                if vertical_diff < min_vertical_diff:
                    min_vertical_diff = vertical_diff
                    best_vertical_match = element
                    print(f"  更新竖直最佳匹配: 差异={vertical_diff:.2f}")
            else:
                # 方向未知，根据差异最小值决定方向
                if horizontal_diff < vertical_diff and horizontal_diff < min_horizontal_diff:
                    min_horizontal_diff = horizontal_diff
                    best_horizontal_match = element
                    print(f"  更新水平最佳匹配(自动判断): 差异={horizontal_diff:.2f}")
                elif vertical_diff < horizontal_diff and vertical_diff < min_vertical_diff:
                    min_vertical_diff = vertical_diff
                    best_vertical_match = element
                    print(f"  更新竖直最佳匹配(自动判断): 差异={vertical_diff:.2f}")
    
    # 使用阈值判断是否"很相似"
    similarity_threshold = 0.2  # 从10%放宽到20%的误差
    border_width_threshold = border_width * similarity_threshold
    border_height_threshold = border_height * similarity_threshold
    
    print(f"\n相似性阈值: 水平={border_width_threshold:.2f}, 竖直={border_height_threshold:.2f}")
    
    # 判断水平方向是否有匹配
    if best_horizontal_match is not None and min_horizontal_diff <= border_width_threshold:
        top_D = best_horizontal_match['max_medium_min'].copy()
        has_arrow = best_horizontal_match.get('arrow_pairs') is not None
        print(f"水平方向找到{'有箭头' if has_arrow else '无箭头'}相似匹配: max_medium_min={top_D}, 差异={min_horizontal_diff:.2f}")
    else:
        # 没有匹配，使用标准值排序
        print(f'水平无相似匹配, 最小差异={min_horizontal_diff:.2f}, 阈值={border_width_threshold:.2f}')
        # 从all_b_elements中按标准值排序，取最大的水平方向元素或第一个元素
        horizontal_elements = [e for e in all_b_elements 
                              if e.get('direction', '').lower() in ['horizontal', 'up', 'down']]
        if horizontal_elements:
            horizontal_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0, reverse=True)
            top_D = horizontal_elements[0]['max_medium_min'].copy()
            print(f"水平方向使用标准值排序: max_medium_min={top_D}")
        else:
            # 使用排序后第一个元素的max_medium_min
            top_D = all_b_elements[0]['max_medium_min'].copy()
            print(f"水平方向使用第一个元素: max_medium_min={top_D}")
    
    # 判断竖直方向是否有匹配
    if best_vertical_match is not None and min_vertical_diff <= border_height_threshold:
        top_E = best_vertical_match['max_medium_min'].copy()
        has_arrow = best_vertical_match.get('arrow_pairs') is not None
        print(f"竖直方向找到{'有箭头' if has_arrow else '无箭头'}相似匹配: max_medium_min={top_E}, 差异={min_vertical_diff:.2f}")
    else:
        # 没有匹配，使用标准值排序
        print(f'竖直无相似匹配, 最小差异={min_vertical_diff:.2f}, 阈值={border_height_threshold:.2f}')
        # 从all_b_elements中按标准值排序，取最大的竖直方向元素或第二个元素
        vertical_elements = [e for e in all_b_elements 
                            if e.get('direction', '').lower() in ['vertical', 'left', 'right']]
        if vertical_elements:
            vertical_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0, reverse=True)
            top_E = vertical_elements[0]['max_medium_min'].copy()
            print(f"竖直方向使用标准值排序: max_medium_min={top_E}")
        else:
            # 使用排序后第二个元素的max_medium_min（如果存在）
            if len(all_b_elements) > 1:
                top_E = all_b_elements[1]['max_medium_min'].copy()
                print(f"竖直方向使用第二个元素: max_medium_min={top_E}")
            else:
                # 如果只有一个元素，使用同一个元素的max_medium_min
                top_E = all_b_elements[0]['max_medium_min'].copy()
                print(f"竖直方向使用第一个元素: max_medium_min={top_E}")
    
    print(f"\n最终结果: top_D={top_D}, top_E={top_E}")
    print("=== extract_top_dimensions 执行结束 ===\n")
    
    return top_D, top_E


def extract_top_D_E(L3,triple_factor):
    top_ocr_data = find_list(L3, "top_ocr_data")
    bottom_ocr_data = find_list(L3, "bottom_ocr_data")
    top_border = find_list(L3, "top_border")
    bottom_border = find_list(L3, "bottom_border")
    top_dbnet_data = find_list(L3, "top_dbnet_data")
    print("调试信息如下")
    print(f'top_ocr_data:{top_ocr_data}')
    print(f'top_dbnet_data:{top_dbnet_data}')
    top_D, top_E = extract_top_dimensions(top_border,top_ocr_data,triple_factor,0)
    if(np.all(np.array(top_D) == 0) or np.all(np.array(top_E) == 0)):
        top_D, top_E = extract_top_dimensions(bottom_border,bottom_ocr_data,triple_factor,1)
    
    # if(top_D[1] > top_E[1]):
    #     top_D, top_E = top_E, top_D
    return top_D, top_E


def extract_bottom_dimensions(bottom_D, bottom_E, pad, bottom_ocr_data_list, triple_factor):
    """
    从bottom视图提取尺寸数据，处理多个OCR数据元素
    
    参数:
    bottom_D: 水平方向尺寸数组 [最大, 标准, 最小]
    bottom_E: 竖直方向尺寸数组 [最大, 标准, 最小]
    pad: 边界框，格式为[[x1, y1, x2, y2]]
    bottom_ocr_data_list: OCR检测数据列表，每个元素包含location和max_medium_min
    triple_factor: 嵌套的视图数据
    
    返回:
    bottom_D2: 水平方向尺寸数组 [最大, 标准, 最小]
    bottom_E2: 竖直方向尺寸数组 [最大, 标准, 最小]
    """
    
    def extract_bottom_elements(data):
        """递归提取view_name为'bottom'的元素"""
        bottom_elements = []
        
        if isinstance(data, dict):
            if data.get('view_name') == 'bottom':
                bottom_elements.append(data)
            for value in data.values():
                if isinstance(value, (dict, list)):
                    bottom_elements.extend(extract_bottom_elements(value))
        elif isinstance(data, list):
            for item in data:
                bottom_elements.extend(extract_bottom_elements(item))
        
        return bottom_elements
    
    print("=== extract_bottom_dimensions 开始执行 ===")
    
    # 初始化输出值
    bottom_D2 = [0, 0, 0]
    bottom_E2 = [0, 0, 0]
    
    # 检查pad是否存在
    if pad is None or len(pad) == 0:
        print("警告: pad为空，返回默认值[0,0,0]")
        return bottom_D2, bottom_E2
    
    print(f"输入参数: bottom_D={bottom_D}, bottom_E={bottom_E}")
    print(f"pad: {pad}")
    
    # 检查输入数据
    if not bottom_ocr_data_list or len(bottom_ocr_data_list) == 0:
        print("警告: bottom_ocr_data_list为空，返回默认值")
        return bottom_D2, bottom_E2
    
    print(f"收到 {len(bottom_ocr_data_list)} 个bottom OCR数据")
    
    # 提取triple_factor中的所有bottom元素
    bottom_elements = extract_bottom_elements(triple_factor)
    
    print(f"找到 {len(bottom_elements)} 个bottom元素")
    
    if not bottom_elements:
        print("警告: 没有找到bottom元素，返回默认值[0,0,0]")
        return bottom_D2, bottom_E2
    
    # 将bottom元素分为两类：有arrow_pairs和没有arrow_pairs的
    bottom_with_arrow = []
    bottom_without_arrow = []
    
    for element in bottom_elements:
        if element.get('arrow_pairs') is not None:
            bottom_with_arrow.append(element)
        else:
            bottom_without_arrow.append(element)
    
    print(f"有arrow_pairs的bottom元素: {len(bottom_with_arrow)} 个")
    print(f"无arrow_pairs的bottom元素: {len(bottom_without_arrow)} 个")
    
    # 为每个OCR数据找到匹配的bottom元素，创建融合结构B
    all_b_elements = []
    
    print(f"开始匹配OCR数据和bottom元素...")
    matched_count = 0
    
    # 使用更宽松的匹配阈值
    position_tolerance = 2.0  # 位置容差从0.001放宽到2.0
    
    for ocr_data in bottom_ocr_data_list:
        ocr_location = ocr_data.get('location', None)
        max_medium_min = ocr_data.get('max_medium_min', [])
        
        if ocr_location is None or len(ocr_location) != 4:
            continue
        
        # 确保max_medium_min是列表格式
        if isinstance(max_medium_min, np.ndarray):
            max_medium_min = max_medium_min.tolist()
        
        # 优先匹配有arrow_pairs的元素
        matched = False
        matched_element = None
        
        # 首先尝试匹配有arrow_pairs的元素
        for bottom_element in bottom_with_arrow:
            element_location = bottom_element.get('location', None)
            if element_location is not None and len(element_location) == 4:
                # 使用放宽的阈值比较location
                if (abs(ocr_location[0] - element_location[0]) < position_tolerance and
                    abs(ocr_location[1] - element_location[1]) < position_tolerance and
                    abs(ocr_location[2] - element_location[2]) < position_tolerance and
                    abs(ocr_location[3] - element_location[3]) < position_tolerance):
                    
                    matched = True
                    matched_element = bottom_element
                    print(f"匹配成功(有箭头): OCR位置{ocr_location} 与 bottom位置{element_location}")
                    break
        
        # 如果没有匹配到有arrow_pairs的元素，再尝试匹配没有arrow_pairs的元素
        if not matched:
            for bottom_element in bottom_without_arrow:
                element_location = bottom_element.get('location', None)
                if element_location is not None and len(element_location) == 4:
                    # 使用放宽的阈值比较location
                    if (abs(ocr_location[0] - element_location[0]) < position_tolerance and
                        abs(ocr_location[1] - element_location[1]) < position_tolerance and
                        abs(ocr_location[2] - element_location[2]) < position_tolerance and
                        abs(ocr_location[3] - element_location[3]) < position_tolerance):
                        
                        matched = True
                        matched_element = bottom_element
                        print(f"匹配成功(无箭头): OCR位置{ocr_location} 与 bottom位置{element_location}")
                        break
        
        # 如果匹配成功，创建融合结构B
        if matched and matched_element is not None:
            b_element = {
                'location': matched_element['location'],
                'direction': matched_element.get('direction', ''),
                'arrow_pairs': matched_element.get('arrow_pairs', None),
                'max_medium_min': max_medium_min  # 使用OCR的max_medium_min
            }
            all_b_elements.append(b_element)
            matched_count += 1
            
            # 从原始列表中移除已匹配的元素，避免重复匹配
            if matched_element in bottom_with_arrow:
                bottom_with_arrow.remove(matched_element)
            elif matched_element in bottom_without_arrow:
                bottom_without_arrow.remove(matched_element)
    
    print(f"匹配完成，共找到 {matched_count} 个匹配项")
    
    if not all_b_elements:
        print("警告: 没有找到匹配的B元素，返回默认值[0,0,0]")
        return bottom_D2, bottom_E2
    
    # 计算pad的长宽
    pad_width = 0
    pad_height = 0
    if pad is not None and len(pad) > 0:
        try:
            pad_box = pad[0]
            pad_width = abs(float(pad_box[2]) - float(pad_box[0]))  # x2 - x1
            pad_height = abs(float(pad_box[3]) - float(pad_box[1]))  # y2 - y1
            print(f"pad尺寸: 宽度={pad_width:.2f}, 高度={pad_height:.2f}")
        except Exception as e:
            print(f"错误: 计算pad尺寸时出错: {e}")
            pad_width = 0
            pad_height = 0
    
    # 按照标准值(中间值)对all_b_elements排序（降序）
    all_b_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0, reverse=True)
    print(f"按标准值排序后，所有B元素的max_medium_min: {[b['max_medium_min'] for b in all_b_elements]}")
    
    # 记录是否通过引线找到匹配
    horizontal_matched_by_arrow = False
    vertical_matched_by_arrow = False
    
    # 如果没有有效的pad尺寸，使用标准值排序方法
    if pad_width == 0 or pad_height == 0:
        print("警告: pad尺寸无效，使用标准值排序方法")
        # 分别收集水平和竖直方向的元素
        horizontal_elements = []
        vertical_elements = []
        
        for element in all_b_elements:
            direction = element.get('direction', '').lower()
            
            # 根据direction判断方向
            if direction in ['horizontal', 'up', 'down']:  # 水平方向：up和down
                horizontal_elements.append(element)
            elif direction in ['vertical', 'left', 'right']:  # 竖直方向：left和right
                vertical_elements.append(element)
            else:
                # 方向未知，两个方向都考虑
                horizontal_elements.append(element)
                vertical_elements.append(element)
        
        print(f"水平方向元素: {len(horizontal_elements)} 个")
        print(f"竖直方向元素: {len(vertical_elements)} 个")
        
        # 获取每个方向的最大标准值元素，但需要跳过与输入参数相同的值
        if horizontal_elements:
            horizontal_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0, reverse=True)
            # 寻找第一个与bottom_D不同的元素
            for element in horizontal_elements:
                candidate = element['max_medium_min'].copy()
                if not np.array_equal(candidate, bottom_D):
                    bottom_D2 = candidate
                    print(f"水平方向选择: max_medium_min={bottom_D2}")
                    break
            else:
                # 如果没有找到不同的元素，使用最大值
                bottom_D2 = horizontal_elements[0]['max_medium_min'].copy()
                print(f"水平方向所有元素都与bottom_D相同，使用最大值: max_medium_min={bottom_D2}")
        else:
            # 从所有元素中找与bottom_D不同的最大值
            for element in all_b_elements:
                candidate = element['max_medium_min'].copy()
                if not np.array_equal(candidate, bottom_D):
                    bottom_D2 = candidate
                    print(f"水平方向无指定元素，使用与bottom_D不同的第一个元素: max_medium_min={bottom_D2}")
                    break
            else:
                print("水平方向没有与bottom_D不同的元素，返回[0,0,0]")
                bottom_D2 = [0, 0, 0]
        
        if vertical_elements:
            vertical_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0, reverse=True)
            # 寻找第一个与bottom_E不同的元素
            for element in vertical_elements:
                candidate = element['max_medium_min'].copy()
                if not np.array_equal(candidate, bottom_E):
                    bottom_E2 = candidate
                    print(f"竖直方向选择: max_medium_min={bottom_E2}")
                    break
            else:
                # 如果没有找到不同的元素，使用最大值
                bottom_E2 = vertical_elements[0]['max_medium_min'].copy()
                print(f"竖直方向所有元素都与bottom_E相同，使用最大值: max_medium_min={bottom_E2}")
        else:
            # 从所有元素中找与bottom_E不同的最大值
            for element in all_b_elements:
                candidate = element['max_medium_min'].copy()
                if not np.array_equal(candidate, bottom_E):
                    bottom_E2 = candidate
                    print(f"竖直方向无指定元素，使用与bottom_E不同的第一个元素: max_medium_min={bottom_E2}")
                    break
            else:
                print("竖直方向没有与bottom_E不同的元素，返回[0,0,0]")
                bottom_E2 = [0, 0, 0]
        
        return bottom_D2, bottom_E2
    
    # 开始与pad尺寸进行比对
    print("开始与pad尺寸进行比对...")
    best_horizontal_match = None
    best_vertical_match = None
    min_horizontal_diff = float('inf')
    min_vertical_diff = float('inf')
    
    # 优先考虑有arrow_pairs的元素进行pad匹配
    for idx, element in enumerate(all_b_elements):
        direction = element.get('direction', '').lower()
        arrow_pairs = element.get('arrow_pairs', None)
        
        # 对于没有arrow_pairs的元素，先跳过
        if arrow_pairs is None or len(arrow_pairs) == 0:
            continue
        
        # 获取最后一位（引线之间距离）
        try:
            arrow_distance = float(arrow_pairs[-1])
        except Exception as e:
            continue
        
        # 计算与pad尺寸的差异
        horizontal_diff = abs(arrow_distance - pad_width)
        vertical_diff = abs(arrow_distance - pad_height)
        
        print(f"元素{idx}(有箭头): 方向={direction}, 箭头距离={arrow_distance:.2f}, "
              f"水平差异={horizontal_diff:.2f}, 垂直差异={vertical_diff:.2f}")
        
        # 根据direction确定主要方向
        if direction in ['horizontal', 'up', 'down']:  # 水平方向
            if horizontal_diff < min_horizontal_diff:
                min_horizontal_diff = horizontal_diff
                best_horizontal_match = element
                print(f"  更新水平最佳匹配: 差异={horizontal_diff:.2f}")
        elif direction in ['vertical', 'left', 'right']:  # 竖直方向
            if vertical_diff < min_vertical_diff:
                min_vertical_diff = vertical_diff
                best_vertical_match = element
                print(f"  更新竖直最佳匹配: 差异={vertical_diff:.2f}")
        else:
            # 方向未知，根据差异最小值决定方向
            if horizontal_diff < vertical_diff and horizontal_diff < min_horizontal_diff:
                min_horizontal_diff = horizontal_diff
                best_horizontal_match = element
                print(f"  更新水平最佳匹配(自动判断): 差异={horizontal_diff:.2f}")
            elif vertical_diff < horizontal_diff and vertical_diff < min_vertical_diff:
                min_vertical_diff = vertical_diff
                best_vertical_match = element
                print(f"  更新竖直最佳匹配(自动判断): 差异={vertical_diff:.2f}")
    
    # 如果通过有arrow_pairs的元素没有找到匹配，再考虑没有arrow_pairs的元素
    if best_horizontal_match is None or best_vertical_match is None:
        print("通过有arrow_pairs的元素未找到足够匹配，考虑无arrow_pairs的元素...")
        for idx, element in enumerate(all_b_elements):
            # 跳过已经有arrow_pairs的元素（已经处理过）
            if element.get('arrow_pairs') is not None:
                continue
                
            direction = element.get('direction', '').lower()
            max_medium_min = element.get('max_medium_min', [])
            
            if len(max_medium_min) < 2:
                continue
            
            std_value = max_medium_min[1]  # 标准值
            
            # 计算与pad尺寸的差异
            horizontal_diff = abs(std_value - pad_width)
            vertical_diff = abs(std_value - pad_height)
            
            print(f"元素{idx}(无箭头): 方向={direction}, 标准值={std_value:.2f}, "
                  f"水平差异={horizontal_diff:.2f}, 垂直差异={vertical_diff:.2f}")
            
            # 根据direction确定主要方向
            if direction in ['horizontal', 'up', 'down']:  # 水平方向
                if horizontal_diff < min_horizontal_diff:
                    min_horizontal_diff = horizontal_diff
                    best_horizontal_match = element
                    print(f"  更新水平最佳匹配: 差异={horizontal_diff:.2f}")
            elif direction in ['vertical', 'left', 'right']:  # 竖直方向
                if vertical_diff < min_vertical_diff:
                    min_vertical_diff = vertical_diff
                    best_vertical_match = element
                    print(f"  更新竖直最佳匹配: 差异={vertical_diff:.2f}")
            else:
                # 方向未知，根据差异最小值决定方向
                if horizontal_diff < vertical_diff and horizontal_diff < min_horizontal_diff:
                    min_horizontal_diff = horizontal_diff
                    best_horizontal_match = element
                    print(f"  更新水平最佳匹配(自动判断): 差异={horizontal_diff:.2f}")
                elif vertical_diff < horizontal_diff and vertical_diff < min_vertical_diff:
                    min_vertical_diff = vertical_diff
                    best_vertical_match = element
                    print(f"  更新竖直最佳匹配(自动判断): 差异={vertical_diff:.2f}")
    
    # 使用阈值判断是否"很相似"
    similarity_threshold = 0.2  # 从10%放宽到20%的误差
    pad_width_threshold = pad_width * similarity_threshold
    pad_height_threshold = pad_height * similarity_threshold
    
    print(f"\n相似性阈值: 水平={pad_width_threshold:.2f}, 竖直={pad_height_threshold:.2f}")
    
    # 判断水平方向是否有匹配
    if best_horizontal_match is not None and min_horizontal_diff <= pad_width_threshold:
        candidate = best_horizontal_match['max_medium_min'].copy()
        # 检查是否与bottom_D相同
        if not np.array_equal(candidate, bottom_D):
            bottom_D2 = candidate
            has_arrow = best_horizontal_match.get('arrow_pairs') is not None
            horizontal_matched_by_arrow = has_arrow  # 记录是否通过引线找到
            print(f"水平方向找到{'有箭头' if has_arrow else '无箭头'}相似匹配: max_medium_min={bottom_D2}, 差异={min_horizontal_diff:.2f}")
        else:
            print(f"水平方向找到相似匹配，但与bottom_D相同，跳过该匹配")
            # 继续寻找其他匹配
            best_horizontal_match = None
            horizontal_matched_by_arrow = False
    
    # 如果水平方向没有匹配或匹配值与bottom_D相同
    if best_horizontal_match is None or np.array_equal(bottom_D2, [0, 0, 0]):
        print(f'水平无有效相似匹配, 最小差异={min_horizontal_diff:.2f}, 阈值={pad_width_threshold:.2f}')
        # 从all_b_elements中按标准值排序，寻找与bottom_D不同的元素
        horizontal_elements = [e for e in all_b_elements 
                              if e.get('direction', '').lower() in ['horizontal', 'up', 'down']]
        if horizontal_elements:
            horizontal_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0, reverse=True)
            # 寻找第一个与bottom_D不同的元素
            for element in horizontal_elements:
                candidate = element['max_medium_min'].copy()
                if not np.array_equal(candidate, bottom_D):
                    bottom_D2 = candidate
                    print(f"水平方向使用标准值排序且与bottom_D不同的元素: max_medium_min={bottom_D2}")
                    break
            else:
                # 如果所有候选都与bottom_D相同，则从所有元素中找与bottom_D不同的元素
                print("水平方向所有候选都与bottom_D相同，从所有元素中寻找")
                for element in all_b_elements:
                    candidate = element['max_medium_min'].copy()
                    if not np.array_equal(candidate, bottom_D):
                        bottom_D2 = candidate
                        print(f"水平方向使用所有元素中与bottom_D不同的元素: max_medium_min={bottom_D2}")
                        break
                else:
                    print("水平方向所有元素都与bottom_D相同，返回[0,0,0]")
                    bottom_D2 = [0, 0, 0]
        else:
            # 从所有元素中寻找与bottom_D不同的元素
            for element in all_b_elements:
                candidate = element['max_medium_min'].copy()
                if not np.array_equal(candidate, bottom_D):
                    bottom_D2 = candidate
                    print(f"水平方向使用与bottom_D不同的第一个元素: max_medium_min={bottom_D2}")
                    break
            else:
                print("水平方向没有与bottom_D不同的元素，返回[0,0,0]")
                bottom_D2 = [0, 0, 0]
    
    # 判断竖直方向是否有匹配
    if best_vertical_match is not None and min_vertical_diff <= pad_height_threshold:
        candidate = best_vertical_match['max_medium_min'].copy()
        # 检查是否与bottom_E相同
        if not np.array_equal(candidate, bottom_E):
            bottom_E2 = candidate
            has_arrow = best_vertical_match.get('arrow_pairs') is not None
            vertical_matched_by_arrow = has_arrow  # 记录是否通过引线找到
            print(f"竖直方向找到{'有箭头' if has_arrow else '无箭头'}相似匹配: max_medium_min={bottom_E2}, 差异={min_vertical_diff:.2f}")
        else:
            print(f"竖直方向找到相似匹配，但与bottom_E相同，跳过该匹配")
            # 继续寻找其他匹配
            best_vertical_match = None
            vertical_matched_by_arrow = False
    
    # 如果竖直方向没有匹配或匹配值与bottom_E相同
    if best_vertical_match is None or np.array_equal(bottom_E2, [0, 0, 0]):
        print(f'竖直无有效相似匹配, 最小差异={min_vertical_diff:.2f}, 阈值={pad_height_threshold:.2f}')
        # 从all_b_elements中按标准值排序，寻找与bottom_E不同的元素
        vertical_elements = [e for e in all_b_elements 
                            if e.get('direction', '').lower() in ['vertical', 'left', 'right']]
        if vertical_elements:
            vertical_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0, reverse=True)
            # 寻找第一个与bottom_E不同的元素
            for element in vertical_elements:
                candidate = element['max_medium_min'].copy()
                if not np.array_equal(candidate, bottom_E):
                    bottom_E2 = candidate
                    print(f"竖直方向使用标准值排序且与bottom_E不同的元素: max_medium_min={bottom_E2}")
                    break
            else:
                # 如果所有候选都与bottom_E相同，则从所有元素中找与bottom_E不同的元素
                print("竖直方向所有候选都与bottom_E相同，从所有元素中寻找")
                for element in all_b_elements:
                    candidate = element['max_medium_min'].copy()
                    if not np.array_equal(candidate, bottom_E):
                        bottom_E2 = candidate
                        print(f"竖直方向使用所有元素中与bottom_E不同的元素: max_medium_min={bottom_E2}")
                        break
                else:
                    print("竖直方向所有元素都与bottom_E相同，返回[0,0,0]")
                    bottom_E2 = [0, 0, 0]
        else:
            # 从所有元素中寻找与bottom_E不同的元素
            for element in all_b_elements:
                candidate = element['max_medium_min'].copy()
                if not np.array_equal(candidate, bottom_E):
                    bottom_E2 = candidate
                    print(f"竖直方向使用与bottom_E不同的第一个元素: max_medium_min={bottom_E2}")
                    break
            else:
                print("竖直方向没有与bottom_E不同的元素，返回[0,0,0]")
                bottom_E2 = [0, 0, 0]
    
    # 应用新规则：如果一边通过引线找到匹配，另一边没有，则没有的一方使用找到引线一方的值
    print(f"\n匹配状态: 水平方向通过引线匹配={horizontal_matched_by_arrow}, 竖直方向通过引线匹配={vertical_matched_by_arrow}")
    
    if horizontal_matched_by_arrow and not vertical_matched_by_arrow:
        # 只有水平方向通过引线找到匹配，竖直方向没有
        if not np.array_equal(bottom_D2, [0, 0, 0]) and np.array_equal(bottom_E2, [0, 0, 0]):
            bottom_E2 = bottom_D2.copy()
            print(f"水平方向通过引线找到匹配，竖直方向没有，设置bottom_E2=bottom_D2: {bottom_E2}")
        elif not np.array_equal(bottom_D2, [0, 0, 0]) and not np.array_equal(bottom_E2, [0, 0, 0]):
            # 如果竖直方向已经有值，但水平方向是通过引线找到的，仍然使用水平方向的值
            print(f"水平方向通过引线找到匹配，竖直方向已有其他值，仍然使用水平方向的值")
            bottom_E2 = bottom_D2.copy()
    elif vertical_matched_by_arrow and not horizontal_matched_by_arrow:
        # 只有竖直方向通过引线找到匹配，水平方向没有
        if not np.array_equal(bottom_E2, [0, 0, 0]) and np.array_equal(bottom_D2, [0, 0, 0]):
            bottom_D2 = bottom_E2.copy()
            print(f"竖直方向通过引线找到匹配，水平方向没有，设置bottom_D2=bottom_E2: {bottom_D2}")
        elif not np.array_equal(bottom_E2, [0, 0, 0]) and not np.array_equal(bottom_D2, [0, 0, 0]):
            # 如果水平方向已经有值，但竖直方向是通过引线找到的，仍然使用竖直方向的值
            print(f"竖直方向通过引线找到匹配，水平方向已有其他值，仍然使用竖直方向的值")
            bottom_D2 = bottom_E2.copy()
    elif not horizontal_matched_by_arrow and not vertical_matched_by_arrow:
        print("水平和竖直方向都没有通过引线找到匹配，保持各自的排序结果")
    
    print(f"\n最终结果: bottom_D2={bottom_D2}, bottom_E2={bottom_E2}")
    print("=== extract_bottom_dimensions 执行结束 ===\n")
    
    return bottom_D2, bottom_E2

def extract_bottom_D2_E2(L3,triple_factor,bottom_D, bottom_E):
    bottom_ocr_data = find_list(L3, "bottom_ocr_data")
    bottom_pad = find_list(L3, "bottom_pad")
    bottom_dbnet_data = find_list(L3, "bottom_dbnet_data")
    print("调试信息如下")
    print(f'bottom_ocr_data:{bottom_ocr_data}')
    print(f'bottom_pad:{bottom_pad}')
    print(f'bottom_dbnet_data:{bottom_dbnet_data}')
    bottom_D2, bottom_E2 = extract_bottom_dimensions(bottom_D, bottom_E,bottom_pad,bottom_ocr_data,triple_factor)
    
    # if(bottom_D2[1] > bottom_E2[1]):
    #     bottom_D2, bottom_E2 = bottom_E2, bottom_D2
        
    return bottom_D2, bottom_E2
    


def extract_pin_boxes_from_txt(file_path):
    """
    从txt文件中提取引脚框数据
    
    Args:
        file_path: txt文件路径
        
    Returns:
        tuple: (pin_box, pin_boxh, pin_boxv)
    """
    # 初始化变量
    pin_boxh = []
    pin_boxv = []
    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
            for line in lines:
                line = line.strip()
                
                # 提取X数据
                if line.startswith('X:'):
                    # 去除'X: '前缀
                    x_data_str = line[2:].strip()
                    # 分割多个框
                    boxes_str = x_data_str.split('],[')
                    
                    for i, box_str in enumerate(boxes_str):
                        # 清理字符串中的括号和空格
                        box_str = box_str.replace('[', '').replace(']', '').strip()
                        # 如果是第一个框且开头有逗号，需要进一步清理
                        if box_str.startswith(','):
                            box_str = box_str[1:]
                        # 分割数字并转换为float
                        coordinates = [float(coord.strip()) for coord in box_str.split(',')]
                        # 添加到pin_boxh
                        pin_boxh.append(coordinates)
                
                # 提取Y数据
                elif line.startswith('Y:'):
                    # 去除'Y: '前缀
                    y_data_str = line[2:].strip()
                    # 分割多个框
                    boxes_str = y_data_str.split('],[')
                    
                    for box_str in boxes_str:
                        # 清理字符串中的括号和空格
                        box_str = box_str.replace('[', '').replace(']', '').strip()
                        # 如果是第一个框且开头有逗号，需要进一步清理
                        if box_str.startswith(','):
                            box_str = box_str[1:]
                        # 分割数字并转换为float
                        coordinates = [float(coord.strip()) for coord in box_str.split(',')]
                        # 添加到pin_boxv
                        pin_boxv.append(coordinates)
        
        # 从pin_boxh中提取第一个框作为pin_box
        if pin_boxh:
            pin_box = [pin_boxh[0]]  # 注意：格式化为列表的列表
        else:
            pin_box = []
            print("警告：X数据为空")
        
        return pin_box, pin_boxh, pin_boxv
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return [], [], []
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return [], [], []



def extract_pin_dimensions(pin_boxs, bottom_ocr_data_list, triple_factor):
    """
    从bottom视图提取与pin相关的尺寸数据
    
    参数:
    pin_boxs: pin角坐标，只有一个框[x1, y1, x2, y2]
    bottom_ocr_data_list: OCR检测数据列表，每个元素包含location和max_medium_min
    triple_factor: 嵌套的视图数据
    
    返回:
    bottom_b: 短边方向尺寸数组 [最大, 标准, 最小]
    bottom_L: 长边方向尺寸数组 [最大, 标准, 最小]
    """
    
    def extract_bottom_elements(data):
        """递归提取view_name为'bottom'的元素"""
        bottom_elements = []
        
        if isinstance(data, dict):
            if data.get('view_name') == 'bottom':
                bottom_elements.append(data)
            for value in data.values():
                if isinstance(value, (dict, list)):
                    bottom_elements.extend(extract_bottom_elements(value))
        elif isinstance(data, list):
            for item in data:
                bottom_elements.extend(extract_bottom_elements(item))
        
        return bottom_elements
    
    print("=== extract_pin_dimensions 开始执行 ===")
    
    # 初始化输出值
    bottom_b = [0, 0, 0]
    bottom_L = [0, 0, 0]
    
    # 检查输入数据
    if not bottom_ocr_data_list or len(bottom_ocr_data_list) == 0:
        print("警告: bottom_ocr_data_list为空，返回默认值")
        return bottom_b, bottom_L
    
    print(f"收到 {len(bottom_ocr_data_list)} 个bottom OCR数据")
    
    # 提取triple_factor中的所有bottom元素
    bottom_elements = extract_bottom_elements(triple_factor)
    
    print(f"找到 {len(bottom_elements)} 个bottom元素")
    
    if not bottom_elements:
        print("警告: 没有找到bottom元素，返回默认值")
        return bottom_b, bottom_L
    
    # 将bottom元素分为两类：有arrow_pairs和没有arrow_pairs的
    bottom_with_arrow = []
    bottom_without_arrow = []
    
    for element in bottom_elements:
        if element.get('arrow_pairs') is not None:
            bottom_with_arrow.append(element)
        else:
            bottom_without_arrow.append(element)
    
    print(f"有arrow_pairs的bottom元素: {len(bottom_with_arrow)} 个")
    print(f"无arrow_pairs的bottom元素: {len(bottom_without_arrow)} 个")
    
    # 为每个OCR数据找到匹配的bottom元素，创建融合结构B
    all_b_elements = []
    
    print(f"开始匹配OCR数据和bottom元素...")
    matched_count = 0
    
    # 使用更宽松的匹配阈值
    position_tolerance = 5.0  # 位置容差从0.001放宽到2.0
    
    for ocr_data in bottom_ocr_data_list:
        ocr_location = ocr_data.get('location', None)
        max_medium_min = ocr_data.get('max_medium_min', [])
        
        if ocr_location is None or len(ocr_location) != 4:
            continue
        
        # 确保max_medium_min是列表格式
        if isinstance(max_medium_min, np.ndarray):
            max_medium_min = max_medium_min.tolist()
        
        # 优先匹配有arrow_pairs的元素
        matched = False
        matched_element = None
        
        # 首先尝试匹配有arrow_pairs的元素
        for bottom_element in bottom_with_arrow:
            element_location = bottom_element.get('location', None)
            if element_location is not None and len(element_location) == 4:
                # 使用放宽的阈值比较location
                if (abs(ocr_location[0] - element_location[0]) < position_tolerance and
                    abs(ocr_location[1] - element_location[1]) < position_tolerance and
                    abs(ocr_location[2] - element_location[2]) < position_tolerance and
                    abs(ocr_location[3] - element_location[3]) < position_tolerance):
                    
                    matched = True
                    matched_element = bottom_element
                    print(f"匹配成功(有箭头): OCR位置{ocr_location} 与 bottom位置{element_location}")
                    break
        
        # 如果没有匹配到有arrow_pairs的元素，再尝试匹配没有arrow_pairs的元素
        if not matched:
            for bottom_element in bottom_without_arrow:
                element_location = bottom_element.get('location', None)
                if element_location is not None and len(element_location) == 4:
                    # 使用放宽的阈值比较location
                    if (abs(ocr_location[0] - element_location[0]) < position_tolerance and
                        abs(ocr_location[1] - element_location[1]) < position_tolerance and
                        abs(ocr_location[2] - element_location[2]) < position_tolerance and
                        abs(ocr_location[3] - element_location[3]) < position_tolerance):
                        
                        matched = True
                        matched_element = bottom_element
                        print(f"匹配成功(无箭头): OCR位置{ocr_location} 与 bottom位置{element_location}")
                        break
        
        # 如果匹配成功，创建融合结构B
        if matched and matched_element is not None:
            b_element = {
                'location': matched_element['location'],
                'direction': matched_element.get('direction', ''),
                'arrow_pairs': matched_element.get('arrow_pairs', None),
                'max_medium_min': max_medium_min  # 使用OCR的max_medium_min
            }
            all_b_elements.append(b_element)
            matched_count += 1
            
            # 从原始列表中移除已匹配的元素，避免重复匹配
            if matched_element in bottom_with_arrow:
                bottom_with_arrow.remove(matched_element)
            elif matched_element in bottom_without_arrow:
                bottom_without_arrow.remove(matched_element)
    
    print(f"匹配完成，共找到 {matched_count} 个匹配项")
    
    if not all_b_elements:
        print("警告: 没有找到匹配的B元素，返回默认值")
        return bottom_b, bottom_L
    
    # 按照标准值(中间值)对all_b_elements排序（升序）
    all_b_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0)
    print(f"按标准值排序后，所有B元素的max_medium_min: {[b['max_medium_min'] for b in all_b_elements]}")
    
    # 检查pin_boxs是否存在
    if pin_boxs is None or len(pin_boxs) == 0:
        print("警告: pin_boxs为空，使用标准值排序方法")
        # 使用排序后第一个元素的max_medium_min作为bottom_b
        if all_b_elements:
            bottom_b = all_b_elements[0]['max_medium_min'].copy()
            print(f"bottom_b使用第一个元素: max_medium_min={bottom_b}")
        
        # bottom_L使用最后一个元素的max_medium_min（如果存在且大于bottom_b），否则使用第二个
        if len(all_b_elements) >= 2:
            # 判断最后一个元素的标准值是否大于第一个元素
            last_std = all_b_elements[-1]['max_medium_min'][1] if len(all_b_elements[-1]['max_medium_min']) > 1 else 0
            first_std = all_b_elements[0]['max_medium_min'][1] if len(all_b_elements[0]['max_medium_min']) > 1 else 0
            
            if last_std > first_std:
                bottom_L = all_b_elements[-1]['max_medium_min'].copy()
                print(f"bottom_L使用最后一个元素: max_medium_min={bottom_L}")
            else:
                bottom_L = all_b_elements[1]['max_medium_min'].copy()
                print(f"bottom_L使用第二个元素: max_medium_min={bottom_L}")
        elif all_b_elements:
            bottom_L = all_b_elements[0]['max_medium_min'].copy()
            print(f"bottom_L只有一个元素可用，使用第一个元素: max_medium_min={bottom_L}")
        
        print(f"\n最终结果: bottom_b={bottom_b}, bottom_L={bottom_L}")
        print("=== extract_pin_dimensions 执行结束 ===\n")
        return bottom_b, bottom_L
    
    # 计算pin_boxs的尺寸（只有一个框）
    try:
        pin_box = pin_boxs[0] if isinstance(pin_boxs, list) else pin_boxs
        pin_width = abs(float(pin_box[2]) - float(pin_box[0]))  # x2 - x1
        pin_height = abs(float(pin_box[3]) - float(pin_box[1]))  # y2 - y1
        
        # 判断短边和长边
        if pin_width <= pin_height:
            # 宽度是短边，高度是长边
            pin_short = pin_width  # 短边
            pin_long = pin_height   # 长边
            print(f"pin_boxs尺寸: 宽度={pin_width:.2f}(短边), 高度={pin_height:.2f}(长边)")
        else:
            # 高度是短边，宽度是长边
            pin_short = pin_height  # 短边
            pin_long = pin_width     # 长边
            print(f"pin_boxs尺寸: 宽度={pin_width:.2f}(长边), 高度={pin_height:.2f}(短边)")
            
    except Exception as e:
        print(f"错误: 计算pin_boxs尺寸时出错: {e}")
        # 使用标准值排序方法
        if all_b_elements:
            bottom_b = all_b_elements[0]['max_medium_min'].copy()
            if len(all_b_elements) >= 2:
                bottom_L = all_b_elements[-1]['max_medium_min'].copy() if all_b_elements[-1]['max_medium_min'][1] > all_b_elements[0]['max_medium_min'][1] else all_b_elements[1]['max_medium_min'].copy()
            else:
                bottom_L = all_b_elements[0]['max_medium_min'].copy()
        
        return bottom_b, bottom_L
    
    # 开始与pin_boxs尺寸进行比对
    print("开始与pin_boxs尺寸进行比对...")
    best_short_match = None
    best_long_match = None
    min_short_diff = float('inf')
    min_long_diff = float('inf')
    
    # 优先选择有arrow_pairs的元素
    for idx, element in enumerate(all_b_elements):
        arrow_pairs = element.get('arrow_pairs', None)
        
        if arrow_pairs is None or len(arrow_pairs) == 0:
            continue  # 跳过没有arrow_pairs的元素
        
        # 获取最后一位（引线之间距离）
        try:
            arrow_distance = float(arrow_pairs[-1])
        except Exception as e:
            continue
        
        # 计算与短边和长边的差异
        short_diff = abs(arrow_distance - pin_short)
        long_diff = abs(arrow_distance - pin_long)
        
        print(f"元素{idx}(有箭头): 箭头距离={arrow_distance:.2f}, "
              f"与短边差异={short_diff:.2f}, 与长边差异={long_diff:.2f}")
        
        # 寻找与短边最相似的元素
        if short_diff < min_short_diff:
            min_short_diff = short_diff
            best_short_match = element
            print(f"  更新短边最佳匹配: 差异={short_diff:.2f}")
        
        # 寻找与长边最相似的元素
        if long_diff < min_long_diff:
            min_long_diff = long_diff
            best_long_match = element
            print(f"  更新长边最佳匹配: 差异={long_diff:.2f}")
    
    # 如果通过有arrow_pairs的元素没有找到匹配，再考虑没有arrow_pairs的元素
    if best_short_match is None or best_long_match is None:
        print("通过有arrow_pairs的元素未找到足够匹配，考虑无arrow_pairs的元素...")
        for idx, element in enumerate(all_b_elements):
            if element.get('arrow_pairs') is not None:
                continue  # 跳过已经有arrow_pairs的元素
            
            # 对于没有arrow_pairs的元素，使用max_medium_min的标准值进行匹配
            max_medium_min = element.get('max_medium_min', [])
            if len(max_medium_min) < 2:
                continue
            
            std_value = max_medium_min[1]  # 标准值
            
            # 计算与短边和长边的差异
            short_diff = abs(std_value - pin_short)
            long_diff = abs(std_value - pin_long)
            
            print(f"元素{idx}(无箭头): 标准值={std_value:.2f}, "
                  f"与短边差异={short_diff:.2f}, 与长边差异={long_diff:.2f}")
            
            # 寻找与短边最相似的元素
            if short_diff < min_short_diff:
                min_short_diff = short_diff
                best_short_match = element
                print(f"  更新短边最佳匹配: 差异={short_diff:.2f}")
            
            # 寻找与长边最相似的元素
            if long_diff < min_long_diff:
                min_long_diff = long_diff
                best_long_match = element
                print(f"  更新长边最佳匹配: 差异={long_diff:.2f}")
    
    # 使用阈值判断是否"很相似"
    similarity_threshold = 0.2  # 从10%放宽到20%的误差
    pin_short_threshold = pin_short * similarity_threshold
    pin_long_threshold = pin_long * similarity_threshold
    
    print(f"\n相似性阈值: 短边={pin_short_threshold:.2f}, 长边={pin_long_threshold:.2f}")
    
    # 记录是否通过引线找到匹配
    short_matched = False
    long_matched = False
    
    # 判断短边是否有匹配
    if best_short_match is not None and min_short_diff <= pin_short_threshold:
        bottom_b = best_short_match['max_medium_min'].copy()
        short_matched = True
        has_arrow = best_short_match.get('arrow_pairs') is not None
        print(f"短边找到{'有箭头' if has_arrow else '无箭头'}匹配: max_medium_min={bottom_b}, 差异={min_short_diff:.2f}")
    else:
        # 没有匹配，使用标准值排序取最小
        print(f'短边无相似匹配, 最小差异={min_short_diff:.2f}, 阈值={pin_short_threshold:.2f}')
        if all_b_elements:
            bottom_b = all_b_elements[0]['max_medium_min'].copy()
            print(f"短边使用标准值排序最小: max_medium_min={bottom_b}")
    
    # 判断长边是否有匹配
    if best_long_match is not None and min_long_diff <= pin_long_threshold:
        # 如果长边匹配的元素与短边匹配的元素相同，且短边已经匹配，则我们需要找另一个元素
        if best_long_match == best_short_match and short_matched:
            print("长边匹配的元素与短边相同，且短边已匹配，为长边寻找次佳匹配")
            # 在剩余元素中寻找与长边最相似的元素
            second_best_long_match = None
            second_min_long_diff = float('inf')
            
            for idx, element in enumerate(all_b_elements):
                if element == best_short_match:
                    continue  # 跳过已经被短边使用的元素
                    
                # 根据是否有arrow_pairs选择比较方式
                if element.get('arrow_pairs') is not None:
                    try:
                        arrow_distance = float(element['arrow_pairs'][-1])
                        long_diff = abs(arrow_distance - pin_long)
                    except:
                        continue
                else:
                    max_medium_min = element.get('max_medium_min', [])
                    if len(max_medium_min) < 2:
                        continue
                    long_diff = abs(max_medium_min[1] - pin_long)
                
                if long_diff < second_min_long_diff:
                    second_min_long_diff = long_diff
                    second_best_long_match = element
            
            # 检查次佳匹配是否满足阈值
            if second_best_long_match is not None and second_min_long_diff <= pin_long_threshold:
                bottom_L = second_best_long_match['max_medium_min'].copy()
                long_matched = True
                has_arrow = second_best_long_match.get('arrow_pairs') is not None
                print(f"长边找到{'有箭头' if has_arrow else '无箭头'}次佳匹配: max_medium_min={bottom_L}, 差异={second_min_long_diff:.2f}")
            else:
                # 没有次佳匹配，使用标准值排序
                print(f'长边无次佳相似匹配')
                if len(all_b_elements) >= 2:
                    # 使用排序后的最后一个元素（最大值）
                    bottom_L = all_b_elements[-1]['max_medium_min'].copy()
                    long_matched = False
                    print(f"长边使用标准值排序最大: max_medium_min={bottom_L}")
                elif all_b_elements:
                    bottom_L = all_b_elements[0]['max_medium_min'].copy()
                    long_matched = False
                    print(f"长边只有一个元素可用，使用第一个: max_medium_min={bottom_L}")
        else:
            bottom_L = best_long_match['max_medium_min'].copy()
            long_matched = True
            has_arrow = best_long_match.get('arrow_pairs') is not None
            print(f"长边找到{'有箭头' if has_arrow else '无箭头'}匹配: max_medium_min={bottom_L}, 差异={min_long_diff:.2f}")
    else:
        # 没有匹配，使用标准值排序
        print(f'长边无相似匹配, 最小差异={min_long_diff:.2f}, 阈值={pin_long_threshold:.2f}')
        if len(all_b_elements) >= 2:
            # 使用排序后的最后一个元素（最大值）
            bottom_L = all_b_elements[-1]['max_medium_min'].copy()
            long_matched = False
            print(f"长边使用标准值排序最大: max_medium_min={bottom_L}")
        elif all_b_elements:
            bottom_L = all_b_elements[0]['max_medium_min'].copy()
            long_matched = False
            print(f"长边只有一个元素可用，使用第一个: max_medium_min={bottom_L}")
    
    print(f"\n最终结果: bottom_b={bottom_b}, bottom_L={bottom_L}")
    print("=== extract_pin_dimensions 执行结束 ===\n")
    
    return bottom_b, bottom_L



def extract_bottom_b_L(L3,triple_factor,pin_boxs):
    bottom_ocr_data = find_list(L3, "bottom_ocr_data")
    bottom_dbnet_data = find_list(L3, "bottom_dbnet_data")
    print("调试信息如下")
    print(f'bottom_ocr_data:{bottom_ocr_data}')
    print(f'bottom_dbnet_data:{bottom_dbnet_data}')
    bottom_b, bottom_L = extract_pin_dimensions(pin_boxs,bottom_ocr_data,triple_factor)
    
    # if(bottom_D2[1] > bottom_E2[1]):
    #     bottom_D2, bottom_E2 = bottom_E2, bottom_D2
        
    return bottom_b, bottom_L








def extract_pitch_dimensions(pin_boxh, pin_boxv, bottom_ocr_data_list, triple_factor):
    """
    提取pitch_x和pitch_y尺寸数据
    
    参数:
    pin_boxh: 水平放置的pin角框列表，维度为[2,4]，表示2个框
    pin_boxv: 竖直放置的pin角框列表，维度为[2,4]，表示2个框
    bottom_ocr_data_list: OCR检测数据列表
    triple_factor: 嵌套的视图数据
    
    返回:
    pitch_x: 水平方向pitch尺寸数组 [最大, 标准, 最小]
    pitch_y: 竖直方向pitch尺寸数组 [最大, 标准, 最小]
    """
    
    def extract_bottom_elements(data):
        """递归提取view_name为'bottom'的元素"""
        bottom_elements = []
        
        if isinstance(data, dict):
            if data.get('view_name') == 'bottom':
                bottom_elements.append(data)
            for value in data.values():
                if isinstance(value, (dict, list)):
                    bottom_elements.extend(extract_bottom_elements(value))
        elif isinstance(data, list):
            for item in data:
                bottom_elements.extend(extract_bottom_elements(item))
        
        return bottom_elements
    
    print("=== extract_pitch_dimensions 开始执行 ===")
    
    # 初始化输出值
    pitch_x = [0, 0, 0]
    pitch_y = [0, 0, 0]
    
    # 记录是否通过相似引线找到匹配
    horizontal_matched_by_arrow = False
    vertical_matched_by_arrow = False
    
    # 检查输入数据
    if not bottom_ocr_data_list or len(bottom_ocr_data_list) == 0:
        print("警告: bottom_ocr_data_list为空，返回默认值")
        return pitch_x, pitch_y
    
    print(f"收到 {len(bottom_ocr_data_list)} 个bottom OCR数据")
    
    # 提取triple_factor中的所有bottom元素
    bottom_elements = extract_bottom_elements(triple_factor)
    
    print(f"找到 {len(bottom_elements)} 个bottom元素")
    
    if not bottom_elements:
        print("警告: 没有找到bottom元素，返回默认值")
        return pitch_x, pitch_y
    
    # 将bottom元素分为两类：有arrow_pairs和没有arrow_pairs的
    bottom_with_arrow = []
    bottom_without_arrow = []
    
    for element in bottom_elements:
        if element.get('arrow_pairs') is not None:
            bottom_with_arrow.append(element)
        else:
            bottom_without_arrow.append(element)
    
    print(f"有arrow_pairs的bottom元素: {len(bottom_with_arrow)} 个")
    print(f"无arrow_pairs的bottom元素: {len(bottom_without_arrow)} 个")
    
    # 为每个OCR数据找到匹配的bottom元素，创建融合结构B
    all_b_elements = []
    
    print(f"开始匹配OCR数据和bottom元素...")
    matched_count = 0
    
    # 使用更宽松的匹配阈值
    position_tolerance = 2.0  # 位置容差从0.001放宽到2.0
    
    for ocr_data in bottom_ocr_data_list:
        ocr_location = ocr_data.get('location', None)
        max_medium_min = ocr_data.get('max_medium_min', [])
        
        if ocr_location is None or len(ocr_location) != 4:
            continue
        
        # 确保max_medium_min是列表格式
        if isinstance(max_medium_min, np.ndarray):
            max_medium_min = max_medium_min.tolist()
        
        # 优先匹配有arrow_pairs的元素
        matched = False
        matched_element = None
        
        # 首先尝试匹配有arrow_pairs的元素
        for bottom_element in bottom_with_arrow:
            element_location = bottom_element.get('location', None)
            if element_location is not None and len(element_location) == 4:
                # 使用放宽的阈值比较location
                if (abs(ocr_location[0] - element_location[0]) < position_tolerance and
                    abs(ocr_location[1] - element_location[1]) < position_tolerance and
                    abs(ocr_location[2] - element_location[2]) < position_tolerance and
                    abs(ocr_location[3] - element_location[3]) < position_tolerance):
                    
                    matched = True
                    matched_element = bottom_element
                    print(f"匹配成功(有箭头): OCR位置{ocr_location} 与 bottom位置{element_location}")
                    break
        
        # 如果没有匹配到有arrow_pairs的元素，再尝试匹配没有arrow_pairs的元素
        if not matched:
            for bottom_element in bottom_without_arrow:
                element_location = bottom_element.get('location', None)
                if element_location is not None and len(element_location) == 4:
                    # 使用放宽的阈值比较location
                    if (abs(ocr_location[0] - element_location[0]) < position_tolerance and
                        abs(ocr_location[1] - element_location[1]) < position_tolerance and
                        abs(ocr_location[2] - element_location[2]) < position_tolerance and
                        abs(ocr_location[3] - element_location[3]) < position_tolerance):
                        
                        matched = True
                        matched_element = bottom_element
                        print(f"匹配成功(无箭头): OCR位置{ocr_location} 与 bottom位置{element_location}")
                        break
        
        # 如果匹配成功，创建融合结构B
        if matched and matched_element is not None:
            b_element = {
                'location': matched_element['location'],
                'direction': matched_element.get('direction', ''),
                'arrow_pairs': matched_element.get('arrow_pairs', None),
                'max_medium_min': max_medium_min  # 使用OCR的max_medium_min
            }
            all_b_elements.append(b_element)
            matched_count += 1
            
            # 从原始列表中移除已匹配的元素，避免重复匹配
            if matched_element in bottom_with_arrow:
                bottom_with_arrow.remove(matched_element)
            elif matched_element in bottom_without_arrow:
                bottom_without_arrow.remove(matched_element)
    
    print(f"匹配完成，共找到 {matched_count} 个匹配项")
    
    if not all_b_elements:
        print("警告: 没有找到匹配的B元素，返回默认值")
        return pitch_x, pitch_y
    
    # 按照标准值(中间值)对all_b_elements排序（升序）
    all_b_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0)
    print(f"按标准值排序后，所有B元素的max_medium_min: {[b['max_medium_min'] for b in all_b_elements]}")
    
    # 计算pin_h和pin_v
    pin_h = 0
    pin_v = 0
    has_pin_h = False
    has_pin_v = False
    
    # 处理pin_boxh
    if pin_boxh is not None:
        try:
            # 检查pin_boxh的格式
            box1 = None
            box2 = None
            
            if isinstance(pin_boxh, list) and len(pin_boxh) >= 2:
                if len(pin_boxh[0]) == 4 and len(pin_boxh[1]) == 4:
                    # 格式正确
                    box1 = pin_boxh[0]
                    box2 = pin_boxh[1]
                elif len(pin_boxh) == 8:
                    # 扁平化的列表
                    box1 = pin_boxh[:4]
                    box2 = pin_boxh[4:8]
            
            if box1 is not None and box2 is not None:
                # 计算第一个框的中心点
                center1_x = (box1[0] + box1[2]) / 2
                center1_y = (box1[1] + box1[3]) / 2
                
                # 计算第二个框的中心点
                center2_x = (box2[0] + box2[2]) / 2
                center2_y = (box2[1] + box2[3]) / 2
                
                # 计算中心点之间的距离
                pin_h = np.sqrt((center2_x - center1_x)**2 + (center2_y - center1_y)**2)
                has_pin_h = True
                print(f"pin_h (水平方向距离): {pin_h:.2f}")
            else:
                print(f"pin_boxh格式无法识别: {pin_boxh}")
        except Exception as e:
            print(f"错误: 计算pin_h时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 处理pin_boxv
    if pin_boxv is not None:
        try:
            # 检查pin_boxv的格式
            box1 = None
            box2 = None
            
            if isinstance(pin_boxv, list) and len(pin_boxv) >= 2:
                if len(pin_boxv[0]) == 4 and len(pin_boxv[1]) == 4:
                    # 格式正确
                    box1 = pin_boxv[0]
                    box2 = pin_boxv[1]
                elif len(pin_boxv) == 8:
                    # 扁平化的列表
                    box1 = pin_boxv[:4]
                    box2 = pin_boxv[4:8]
            
            if box1 is not None and box2 is not None:
                # 计算第一个框的中心点
                center1_x = (box1[0] + box1[2]) / 2
                center1_y = (box1[1] + box1[3]) / 2
                
                # 计算第二个框的中心点
                center2_x = (box2[0] + box2[2]) / 2
                center2_y = (box2[1] + box2[3]) / 2
                
                # 计算中心点之间的距离
                pin_v = np.sqrt((center2_x - center1_x)**2 + (center2_y - center1_y)**2)
                has_pin_v = True
                print(f"pin_v (竖直方向距离): {pin_v:.2f}")
            else:
                print(f"pin_boxv格式无法识别: {pin_boxv}")
        except Exception as e:
            print(f"错误: 计算pin_v时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 如果没有pin_boxh和pin_boxv，使用标准值排序方法
    if not has_pin_h and not has_pin_v:
        print("警告: pin_boxh和pin_boxv都无效，使用标准值排序方法")
        # 分别收集水平和竖直方向的元素
        horizontal_elements = []
        vertical_elements = []
        
        for element in all_b_elements:
            direction = element.get('direction', '').lower()
            
            # 根据direction判断方向
            if direction in ['horizontal', 'up', 'down']:  # 水平方向
                horizontal_elements.append(element)
            elif direction in ['vertical', 'left', 'right']:  # 竖直方向
                vertical_elements.append(element)
            else:
                # 方向未知，两个方向都考虑
                horizontal_elements.append(element)
                vertical_elements.append(element)
        
        print(f"水平方向元素: {len(horizontal_elements)} 个")
        print(f"竖直方向元素: {len(vertical_elements)} 个")
        
        # 获取水平方向次小的标准值元素
        if horizontal_elements:
            # 按标准值排序
            horizontal_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0)
            # 取次小的（如果有2个或以上元素）
            if len(horizontal_elements) >= 2:
                pitch_x = horizontal_elements[1]['max_medium_min'].copy()
                print(f"水平方向选择次小标准值: max_medium_min={pitch_x}")
            else:
                # 只有一个元素，使用该元素
                pitch_x = horizontal_elements[0]['max_medium_min'].copy()
                print(f"水平方向只有一个元素，使用该元素: max_medium_min={pitch_x}")
        else:
            # 没有水平方向元素，使用排序后的第二个元素（如果存在）
            if len(all_b_elements) >= 2:
                pitch_x = all_b_elements[1]['max_medium_min'].copy()
                print(f"水平方向无指定元素，使用排序后第二个元素: max_medium_min={pitch_x}")
            else:
                pitch_x = all_b_elements[0]['max_medium_min'].copy()
                print(f"水平方向无指定元素且元素不足，使用第一个元素: max_medium_min={pitch_x}")
        
        # 获取竖直方向次小的标准值元素
        if vertical_elements:
            # 按标准值排序
            vertical_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0)
            # 取次小的（如果有2个或以上元素）
            if len(vertical_elements) >= 2:
                pitch_y = vertical_elements[1]['max_medium_min'].copy()
                print(f"竖直方向选择次小标准值: max_medium_min={pitch_y}")
            else:
                # 只有一个元素，使用该元素
                pitch_y = vertical_elements[0]['max_medium_min'].copy()
                print(f"竖直方向只有一个元素，使用该元素: max_medium_min={pitch_y}")
        else:
            # 没有竖直方向元素，使用排序后的第二个元素（如果存在）
            if len(all_b_elements) >= 2:
                pitch_y = all_b_elements[1]['max_medium_min'].copy()
                print(f"竖直方向无指定元素，使用排序后第二个元素: max_medium_min={pitch_y}")
            else:
                pitch_y = all_b_elements[0]['max_medium_min'].copy()
                print(f"竖直方向无指定元素且元素不足，使用第一个元素: max_medium_min={pitch_y}")
        
        print(f"\n最终结果: pitch_x={pitch_x}, pitch_y={pitch_y}")
        print("=== extract_pitch_dimensions 执行结束 ===\n")
        return pitch_x, pitch_y
    
    # 开始与pin_h和pin_v进行比对
    print("开始与pin_h和pin_v进行比对...")
    best_horizontal_match = None
    best_vertical_match = None
    min_horizontal_diff = float('inf')
    min_vertical_diff = float('inf')
    
    # 优先考虑有arrow_pairs的元素进行匹配
    for idx, element in enumerate(all_b_elements):
        direction = element.get('direction', '').lower()
        arrow_pairs = element.get('arrow_pairs', None)
        
        # 对于没有arrow_pairs的元素，先跳过
        if arrow_pairs is None or len(arrow_pairs) == 0:
            continue
        
        # 获取最后一位（引线之间距离）
        try:
            arrow_distance = float(arrow_pairs[-1])
        except Exception as e:
            continue
        
        # 计算与pin_h和pin_v的差异
        horizontal_diff = abs(arrow_distance - pin_h) if has_pin_h else float('inf')
        vertical_diff = abs(arrow_distance - pin_v) if has_pin_v else float('inf')
        
        # 修复：安全地格式化输出
        if has_pin_h and horizontal_diff != float('inf'):
            horizontal_diff_str = f"{horizontal_diff:.2f}"
        else:
            horizontal_diff_str = "N/A"
            
        if has_pin_v and vertical_diff != float('inf'):
            vertical_diff_str = f"{vertical_diff:.2f}"
        else:
            vertical_diff_str = "N/A"
        
        print(f"元素{idx}(有箭头): 方向={direction}, 箭头距离={arrow_distance:.2f}, "
              f"与pin_h差异={horizontal_diff_str}, 与pin_v差异={vertical_diff_str}")
        
        # 根据direction确定主要方向
        if direction in ['horizontal', 'up', 'down']:  # 水平方向
            if has_pin_h and horizontal_diff < min_horizontal_diff:
                min_horizontal_diff = horizontal_diff
                best_horizontal_match = element
                print(f"  更新水平最佳匹配: 差异={horizontal_diff:.2f}")
        elif direction in ['vertical', 'left', 'right']:  # 竖直方向
            if has_pin_v and vertical_diff < min_vertical_diff:
                min_vertical_diff = vertical_diff
                best_vertical_match = element
                print(f"  更新竖直最佳匹配: 差异={vertical_diff:.2f}")
        else:
            # 方向未知，根据差异最小值决定方向
            if has_pin_h and has_pin_v:
                if horizontal_diff < vertical_diff and horizontal_diff < min_horizontal_diff:
                    min_horizontal_diff = horizontal_diff
                    best_horizontal_match = element
                    print(f"  更新水平最佳匹配(自动判断): 差异={horizontal_diff:.2f}")
                elif vertical_diff < horizontal_diff and vertical_diff < min_vertical_diff:
                    min_vertical_diff = vertical_diff
                    best_vertical_match = element
                    print(f"  更新竖直最佳匹配(自动判断): 差异={vertical_diff:.2f}")
    
    # 如果通过有arrow_pairs的元素没有找到匹配，再考虑没有arrow_pairs的元素
    if (has_pin_h and best_horizontal_match is None) or (has_pin_v and best_vertical_match is None):
        print("通过有arrow_pairs的元素未找到足够匹配，考虑无arrow_pairs的元素...")
        for idx, element in enumerate(all_b_elements):
            # 跳过已经有arrow_pairs的元素（已经处理过）
            if element.get('arrow_pairs') is not None:
                continue
                
            direction = element.get('direction', '').lower()
            max_medium_min = element.get('max_medium_min', [])
            
            if len(max_medium_min) < 2:
                continue
            
            std_value = max_medium_min[1]  # 标准值
            
            # 计算与pin_h和pin_v的差异
            horizontal_diff = abs(std_value - pin_h) if has_pin_h else float('inf')
            vertical_diff = abs(std_value - pin_v) if has_pin_v else float('inf')
            
            # 修复：安全地格式化输出
            if has_pin_h and horizontal_diff != float('inf'):
                horizontal_diff_str = f"{horizontal_diff:.2f}"
            else:
                horizontal_diff_str = "N/A"
                
            if has_pin_v and vertical_diff != float('inf'):
                vertical_diff_str = f"{vertical_diff:.2f}"
            else:
                vertical_diff_str = "N/A"
            
            print(f"元素{idx}(无箭头): 方向={direction}, 标准值={std_value:.2f}, "
                  f"与pin_h差异={horizontal_diff_str}, 与pin_v差异={vertical_diff_str}")
            
            # 根据direction确定主要方向
            if direction in ['horizontal', 'up', 'down']:  # 水平方向
                if has_pin_h and horizontal_diff < min_horizontal_diff:
                    min_horizontal_diff = horizontal_diff
                    best_horizontal_match = element
                    print(f"  更新水平最佳匹配: 差异={horizontal_diff:.2f}")
            elif direction in ['vertical', 'left', 'right']:  # 竖直方向
                if has_pin_v and vertical_diff < min_vertical_diff:
                    min_vertical_diff = vertical_diff
                    best_vertical_match = element
                    print(f"  更新竖直最佳匹配: 差异={vertical_diff:.2f}")
            else:
                # 方向未知，根据差异最小值决定方向
                if has_pin_h and has_pin_v:
                    if horizontal_diff < vertical_diff and horizontal_diff < min_horizontal_diff:
                        min_horizontal_diff = horizontal_diff
                        best_horizontal_match = element
                        print(f"  更新水平最佳匹配(自动判断): 差异={horizontal_diff:.2f}")
                    elif vertical_diff < horizontal_diff and vertical_diff < min_vertical_diff:
                        min_vertical_diff = vertical_diff
                        best_vertical_match = element
                        print(f"  更新竖直最佳匹配(自动判断): 差异={vertical_diff:.2f}")
    
    # 使用阈值判断是否"很相似"
    similarity_threshold = 0.2  # 从10%放宽到20%的误差
    
    # 判断水平方向是否有匹配
    if has_pin_h and best_horizontal_match is not None:
        pin_h_threshold = pin_h * similarity_threshold
        if min_horizontal_diff <= pin_h_threshold:
            pitch_x = best_horizontal_match['max_medium_min'].copy()
            horizontal_matched_by_arrow = True
            has_arrow = best_horizontal_match.get('arrow_pairs') is not None
            print(f"水平方向找到{'有箭头' if has_arrow else '无箭头'}相似匹配: max_medium_min={pitch_x}, 差异={min_horizontal_diff:.2f}, 阈值={pin_h_threshold:.2f}")
        else:
            # 没有匹配，使用标准值排序
            print(f'水平无相似匹配, 最小差异={min_horizontal_diff:.2f}, 阈值={pin_h_threshold:.2f}')
            # 从all_b_elements中按标准值排序，取次小的水平方向元素
            horizontal_elements = [e for e in all_b_elements 
                                  if e.get('direction', '').lower() in ['horizontal', 'up', 'down']]
            if horizontal_elements:
                horizontal_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0)
                # 取次小的（如果有2个或以上元素）
                if len(horizontal_elements) >= 2:
                    pitch_x = horizontal_elements[1]['max_medium_min'].copy()
                    print(f"水平方向使用标准值排序取次小: max_medium_min={pitch_x}")
                else:
                    # 只有一个元素，使用该元素
                    pitch_x = horizontal_elements[0]['max_medium_min'].copy()
                    print(f"水平方向只有一个元素，使用该元素: max_medium_min={pitch_x}")
            else:
                # 使用排序后的第二个元素（如果存在）
                if len(all_b_elements) >= 2:
                    pitch_x = all_b_elements[1]['max_medium_min'].copy()
                    print(f"水平方向无指定元素，使用排序后第二个元素: max_medium_min={pitch_x}")
                else:
                    pitch_x = all_b_elements[0]['max_medium_min'].copy()
                    print(f"水平方向无指定元素且元素不足，使用第一个元素: max_medium_min={pitch_x}")
    elif not has_pin_h:
        # 没有pin_h，使用标准值排序
        print("pin_h无效，使用标准值排序")
        horizontal_elements = [e for e in all_b_elements 
                              if e.get('direction', '').lower() in ['horizontal', 'up', 'down']]
        if horizontal_elements:
            horizontal_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0)
            if len(horizontal_elements) >= 2:
                pitch_x = horizontal_elements[1]['max_medium_min'].copy()
                print(f"水平方向使用标准值排序取次小: max_medium_min={pitch_x}")
            else:
                pitch_x = horizontal_elements[0]['max_medium_min'].copy()
                print(f"水平方向只有一个元素，使用该元素: max_medium_min={pitch_x}")
        else:
            if len(all_b_elements) >= 2:
                pitch_x = all_b_elements[1]['max_medium_min'].copy()
                print(f"水平方向无指定元素，使用排序后第二个元素: max_medium_min={pitch_x}")
            else:
                pitch_x = all_b_elements[0]['max_medium_min'].copy()
                print(f"水平方向无指定元素且元素不足，使用第一个元素: max_medium_min={pitch_x}")
    
    # 判断竖直方向是否有匹配
    if has_pin_v and best_vertical_match is not None:
        pin_v_threshold = pin_v * similarity_threshold
        if min_vertical_diff <= pin_v_threshold:
            pitch_y = best_vertical_match['max_medium_min'].copy()
            vertical_matched_by_arrow = True
            has_arrow = best_vertical_match.get('arrow_pairs') is not None
            print(f"竖直方向找到{'有箭头' if has_arrow else '无箭头'}相似匹配: max_medium_min={pitch_y}, 差异={min_vertical_diff:.2f}, 阈值={pin_v_threshold:.2f}")
        else:
            # 没有匹配，使用标准值排序
            print(f'竖直无相似匹配, 最小差异={min_vertical_diff:.2f}, 阈值={pin_v_threshold:.2f}')
            # 从all_b_elements中按标准值排序，取次小的竖直方向元素
            vertical_elements = [e for e in all_b_elements 
                                if e.get('direction', '').lower() in ['vertical', 'left', 'right']]
            if vertical_elements:
                vertical_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0)
                # 取次小的（如果有2个或以上元素）
                if len(vertical_elements) >= 2:
                    pitch_y = vertical_elements[1]['max_medium_min'].copy()
                    print(f"竖直方向使用标准值排序取次小: max_medium_min={pitch_y}")
                else:
                    # 只有一个元素，使用该元素
                    pitch_y = vertical_elements[0]['max_medium_min'].copy()
                    print(f"竖直方向只有一个元素，使用该元素: max_medium_min={pitch_y}")
            else:
                # 使用排序后的第二个元素（如果存在）
                if len(all_b_elements) >= 2:
                    pitch_y = all_b_elements[1]['max_medium_min'].copy()
                    print(f"竖直方向无指定元素，使用排序后第二个元素: max_medium_min={pitch_y}")
                else:
                    pitch_y = all_b_elements[0]['max_medium_min'].copy()
                    print(f"竖直方向无指定元素且元素不足，使用第一个元素: max_medium_min={pitch_y}")
    elif not has_pin_v:
        # 没有pin_v，使用标准值排序
        print("pin_v无效，使用标准值排序")
        vertical_elements = [e for e in all_b_elements 
                            if e.get('direction', '').lower() in ['vertical', 'left', 'right']]
        if vertical_elements:
            vertical_elements.sort(key=lambda x: x['max_medium_min'][1] if len(x['max_medium_min']) > 1 else 0)
            if len(vertical_elements) >= 2:
                pitch_y = vertical_elements[1]['max_medium_min'].copy()
                print(f"竖直方向使用标准值排序取次小: max_medium_min={pitch_y}")
            else:
                pitch_y = vertical_elements[0]['max_medium_min'].copy()
                print(f"竖直方向只有一个元素，使用该元素: max_medium_min={pitch_y}")
        else:
            if len(all_b_elements) >= 2:
                pitch_y = all_b_elements[1]['max_medium_min'].copy()
                print(f"竖直方向无指定元素，使用排序后第二个元素: max_medium_min={pitch_y}")
            else:
                pitch_y = all_b_elements[0]['max_medium_min'].copy()
                print(f"竖直方向无指定元素且元素不足，使用第一个元素: max_medium_min={pitch_y}")
    
    # 应用新规则：如果一个通过相似引线找到，另一个不是，则另一个与找到的那个相同
    print(f"\n匹配状态: 水平方向通过引线匹配={horizontal_matched_by_arrow}, 竖直方向通过引线匹配={vertical_matched_by_arrow}")
    
    if horizontal_matched_by_arrow and not vertical_matched_by_arrow:
        # 只有水平方向通过相似引线找到，竖直方向没有
        pitch_y = pitch_x.copy()
        print(f"水平方向通过引线找到匹配，竖直方向没有，设置pitch_y=pitch_x: {pitch_y}")
    elif vertical_matched_by_arrow and not horizontal_matched_by_arrow:
        # 只有竖直方向通过相似引线找到，水平方向没有
        pitch_x = pitch_y.copy()
        print(f"竖直方向通过引线找到匹配，水平方向没有，设置pitch_x=pitch_y: {pitch_x}")
    elif not horizontal_matched_by_arrow and not vertical_matched_by_arrow:
        print("水平和竖直方向都没有通过引线找到匹配，保持各自的排序结果")
    
    print(f"\n最终结果: pitch_x={pitch_x}, pitch_y={pitch_y}")
    print("=== extract_pitch_dimensions 执行结束 ===\n")
    
    return pitch_x, pitch_y





def extract_bottom_pitch_x_and_pitch_y(L3,triple_factor,pin_boxh, pin_boxv):
    bottom_ocr_data = find_list(L3, "bottom_ocr_data")
    bottom_dbnet_data = find_list(L3, "bottom_dbnet_data")
    print("调试信息如下")
    print(f'bottom_ocr_data:{bottom_ocr_data}')
    print(f'bottom_dbnet_data:{bottom_dbnet_data}')
    bottom_pitch_x, bottom_pitch_y = extract_pitch_dimensions(pin_boxh, pin_boxv,bottom_ocr_data,triple_factor)
    
    # if(bottom_D2[1] > bottom_E2[1]):
    #     bottom_D2, bottom_E2 = bottom_E2, bottom_D2
        
    return bottom_pitch_x, bottom_pitch_y

##############################################################











def extract_pin_serials(L3, package_classes: str):
    """F4.8：提取序号/PIN 相关信息，兼容 BGA/QFP 等封装。"""

    top_yolox_serial_num = find_list(L3, "top_yolox_serial_num")
    bottom_yolox_serial_num = find_list(L3, "bottom_yolox_serial_num")
    top_ocr_data = find_list(L3, "top_ocr_data")
    bottom_ocr_data = find_list(L3, "bottom_ocr_data")

    if package_classes in {"QFP", "QFN", "SOP", "SON"}:
        (
            top_serial_numbers_data,
            bottom_serial_numbers_data,
            top_ocr_data,
            bottom_ocr_data,
        ) = _pairs_module.find_PIN(
            top_yolox_serial_num,
            bottom_yolox_serial_num,
            top_ocr_data,
            bottom_ocr_data,
        )

        recite_data(L3, "top_serial_numbers_data", top_serial_numbers_data)
        recite_data(L3, "bottom_serial_numbers_data", bottom_serial_numbers_data)
        recite_data(L3, "top_ocr_data", top_ocr_data)
        recite_data(L3, "bottom_ocr_data", bottom_ocr_data)

    # if package_classes == "BGA":
    #     bottom_BGA_serial_number = find_list(L3, "bottom_BGA_serial_num")
    #     bottom_BGA_serial_letter = find_list(L3, "bottom_BGA_serial_letter")
    #
    #     (
    #         bottom_BGA_serial_number,
    #         bottom_BGA_serial_letter,
    #         bottom_ocr_data,
    #     ) = extract_BGA_PIN()
    #
    #     serial_numbers_data = np.empty((0, 4))
    #     for item in bottom_BGA_serial_number:
    #         mid = np.empty(5)
    #         mid[0:4] = item["location"].astype(str)
    #         mid[4] = item["key_info"][0]
    #         serial_numbers_data = np.r_[serial_numbers_data, [mid]]
    #
    #     serial_letters_data = np.empty((0, 4))
    #     for item in bottom_BGA_serial_letter:
    #         mid = np.empty(5)
    #         mid[0:4] = item["location"].astype(str)
    #         mid[4] = item["key_info"][0]
    #         serial_letters_data = np.r_[serial_letters_data, [mid]]
    #
    #     (
    #         pin_num_x_serial,
    #         pin_num_y_serial,
    #         pin_1_location,
    #     ) = _pairs_module.find_pin_num_pin_1(
    #         serial_numbers_data,
    #         serial_letters_data,
    #         bottom_BGA_serial_number,
    #         bottom_BGA_serial_letter,
    #     )
    #
    #     recite_data(L3, "bottom_BGA_serial_num", bottom_BGA_serial_number)
    #     recite_data(L3, "bottom_BGA_serial_letter", bottom_BGA_serial_letter)
    #     recite_data(L3, "bottom_ocr_data", bottom_ocr_data)
    #     recite_data(L3, "pin_num_x_serial", pin_num_x_serial)
    #     recite_data(L3, "pin_num_y_serial", pin_num_y_serial)
    #     recite_data(L3, "pin_1_location", pin_1_location)

    return L3


def match_pairs_with_text(L3, key: int):
    """F4.8：将尺寸线与 OCR 文本重新配对。"""

    top_yolox_pairs = find_list(L3, "top_yolox_pairs")
    bottom_yolox_pairs = find_list(L3, "bottom_yolox_pairs")
    side_yolox_pairs = find_list(L3, "side_yolox_pairs")
    detailed_yolox_pairs = find_list(L3, "detailed_yolox_pairs")
    side_angle_pairs = find_list(L3, "side_angle_pairs")
    detailed_angle_pairs = find_list(L3, "detailed_angle_pairs")
    top_border = find_list(L3, "top_border")
    bottom_border = find_list(L3, "bottom_border")
    top_ocr_data = find_list(L3, "top_ocr_data")
    bottom_ocr_data = find_list(L3, "bottom_ocr_data")
    side_ocr_data = find_list(L3, "side_ocr_data")
    detailed_ocr_data = find_list(L3, "detailed_ocr_data")

    (
        top_ocr_data,
        bottom_ocr_data,
        side_ocr_data,
        detailed_ocr_data,
    ) = _pairs_module.MPD(
        key,
        top_yolox_pairs,
        bottom_yolox_pairs,
        side_yolox_pairs,
        detailed_yolox_pairs,
        side_angle_pairs,
        detailed_angle_pairs,
        top_border,
        bottom_border,
        top_ocr_data,
        bottom_ocr_data,
        side_ocr_data,
        detailed_ocr_data,
    )

    recite_data(L3, "top_ocr_data", top_ocr_data)
    recite_data(L3, "bottom_ocr_data", bottom_ocr_data)
    recite_data(L3, "side_ocr_data", side_ocr_data)
    recite_data(L3, "detailed_ocr_data", detailed_ocr_data)

    return L3


def finalize_pairs(L3):
    """F4.8：清理配对结果，输出最终可用的尺寸线集合。"""

    top_ocr_data = find_list(L3, "top_ocr_data")
    bottom_ocr_data = find_list(L3, "bottom_ocr_data")
    side_ocr_data = find_list(L3, "side_ocr_data")
    detailed_ocr_data = find_list(L3, "detailed_ocr_data")
    top_yolox_pairs_length = find_list(L3, "top_yolox_pairs_length")
    bottom_yolox_pairs_length = find_list(L3, "bottom_yolox_pairs_length")
    side_yolox_pairs_length = find_list(L3, "side_yolox_pairs_length")
    detailed_yolox_pairs_length = find_list(L3, "detailed_yolox_pairs_length")
    top_yolox_pairs_copy = find_list(L3, "top_yolox_pairs_copy")
    bottom_yolox_pairs_copy = find_list(L3, "bottom_yolox_pairs_copy")
    side_yolox_pairs_copy = find_list(L3, "side_yolox_pairs_copy")
    detailed_yolox_pairs_copy = find_list(L3, "detailed_yolox_pairs_copy")

    (
        top_ocr_data,
        bottom_ocr_data,
        side_ocr_data,
        detailed_ocr_data,
        yolox_pairs_top,
        yolox_pairs_bottom,
        yolox_pairs_side,
        yolox_pairs_detailed,
    ) = _pairs_module.get_better_data_2(
        top_ocr_data,
        bottom_ocr_data,
        side_ocr_data,
        detailed_ocr_data,
        top_yolox_pairs_length,
        bottom_yolox_pairs_length,
        side_yolox_pairs_length,
        detailed_yolox_pairs_length,
        top_yolox_pairs_copy,
        bottom_yolox_pairs_copy,
        side_yolox_pairs_copy,
        detailed_yolox_pairs_copy,
    )

    recite_data(L3, "top_ocr_data", top_ocr_data)
    recite_data(L3, "bottom_ocr_data", bottom_ocr_data)
    recite_data(L3, "side_ocr_data", side_ocr_data)
    recite_data(L3, "detailed_ocr_data", detailed_ocr_data)
    recite_data(L3, "yolox_pairs_top", yolox_pairs_top)
    recite_data(L3, "yolox_pairs_bottom", yolox_pairs_bottom)
    recite_data(L3, "yolox_pairs_side", yolox_pairs_side)
    recite_data(L3, "yolox_pairs_detailed", yolox_pairs_detailed)

    print("***/数据整理结果/***")
    print("top视图数据整理结果:\n", *top_ocr_data, sep="\n")
    print("bottom视图数据整理结果:\n", *bottom_ocr_data, sep="\n")
    print("side视图数据整理结果:\n", *side_ocr_data, sep="\n")
    print("detailed视图数据整理结果:\n", *detailed_ocr_data, sep="\n")

    return L3


def compute_BGA_parameters(L3):
    """F4.9：根据配对结果计算 BGA 参数列表。"""

    top_serial_numbers_data = find_list(L3, "top_serial_numbers_data")
    bottom_serial_numbers_data = find_list(L3, "bottom_serial_numbers_data")
    top_ocr_data = find_list(L3, "top_ocr_data")
    bottom_ocr_data = find_list(L3, "bottom_ocr_data")
    side_ocr_data = find_list(L3, "side_ocr_data")
    detailed_ocr_data = find_list(L3, "detailed_ocr_data")
    yolox_pairs_top = find_list(L3, "yolox_pairs_top")
    yolox_pairs_bottom = find_list(L3, "yolox_pairs_bottom")
    top_yolox_pairs_length = find_list(L3, "top_yolox_pairs_length")
    bottom_yolox_pairs_length = find_list(L3, "bottom_yolox_pairs_length")
    top_border = find_list(L3, "top_border")
    bottom_border = find_list(L3, "bottom_border")

    nx, ny = _pairs_module.get_serial(top_serial_numbers_data, bottom_serial_numbers_data)
    body_x, body_y = _pairs_module.get_body(
        yolox_pairs_top,
        top_yolox_pairs_length,
        yolox_pairs_bottom,
        bottom_yolox_pairs_length,
        top_border,
        bottom_border,
        top_ocr_data,
        bottom_ocr_data,
    )

    BGA_parameter_list = _pairs_module.get_BGA_parameter_list(
        top_ocr_data,
        bottom_ocr_data,
        side_ocr_data,
        detailed_ocr_data,
        body_x,
        body_y,
    )

    BGA_parameter_list = _pairs_module.resort_parameter_list_2(BGA_parameter_list)

    if len(BGA_parameter_list[4]["maybe_data"]) > 1:
        high = _pairs_module.get_QFP_high(BGA_parameter_list[4]["maybe_data"])
        if len(high) > 0:
            BGA_parameter_list[4]["maybe_data"] = high
            BGA_parameter_list[4]["maybe_data_num"] = len(high)

    if (
        len(BGA_parameter_list[5]["maybe_data"]) > 1
        or len(BGA_parameter_list[6]["maybe_data"]) > 1
    ):
        pitch_x, pitch_y = _pairs_module.get_QFP_pitch(
            BGA_parameter_list[5]["maybe_data"],
            body_x,
            body_y,
            nx,
            ny,
        )
        if len(pitch_x) > 0:
            BGA_parameter_list[5]["maybe_data"] = pitch_x
            BGA_parameter_list[5]["maybe_data_num"] = len(pitch_x)
        if len(pitch_y) > 0:
            BGA_parameter_list[6]["maybe_data"] = pitch_y
            BGA_parameter_list[6]["maybe_data_num"] = len(pitch_y)

    BGA_parameter_list = _pairs_module.resort_parameter_list_2(BGA_parameter_list)

    # run_and_save_resort_log(BGA_parameter_list) # 将参数候选列表保存至txt文件

    return BGA_parameter_list, nx, ny




