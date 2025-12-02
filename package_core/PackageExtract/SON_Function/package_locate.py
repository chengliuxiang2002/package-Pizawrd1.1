import argparse
import json
import os
import time
import re
import shutil
from pathlib import Path
from shapely.geometry import Polygon, Point
import xml.etree.ElementTree as ET
import xml.dom.minidom

import cv2
import numpy as np
import fitz
import onnxruntime

try:
    from package_core.PackageExtract.yolox_onnx_py.model_paths import yolo_model_path, result_path
except ModuleNotFoundError:  # pragma: no cover - 兼容脚本直接运行
    def yolo_model_path(*parts):
        return str(Path(__file__).resolve().parents[3] / 'model' / 'yolo_model' / Path(*parts))
    def result_path(*parts):
        return str(Path(__file__).resolve().parents[3] / 'Result' / Path(*parts))

VOC_CLASSES = ('Border',
               'Pad',
               'Pin',
               'multi_value_1',
               'multi_value_2',
               'multi_value_3',
               'multi_value_triangle',
               'other',
               'pairs_inside_col',
               'pairs_inside_row',
               'pairs_outside_col',
               'pairs_outside_row',
               'serial_number')
# yolox_onnx 需要的一些函数(从yolox中提取)
def preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)

def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets

def demo_postprocess(outputs, img_size, p6=False):
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    _COLORS = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            0.000, 0.447, 0.741,
            0.314, 0.717, 0.741,
            0.50, 0.5, 0
        ]
    ).astype(np.float32).reshape(-1, 3)
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img

def empty_file(directory):
    """
        若存在该文件目录，则清空directory路径下的文件夹里的内容，否则创建空文件目录
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)  # 删除目录下所有文件及目录
        os.makedirs(directory)  # 创建目录
    else:
        os.makedirs(directory)

def natural_sort_key(s):
    """
    使用正则匹配文件名中的数字
    假设文件名格式为'1.png'
    """
    int_part = re.search(r'\d+', s).group()
    return int(int_part), s

def get_rects_d(rect1_coords, rect2_coords):
    """
        输入两个矩形坐标，返回这两个矩形之间的最短距离
    """
    rect1 = Polygon([(rect1_coords[0], rect1_coords[1]),
                     (rect1_coords[0], rect1_coords[3]),
                     (rect1_coords[2], rect1_coords[3]),
                     (rect1_coords[2], rect1_coords[1])])

    rect2 = Polygon([(rect2_coords[0], rect2_coords[1]),
                     (rect2_coords[0], rect2_coords[3]),
                     (rect2_coords[2], rect2_coords[3]),
                     (rect2_coords[2], rect2_coords[1])])

    return rect1.distance(rect2)

def get_max_conf_index(type_index, data):
    """
        根据索引值，获取最大置信度的索引
    """
    index = 0
    max = data[type_index[0]]
    for i in range(len(type_index)):
        if (data[type_index[i]] >= max):
            max = data[type_index[i]]
            index = i
    return i

def make_parser():
    """
    onnx格式的yolox检测 一些参数设置
    """
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=yolo_model_path("package_model", "DFN_SON_0910.onnx"),
        help="Input your onnx model.",
    )                                      # 权重参数选择
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default=r'./dataset',
        help="Path to your input image.",
    )                                     # 待检测图片文件夹
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.5,#默认0.8
        help="Score threshould to filter the result.",
    )                                         # 置信度阈值
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )                                           # 输入图片shape
    return parser


def onnx_inference(file_path):
    """
    onnx格式的yolox对指定路径下图片进行检测，返回信息列表 [{page: 1, pos: [[], [], ...], index[0, 0]}, {}, ...]
    """
    save_dir = result_path('Package_extract', 'yolox_son_onnx_result')
    args = make_parser().parse_args()
    input_shape = tuple(map(int, args.input_shape.split(',')))
    data = []                  # 存放字典信息
    # 生成文件保存地址文件夹
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result_file_dir = os.path.join(save_dir, '2024')   # 生成子文件夹
    if not os.path.exists(result_file_dir):
        os.makedirs(result_file_dir, exist_ok=True)    # 含结果的图片保存在该文件夹目录下
    #time3 = time.time()
    # img_path_list = os.listdir(args.image_path)
    # img_item_list = sorted(img_path_list, key=natural_sort_key)
    #for img_item in (img_item_list):
    flag = 0                 # 默认该页不画框
    origin_img = cv2.imread(file_path)
    img, ratio = preprocess(origin_img, input_shape)
    session = onnxruntime.InferenceSession(args.model)
    img_info = {"id": 0}
    height, width = origin_img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["file_name"] = os.path.basename(file_path)
    #print(ratio)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    predictions = demo_postprocess(output[0], input_shape)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    #print(scores)

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    #print(dets)
    if dets is not None:
        # 判断是否存在置信度满足要求的框
        for item in dets:
            if item[4] >= args.score_thr:
                flag = 1              # 需要画框
                break
        # 外加判断条件 是否存在置信度满足要求的框 -> 画框
        if (flag):
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            # 可视化显示
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                            conf=args.score_thr, class_names=VOC_CLASSES)  # 返回绘制好的图片
            # 存放符合阈值要求的封装图信息
            output_path = os.path.join(result_file_dir, os.path.basename(file_path))   # 带框文件保存路径

            cv2.imwrite(output_path, origin_img)   # 保存文件

            # 存取相关带框封装图信息,矩形框坐标,类型
            data_dic = {'filename': os.path.basename(file_path).split('.png')[0], 'pos': [], 'img_type': []
                , 'confidence': []}  # 实际文件
            for item in dets:
                if item[4] >= args.score_thr:
                    item0=round(item[0],4)
                    item1=round(item[1],4)
                    item2=round(item[2],4)
                    item3=round(item[3],4)
                    data_dic['pos'].append([item0, item1, item2, item3])   # 框坐标位置
                    data_dic['img_type'].append(int(item[5]))    # 特征图类型
                    data_dic['confidence'].append(item[4])   # 该标签的置信度值
            data.append(data_dic)      # 添加符号图相关信息
    return data  # 返回符号图字典列表信息




def imagevalue_classify_onnx(file_path):
    # 调用onnx格式的yolox检测函数，函数返回符号图信息列表
    data = onnx_inference(file_path)   # [{'filename': 1, 'pos': [[x1, y1, x2, y2], ...], 'img_type': [0, ...]}, {}, ...]
    return data

if __name__ == '__main__':
    imagevalue_classify_onnx("dataset/1.png")
