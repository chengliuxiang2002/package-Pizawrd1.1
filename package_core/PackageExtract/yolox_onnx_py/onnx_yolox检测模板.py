"""YOLOX ONNX 通用检测模板脚本。"""

#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
import numpy as np

import onnxruntime
try:
    from packagefiles.PackageExtract.yolox_onnx_py.yolox_onnx_shared import (
        demo_postprocess,
        mkdir,
        multiclass_nms,
        preprocess,
        vis,
    )
except ModuleNotFoundError:  # pragma: no cover - 兼容脚本直接运行
    from pythonProject.packagefiles.PackageExtract.yolox_onnx_py.yolox_onnx_shared import (
        demo_postprocess,
        mkdir,
        multiclass_nms,
        preprocess,
        vis,
    )

import argparse
import json
import os
import time
import re
import shutil
from shapely.geometry import Polygon, Point

import cv2
import numpy as np
import fitz
import onnxruntime

# from yolox.data.data_augment import preproc as preprocess
# from yolox.data.datasets import COCO_CLASSES
# from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis


def make_parser():
    """构建命令行参数解析器，配置模型与输入输出路径。"""
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox_onnx/output_pairs_data_pin_5.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default='data/bottom.jpg',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='onnx_output/output_pairs_data_pin_5.onnx',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    return parser

# 公共推理工具函数通过 yolox_onnx_shared 复用。
def onnx_output_pairs_data_pin_5():
    """运行 YOLOX 模型获取 PIN 检测框并保存可视化。"""
    VOC_CLASSES = ('package', "bga", "qfn", "Form", "Note")
    args = make_parser().parse_args()

    input_shape = tuple(map(int, args.input_shape.split(',')))
    origin_img = cv2.imread(args.image_path)
    img, ratio = preprocess(origin_img, input_shape)

    session = onnxruntime.InferenceSession(args.model)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    predictions = demo_postprocess(output[0], input_shape)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                         conf=args.score_thr, class_names=VOC_CLASSES)
    '''
    final_boxes:记录yolox检测的坐标位置np(, 4)[x1,y1,x2,y2]
    final_cls_inds:记录每个yolox检测的种类np(, )[1,2,3,]
    final_scores:记录yolox每个检测的分数np(, )[80.9,90.1,50.2,]
    '''

    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
    cv2.imwrite(output_path, origin_img)


if __name__ == '__main__':
    VOC_CLASSES = ('package', "bga", "qfn", "Form", "Note")
    args = make_parser().parse_args()

    input_shape = tuple(map(int, args.input_shape.split(',')))
    origin_img = cv2.imread(args.image_path)
    img, ratio = preprocess(origin_img, input_shape)

    session = onnxruntime.InferenceSession(args.model)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    predictions = demo_postprocess(output[0], input_shape)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                         conf=args.score_thr, class_names=VOC_CLASSES)
    else:
        final_boxes = np.zeros((0, 4))
        final_scores = np.zeros(0)
        final_cls_inds = np.zeros(0)
    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
    cv2.imwrite(output_path, origin_img)
