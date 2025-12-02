"""YOLOX ONNX 外框检测脚本。"""

#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
import numpy as np

import onnxruntime
try:
    from package_core.PackageExtract.yolox_onnx_py.yolox_onnx_shared import (
        demo_postprocess,
        mkdir,
        multiclass_nms,
        preprocess,
        vis,
    )
except ModuleNotFoundError:  # pragma: no cover - 兼容脚本直接运行
    from package_core.PackageExtract.yolox_onnx_py.yolox_onnx_shared import (
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
from math import sqrt


# from yolox.data.data_augment import preproc as preprocess
# from yolox.data.datasets import COCO_CLASSES
# from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis


def make_parser(path):
    """构建命令行参数解析器，配置模型与输入输出路径。"""
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox_onnx/bottom_waikuang.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default=path,
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='onnx_output/bottom_waikuang.onnx',
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


def onnx_output_waikuang(path):
    """执行外框检测并返回对应坐标。"""
    VOC_CLASSES = ('waikuang')
    args = make_parser(path).parse_args()

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
    location = get_the_only_waikuang(final_boxes, final_cls_inds, final_scores)

    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
    cv2.imwrite(output_path, origin_img)

    return location


def get_the_only_waikuang(final_boxes, final_cls_inds, final_scores):
    """在候选中挑选唯一的外框结果。"""
    # 修正yolox框选超出图片边界的问题
    img_path = r'data/bottom.jpg'
    img = cv2.imread(img_path)
    w, h = img.shape[1], img.shape[0]
    for i in range(len(final_boxes)):
        if final_boxes[i][0] < 0:
            final_boxes[i][0] = 0
        if final_boxes[i][1] < 0:
            final_boxes[i][1] = 0
        if final_boxes[i][2] > w:
            final_boxes[i][2] = w
        if final_boxes[i][3] > h:
            final_boxes[i][3] = h

    location = []
    if len(final_boxes) == 0:
        return location
    else:
        if len(final_boxes) > 0:
            if len(final_boxes) == 1:
                location.append(final_boxes[0][0])
                location.append(final_boxes[0][1])
                location.append(final_boxes[0][2])
                location.append(final_boxes[0][3])
                return location
            if len(final_boxes) > 1:
                mark_location = np.zeros((len(final_boxes), 1))
                for i in range(len(final_boxes)):
                    mark_location[i] = sqrt((final_boxes[i][2] - final_boxes[i][0]) ** 2 + (
                            final_boxes[i][3] - final_boxes[i][1]) ** 2)
                max_no = np.argmax(mark_location)

                location.append(final_boxes[max_no][0])
                location.append(final_boxes[max_no][1])
                location.append(final_boxes[max_no][2])
                location.append(final_boxes[max_no][3])

                return location


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
    else:
        final_boxes = np.zeros((0, 4))
        final_scores = np.zeros(0)
        final_cls_inds = np.zeros(0)
    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
    cv2.imwrite(output_path, origin_img)
