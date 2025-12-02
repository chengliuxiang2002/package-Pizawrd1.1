#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.
"""YOLOX ONNX 引脚检测脚本。

用于在 ONNXRuntime 中执行 PIN 相关目标检测，并在 F4.6-F4.9
流程中为后续参数推断提供基础几何信息。
"""

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
    from package_core.PackageExtract.yolox_onnx_py.yolox_onnx_shared import (
        demo_postprocess,
        mkdir,
        multiclass_nms,
        preprocess,
        vis,
    )


def make_parser(path):
    """构建解析器，配置模型、输入图像与输出目录等参数。"""
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox_onnx/output_pin_num2.onnx",
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
        default='onnx_output/pin',
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
def onnx_output_pairs_data_pin_5(img_path):
    """运行 YOLOX 模型获取 PIN 检测框并保存可视化结果。"""
    VOC_CLASSES = ('0', "1", "2", "3", "4", '5', "bga", "qfn", "Form", "Note")
    args = make_parser(img_path).parse_args()

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
    pin = np.zeros((0, 4))
    for i in range(len(final_cls_inds)):
        if final_cls_inds[i] == 5:
            pin = np.r_[pin, [final_boxes[i]]]

    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
    cv2.imwrite(output_path, origin_img)

    return pin

if __name__ == '__main__':
    #遍历文件夹中的所有文件
    path = r'bottom_pin_jpg'
    for root, dirs, files in os.walk(path):
        for file in files:
            new_path = path + '/' + file
            VOC_CLASSES = ('package', "bga", "qfn", "Form", "1", "pin", "3")
            args = make_parser(new_path).parse_args()

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
