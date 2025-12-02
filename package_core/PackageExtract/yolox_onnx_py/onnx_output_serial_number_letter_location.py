"""YOLOX ONNX 序号字母检测脚本。"""

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



def make_parser(img_path):
    """构建命令行参数解析器，配置模型与输入输出路径。"""
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox_onnx/serial_numbers_letters.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default=img_path,
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='onnx_output/serial_numbers_letters.onnx',
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

def get_img_info(img_path):
    """读取图片尺寸信息并返回宽高。"""

    image = cv2.imread(img_path)
    size = image.shape
    w = size[1]  # 宽度
    h = size[0]  # 高度
    print(w,h)
    return w, h
def get_rotate_crop_image(img, points):  # 图片分割，在ultil中的原有函数,from utils import get_rotate_crop_image
    """依据四点坐标裁剪旋转矩形区域。"""
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
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    # if dst_img_height * 1.0 / dst_img_width >= 1:
    #     dst_img = np.rot90(dst_img)
    return dst_img

def onnx_inference(img_path):
    """执行 ONNX 模型推理并整理检测结果。"""
    VOC_CLASSES = ('package', "bga", "qfn", "Form", "Note")
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
    else:
        final_boxes = np.zeros((0, 4))
        final_scores = np.zeros(0)
        final_cls_inds = np.zeros(0)
    bboxes = final_boxes
    cls = final_cls_inds
    bboxes_np = np.array(bboxes)
    other = np.zeros((0, 4))
    guangjiedu = np.zeros((0, 4))
    for i in range(len(cls)):
        if cls[i] == 0:  # other
            other = np.r_[other, [bboxes_np[i]]]
        if cls[i] == 1:  # 光洁度
            guangjiedu = np.r_[guangjiedu, [bboxes_np[i]]]
    global OTHER
    global GUANGJIEDU
    OTHER = other
    GUANGJIEDU = guangjiedu
    '''
    final_boxes:记录yolox检测的坐标位置np(, 4)[x1,y1,x2,y2]
    final_cls_inds:记录每个yolox检测的种类np(, )[1,2,3,]
    final_scores:记录yolox每个检测的分数np(, )[80.9,90.1,50.2,]
    '''

    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
    cv2.imwrite(output_path, origin_img)

def output_pairs_data_location(cls, bboxes):
    """拆分模型输出，生成业务需要的坐标集合。"""
    # print("cls",cls)#tensor([1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.])
    # print("bboxes",bboxes)#(x1,y1,x2,y2)左上角与右下角坐标，yolox坐标原点是左上角
    # tensor([[ 781.2277,  311.5244,  820.7728,  350.0395],
    # [1039.5348,  101.0938, 1071.0819,  149.9549],
    # [ 532.6776,   97.2061,  604.5829,  143.0282],
    # [ 764.6501,  447.9761,  804.3810,  484.4607],
    # [ 317.2012,  520.6475,  372.4589,  589.8723],
    # [ 721.6006,  253.5669,  795.9207,  293.0987],
    # [ 754.2359,  209.0722,  765.7953,  417.7099],
    # [ 432.3812,  123.5709,  689.7897,  133.8431],
    # [1065.0292,  153.1357, 1110.1744,  163.3379],
    # [ 831.5218,  337.4080,  841.0558,  387.1069],
    # [ 389.1720,  567.0306,  400.0041,  606.2711],
    # [ 814.5815,  403.3475,  823.8375,  460.4219]])


    num = 0
    # 获取类型的数量
    for i in range(len(cls)):

        if cls[i] == 0:
            num += 1
    # print(num)
    global location
    location = np.zeros((num, 4))
    location = np.array(bboxes)
    #当模型检测pad时
    if cls.any():
        bboxes_np = np.array(bboxes)
        location = np.zeros((0, 4))
        for i in range(len(cls)):
            if cls[i] == 3:
                num += 1
                location = np.r_[location, [bboxes_np[i]]]
def begain_output_serial_number_letter_location(img_path):
    """封装入口：输出序号字母检测结果。"""
    global OTHER
    global GUANGJIEDU

    OTHER = np.zeros((0, 4))
    GUANGJIEDU = np.zeros((0, 4))

    onnx_inference(img_path)

    serial_numbers = OTHER
    serial_letters = GUANGJIEDU
    return serial_numbers, serial_letters

if __name__ == '__main__':
   pass
