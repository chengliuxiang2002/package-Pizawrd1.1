# 4版本，先yolox检测出pinmap位置防止误检，通过裁剪bottom图为几份，分别检测ball并最后合并结果
"""YOLOX ONNX 引脚数量检测与 PinMap 生成脚本。"""

# 问题：换行问题：以解决
# 问题：如何结束循环：目前测试的图片均已结束循环
# 局部缺整行整列pin时，会报错：已解决
# 拼接问题，需要同行图片的y坐标对齐：已解决
# 问题：分割的图片有时候会多出一行或者一列，这是由于yolox检测pin的框图不整齐，导致累计误差
import argparse
import os

import matplotlib.pyplot as plt
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

from PIL import Image
import shutil
from math import sqrt
import glob
import cv2
from random import randint
import math


def make_parser(img_path, conf):
    """构建命令行参数解析器，配置模型与输入输出路径。"""
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
        default=img_path,
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='onnx_output/output_pin_num2.onnx',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=conf,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    return parser


def ResizeImage(filein, fileout, scale=1):
    """
    改变图片大小
    :param filein: 输入图片
    :param fileout: 输出图片
    :param width: 输出图片宽度
    :param height: 输出图片宽度
    :param type: 输出图片类型（png, gif, jpeg...）
    :return:
    """

    img = Image.open(filein)
    if img.size[0] < 1080 and img.size[1] < 1080:  # 限制图片大小避免过大

        width = int(img.size[0] * scale)
        height = int(img.size[1] * scale)
        type = img.format
        out = img.resize((width, height), Image.LANCZOS)
        # 第二个参数：
        # Image.NEAREST ：低质量
        # Image.BILINEAR：双线性
        # Image.BICUBIC ：三次样条插值
        # Image.ANTIALIAS：高质量
        out.save(fileout, type)


def ResizeImage_137(filein, fileout):
    """
    改变图片大小
    :param filein: 输入图片
    :param fileout: 输出图片
    :param width: 输出图片宽度
    :param height: 输出图片宽度
    :param type: 输出图片类型（png, gif, jpeg...）
    :return:
    """

    img = Image.open(filein)

    height_1, width_1 = img.size[0], img.size[1]
    width = int(137)
    height = int(height_1 * 137 / width_1)
    type = img.format
    out = img.resize((width, height), Image.LANCZOS)
    # 第二个参数：
    # Image.NEAREST ：低质量
    # Image.BILINEAR：双线性
    # Image.BICUBIC ：三次样条插值
    # Image.ANTIALIAS：高质量
    out.save(fileout, type)


def ResizeImage_137_4(filein, fileout):
    """
    改变图片大小
    :param filein: 输入图片
    :param fileout: 输出图片
    :param width: 输出图片宽度
    :param height: 输出图片宽度
    :param type: 输出图片类型（png, gif, jpeg...）
    :return:
    """

    img = Image.open(filein)

    height_1, width_1 = img.size[0], img.size[1]
    width = int(137 * 4)
    height = int(height_1 * 137 * 4 / width_1)
    type = img.format
    out = img.resize((width, height), Image.LANCZOS)
    # 第二个参数：
    # Image.NEAREST ：低质量
    # Image.BILINEAR：双线性
    # Image.BICUBIC ：三次样条插值
    # Image.ANTIALIAS：高质量
    out.save(fileout, type)


# 公共推理工具函数通过 yolox_onnx_shared 复用。


def get_img_info(img_path):
    """读取图片尺寸信息并返回宽高。"""
    image = cv2.imread(img_path)
    size = image.shape
    w = size[1]  # 宽度
    h = size[0]  # 高度
    print(w, h)
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


def find_the_only_body(img_path):
    """在检测结果中定位唯一外框。"""
    global location
    global YOLOX_body
    print(location)
    if len(location) == 0:
        YOLOX_body = np.array([[1, 1, 1, 1]])
    if len(location) == 1:
        YOLOX_body = np.zeros((1, 4))
        YOLOX_body[0] = location[0]
    if len(location) > 1:
        new_location = location.copy()
        while len(new_location) > 1:
            # 删除离图片边界过近的框、超过图片边界的框
            w, h = get_img_info(img_path)
            print("w, h", w, h)
            mark_location = np.zeros((len(location)))
            new_new_location = np.zeros((0, 4))
            for i in range(len(location)):
                if location[i][0] < 0 or location[i][1] < 0 or location[i][2] > w or location[i][3] > h:
                    mark_location[i] = 1
            for i in range(len(mark_location)):
                if mark_location[i] == 0:
                    new_new_location = np.r_[new_new_location, [location[i]]]

            # 如果删除了距离边界过近的框还是数量大于1，找到最大的框作为body

            mark_location = np.zeros((len(new_new_location)))
            print("new_new_location", new_new_location)
            if len(new_new_location) > 1:
                for i in range(len(new_new_location)):
                    mark_location[i] = sqrt((new_new_location[i][2] - new_new_location[i][0]) ** 2 + (
                                new_new_location[i][3] - new_new_location[i][1]) ** 2)
                max_no = np.argmax(mark_location)

                new_location = np.zeros((0, 4))

                new_location = np.r_[new_location, [new_new_location[max_no]]]
                new_new_location = new_location
            new_location = new_new_location
            print("new_location", new_location)
        location = new_location
    if len(location) == 0:
        YOLOX_body = np.array([[1, 1, 1, 1]])
    if len(location) == 1:
        YOLOX_body = location
        box = np.array([[YOLOX_body[0][0], YOLOX_body[0][1]], [YOLOX_body[0][2], YOLOX_body[0][1]],
                        [YOLOX_body[0][2], YOLOX_body[0][3]], [YOLOX_body[0][0], YOLOX_body[0][3]]], np.float32)
        with open(img_path, 'rb') as f:
            np_arr = np.frombuffer(f.read(), dtype=np.uint8)
            # img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)    #以彩图读取
            img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)  # 以灰度图读取
        box_img = get_rotate_crop_image(img, box)
        cv2.namedWindow('origin', 0)
        cv2.imshow('origin', box_img)  # 显示当前ocr的识别区域
        cv2.waitKey(0)
    return YOLOX_body


def onnx_inference(img_path, conf):
    """执行 ONNX 模型推理并整理检测结果。"""
    VOC_CLASSES = ('1', "2", "3", "4", "5", "6", "7", "8", "9")
    args = make_parser(img_path, conf).parse_args()

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
    cls = np.array(final_cls_inds)
    bboxes = np.array(final_boxes)
    output_pin_num(cls, bboxes)
    '''
    final_boxes:记录yolox检测的坐标位置np(, 4)[x1,y1,x2,y2]
    final_cls_inds:记录每个yolox检测的种类np()[1,2,3,]
    final_scores:记录yolox每个检测的分数np()[80.9,90.1,50.2,]
    '''

    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, os.path.basename(args.image_path))
    cv2.imwrite(output_path, origin_img)


def output_pin_num(cls, bboxes):
    """整理 PIN 数量检测结果并输出。"""
    #########################################输出识别的类别和对角线坐标

    # cls_np = np.array(cls)
    bboxes_np = np.array(bboxes)
    # print("cls",cls)#tensor([1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.])
    # print("bboxes",bboxes)#(x1,y1,x2,y2)左下角与右上角坐标
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
    pin_num = 0

    # 获取pin的行和列的数量############################################################
    for i in range(len(cls)):
        if cls[i] == 5:  # 5=ball=pin
            pin_num += 1
    # 当未识别到pin时
    if pin_num == 0:
        txt_path = r'yolox_data\pin.txt'
        with open(txt_path, 'a+', encoding='utf-8') as test:
            test.truncate(0)
        pin_num_x_y = np.array([10, 10])
        pin_num_x_y = pin_num_x_y.astype(int)
        path = r'yolox_data\pin_num.txt'
        # print("[一行的ball数,一列的ball数]", pin_num_x_y)
        np.savetxt(path, pin_num_x_y)
        return

    # print("pin_num\n", pin_num)
    # 获取pin框的平均宽度
    x_length = []
    y_length = []
    key = 0
    for i in range(len(cls)):
        if cls[i] == 5:
            x_length.append(bboxes_np[i][2] - bboxes_np[i][0])
            y_length.append(bboxes_np[i][3] - bboxes_np[i][1])
    if pin_num > 9:  # 照顾到仅有个位数pin数量时，不删除最大最小值
        x_length.remove(max(x_length))
        x_length.remove(min(x_length))
        x_length.remove(max(x_length))
        x_length.remove(min(x_length))
        y_length.remove(max(y_length))
        y_length.remove(min(y_length))
        y_length.remove(max(y_length))
        y_length.remove(min(y_length))
    average_x_length = np.mean(x_length)
    average_y_length = np.mean(y_length)
    # 将cls和bbox中的pin集合在一起并删除明显小一些的pin视为噪声
    pin = np.zeros((pin_num, 4))
    pin_diagonal = np.zeros((pin_num,))
    key = 0
    ratio = 0.3  # 控制筛选的变量，越小则越筛选越宽松

    for i in range(len(cls)):
        if cls[i] == 5 and (bboxes[i][2] - bboxes[i][0]) / average_x_length > ratio and (
                bboxes[i][3] - bboxes[i][1]) / average_y_length > ratio:  # 筛选长和宽相比平均长和宽不正常的pin
            pin[key] = bboxes[i]
            pin_diagonal[key] = math.sqrt((bboxes[i][2] - bboxes[i][0]) ** 2 + (bboxes[i][3] - bboxes[i][1]) ** 2)
            key += 1

    a = len(pin)
    for i in range(len(pin)):  # 同步去除pin和pin_diagonal的全零项
        if pin[i][0] == 0 or pin[i][1] == 0 or pin[i][2] == 0 or pin[i][3] == 0:
            pin = np.delete(pin, np.s_[-(len(pin) - i):], 0)
            # pin_diagonal = np.delete(pin_diagonal,[-(len(pin)-i):])
            pin_diagonal = pin_diagonal[:-(a - i)]

            break

    # print(pin)#存储所有pin框的坐标
    # 剔除pin中斜边明显异常值
    n = 8  # 去掉最大值最小值的个数
    pin_diagonal_copy = pin_diagonal.copy()
    c = pin
    # print(c)
    if pin_num > 9:

        pin_diagonal.sort()

        avv = pin_diagonal[int(n * 0.5):-(n - int(n * 0.5))]  # 去掉n个最小最大值

        av = np.mean(avv)

        new_pin = np.empty((pin.shape[0], pin.shape[1] + 1))

        # pin和pin_diagonal_copy结合
        for i in range(pin.shape[0]):
            # 将原始行的元素复制到新数组中
            new_pin[i, :-1] = pin[i]
            # 在新数组的末尾添加值
            new_pin[i, -1] = pin_diagonal_copy[i]

        # print("new_pin",new_pin)

        # 布尔索引删除对角线长度异常的pin
        last_column = new_pin[:, -1]
        # 使用条件进行布尔索引，选择满足条件的行
        mask = abs(last_column - av) / av < 0.5
        new_pin = new_pin[mask]

        pin = new_pin[:, :-1]
        pin_num = len(pin)

    ########################################################################################判断是否缺pin:开始
    lie_pin = 0  # 几列
    key = np.empty((0,), dtype=int)  # 存储已经被分类成列的pin的序号
    lie_set = np.full((99, 99), -1)  # 存储列pin
    lie_set_key = 0


    #************************old
    # for i in range(len(pin)):  # 将相同列的pin序号排在一起
    #     if i not in key:
    #         lie_set_set_key = 0
    #         key = np.append(key, i)
    #         x_stand = (pin[i][0] + pin[i][2]) * 0.5
    #         lie_set[lie_set_key][lie_set_set_key] = i
    #
    #         lie_set_set_key += 1
    #         for j in range(len(pin)):
    #             if j not in key and j != i:
    #                 x_stand_j = (pin[j][0] + pin[j][2]) * 0.5
    #                 if abs(x_stand - x_stand_j) < average_x_length:
    #                     key = np.append(key, j)
    #                     lie_set[lie_set_key][lie_set_set_key] = j
    #                     lie_set_set_key += 1
    #         lie_set_key += 1
    #************************old
    #******************************new
    # 如果两个pin在横坐标上重合的距离在两边占比都超过50%
    for i in range(len(pin)):  # 将相同列的pin序号排在一起
        if i not in key:
            lie_set_set_key = 0
            key = np.append(key, i)
            lie_set[lie_set_key][lie_set_set_key] = i

            lie_set_set_key += 1
            for j in range(len(pin)):
                if j not in key and j != i:
                    if max(pin[i][2], pin[j][2]) - min(pin[i][0], pin[j][0]) < pin[i][2] + pin[j][2] - pin[i][0] - pin[j][0]:
                        a = max(pin[i][2], pin[j][2]) - min(pin[i][0], pin[j][0])
                        b = pin[i][2] + pin[j][2] - pin[i][0] - pin[j][0]
                        if (b - a)/(pin[i][2] - pin[i][0]) > 0.5 and (b - a)/(pin[j][2] - pin[j][0]) > 0.5:
                            key = np.append(key, j)
                            lie_set[lie_set_key][lie_set_set_key] = j
                            lie_set_set_key += 1
            lie_set_key += 1

    #******************************new
    pin_loss_key = 0  # 0=不缺pin 1=缺整列 2=缺整行pin
    # data_set = lie_set
    for i in range(len(lie_set)):
        pin_no = lie_set[i][0]
        x_stand = (pin[pin_no][0] + pin[pin_no][2]) * 0.5
        min_pitch = 9999
        for j in range(len(lie_set)):
            if j != i:
                pin_no_j = lie_set[j][0]
                x_stand_j = (pin[pin_no_j][0] + pin[pin_no_j][2]) * 0.5
                if x_stand_j < x_stand:
                    continue
                elif average_x_length < x_stand_j - x_stand < min_pitch:
                    min_pitch = abs(x_stand - x_stand_j)
        # print(min_pitch)
        if min_pitch > average_x_length * 3 and min_pitch != 9999:
            print("检测到整列缺pin")
            data_set = lie_set
            pin_loss_key = 1

    hang_pin = 0  # 几行
    key = np.empty((0,), dtype=int)  # 存储已经被分类成行的pin的序号
    hang_set = np.full((99, 99), -1)  # 存储行pin
    hang_set_key = 0

    #*********************************old
    # for i in range(len(pin)):  # 将相同行的pin序号排在一起
    #     if i not in key:
    #         hang_set_set_key = 0
    #         key = np.append(key, i)
    #         y_stand = (pin[i][1] + pin[i][3]) * 0.5
    #         hang_set[hang_set_key][hang_set_set_key] = i
    #
    #         hang_set_set_key += 1
    #         for j in range(len(pin)):
    #             if j not in key and j != i:
    #                 y_stand_j = (pin[j][1] + pin[j][3]) * 0.5
    #                 if abs(y_stand - y_stand_j) < average_y_length:
    #                     key = np.append(key, j)
    #                     hang_set[hang_set_key][hang_set_set_key] = j
    #                     hang_set_set_key += 1
    #         hang_set_key += 1
    # *********************************old
    # *********************************new
    for i in range(len(pin)):  # 将相同行的pin序号排在一起
        if i not in key:
            hang_set_set_key = 0
            key = np.append(key, i)
            # y_stand = (pin[i][1] + pin[i][3]) * 0.5
            hang_set[hang_set_key][hang_set_set_key] = i

            hang_set_set_key += 1
            for j in range(len(pin)):
                if j not in key and j != i:
                    if max(pin[i][3], pin[j][3]) - min(pin[i][1], pin[j][1]) < pin[i][3] + pin[j][3] - pin[i][1] - pin[j][1]:
                        a = max(pin[i][3], pin[j][3]) - min(pin[i][1], pin[j][1])
                        b = pin[i][3] + pin[j][3] - pin[i][1] - pin[j][1]
                        if (b - a)/(pin[i][3] - pin[i][1]) > 0.5 and (b - a)/(pin[j][3] - pin[j][1]) > 0.5:
                    # y_stand_j = (pin[j][1] + pin[j][3]) * 0.5
                    # if abs(y_stand - y_stand_j) < average_y_length:
                            key = np.append(key, j)
                            hang_set[hang_set_key][hang_set_set_key] = j
                            hang_set_set_key += 1
            hang_set_key += 1
    # *********************************new
    # print(hang_set)
    # data_set = hang_set
    for i in range(len(hang_set)):
        pin_no = hang_set[i][0]
        y_stand = (pin[pin_no][1] + pin[pin_no][3]) * 0.5
        min_pitch = 9999
        for j in range(len(hang_set)):
            if j != i:
                pin_no_j = hang_set[j][0]
                y_stand_j = (pin[pin_no_j][1] + pin[pin_no_j][3]) * 0.5
                if y_stand_j < y_stand:
                    continue
                elif average_y_length < y_stand_j - y_stand < min_pitch:
                    min_pitch = abs(y_stand - y_stand_j)

        if min_pitch > average_y_length * 3 and min_pitch != 9999:
            print("检测到整行缺pin")
            data_set = hang_set
            pin_loss_key = 2

    if pin_loss_key == 1:  # 缺整列则列pin数完整，输出列pin，行pin为0
        for_num = 6  # 随机抽取次数

        x_pitch_1 = 9999
        y_pitch_1 = 9999
        for i in range(for_num):
            pin_no = randint(0, pin_num - 1)
            x_middle = (pin[pin_no][2] + pin[pin_no][0]) / 2
            y_middle = (pin[pin_no][3] + pin[pin_no][1]) / 2

            for j in range(pin_num):
                if j != pin_no:
                    x_j_middle = (pin[j][2] + pin[j][0]) / 2
                    y_j_middle = (pin[j][3] + pin[j][1]) / 2
                    x_pitch_2 = abs(x_j_middle - x_middle)
                    y_pitch_2 = abs(y_j_middle - y_middle)
                    # print((x_pitch_2))
                    if average_x_length < x_pitch_2 and x_pitch_2 < x_pitch_1:
                        x_pitch_1 = x_pitch_2
                    if average_y_length < y_pitch_2 and y_pitch_2 < y_pitch_1:
                        y_pitch_1 = y_pitch_2
        # print(x_pitch_1)
        x_min = 9999
        x_max = 0
        for i in range(pin_num):
            if pin[i][0] < x_min:
                x_min = pin[i][0]
            if pin[i][2] > x_max:
                x_max = pin[i][2]
        y_min = 9999
        y_max = 0
        for i in range(pin_num):
            if pin[i][1] < y_min:
                y_min = pin[i][1]
            if pin[i][3] > y_max:
                y_max = pin[i][3]
        pin_x_num = (x_max - x_min) / x_pitch_1
        pin_x_num = round(pin_x_num)
        pin_y_num = (y_max - y_min) / y_pitch_1
        pin_y_num = round(pin_y_num)
        # print(pin_x_num,pin_y_num)

        pin_num_x_y = np.array([0, pin_y_num])
        pin_num_x_y = pin_num_x_y.astype(int)
        path = r'yolox_data\pin_num.txt'
        # print("[一行的ball数,一列的ball数]", pin_num_x_y)
        np.savetxt(path, pin_num_x_y)

    if pin_loss_key == 2:  # 缺整行则列pin数完整，输出列pin，行pin为0
        for_num = 6  # 随机抽取次数

        x_pitch_1 = 9999
        y_pitch_1 = 9999
        for i in range(for_num):
            pin_no = randint(0, pin_num - 1)
            x_middle = (pin[pin_no][2] + pin[pin_no][0]) / 2
            y_middle = (pin[pin_no][3] + pin[pin_no][1]) / 2

            for j in range(pin_num):
                if j != pin_no:
                    x_j_middle = (pin[j][2] + pin[j][0]) / 2
                    y_j_middle = (pin[j][3] + pin[j][1]) / 2
                    x_pitch_2 = abs(x_j_middle - x_middle)
                    y_pitch_2 = abs(y_j_middle - y_middle)
                    # print((x_pitch_2))
                    if average_x_length < x_pitch_2 and x_pitch_2 < x_pitch_1:
                        x_pitch_1 = x_pitch_2
                    if average_y_length < y_pitch_2 and y_pitch_2 < y_pitch_1:
                        y_pitch_1 = y_pitch_2
        # print(x_pitch_1)
        x_min = 9999
        x_max = 0
        for i in range(pin_num):
            if pin[i][0] < x_min:
                x_min = pin[i][0]
            if pin[i][2] > x_max:
                x_max = pin[i][2]
        y_min = 9999
        y_max = 0
        for i in range(pin_num):
            if pin[i][1] < y_min:
                y_min = pin[i][1]
            if pin[i][3] > y_max:
                y_max = pin[i][3]
        pin_x_num = (x_max - x_min) / x_pitch_1
        pin_x_num = round(pin_x_num)
        pin_y_num = (y_max - y_min) / y_pitch_1
        pin_y_num = round(pin_y_num)
        # print(pin_x_num,pin_y_num)

        pin_num_x_y = np.array([pin_x_num, 0])
        pin_num_x_y = pin_num_x_y.astype(int)
        path = r'yolox_data\pin_num.txt'
        # print("[一行的ball数,一列的ball数]", pin_num_x_y)
        np.savetxt(path, pin_num_x_y)

    ################################传递pin average_x_num average_y_num data_set
    if pin_loss_key != 0:
        txt_path = r'yolox_data\pin.txt'
        np.savetxt(txt_path, pin, delimiter=',')

        txt_path = r'yolox_data\hang_or_lie_set.txt'
        np.savetxt(txt_path, data_set, delimiter=',')

        # average_length_x_y = np.array([average_x_length, average_y_length])
        # path = r'G:\PaddleOCR-release-2.7\yolox_data\average_x_y.txt'
        # np.savetxt(path, average_length_x_y)

        for_num = 6  # 随机抽取次数

        x_pitch_1 = 9999
        y_pitch_1 = 9999
        for i in range(for_num):
            pin_no = randint(0, pin_num - 1)
            x_middle = (pin[pin_no][2] + pin[pin_no][0]) / 2
            y_middle = (pin[pin_no][3] + pin[pin_no][1]) / 2

            for j in range(pin_num):
                if j != pin_no:
                    x_j_middle = (pin[j][2] + pin[j][0]) / 2
                    y_j_middle = (pin[j][3] + pin[j][1]) / 2
                    x_pitch_2 = abs(x_j_middle - x_middle)
                    y_pitch_2 = abs(y_j_middle - y_middle)
                    # print((x_pitch_2))
                    if average_x_length < x_pitch_2 and x_pitch_2 < x_pitch_1:
                        x_pitch_1 = x_pitch_2
                    if average_y_length < y_pitch_2 and y_pitch_2 < y_pitch_1:
                        y_pitch_1 = y_pitch_2
        average_pitch_x_y = np.array([x_pitch_1, y_pitch_1])
        path = r'yolox_data\average_x_y.txt'
        np.savetxt(path, average_pitch_x_y)

    ####################################################################################判断是否缺pin：结束
    # 随机取pin计算最小pin的x和y的间距，然后取平均值作为pin间距
    if pin_loss_key == 0:  # 没有缺整行或列pin，但是有可能少pin
        for_num = 6  # 随机抽取次数

        x_pitch_1 = 9999
        y_pitch_1 = 9999
        for i in range(for_num):
            pin_no = randint(0, pin_num - 1)
            x_middle = (pin[pin_no][2] + pin[pin_no][0]) / 2
            y_middle = (pin[pin_no][3] + pin[pin_no][1]) / 2

            for j in range(len(pin)):
                if j != pin_no:
                    x_j_middle = (pin[j][2] + pin[j][0]) / 2
                    y_j_middle = (pin[j][3] + pin[j][1]) / 2
                    x_pitch_2 = abs(x_j_middle - x_middle)
                    y_pitch_2 = abs(y_j_middle - y_middle)
                    # print((x_pitch_2))
                    if average_x_length < x_pitch_2 and x_pitch_2 < x_pitch_1:
                        x_pitch_1 = x_pitch_2
                    if average_y_length < y_pitch_2 and y_pitch_2 < y_pitch_1:
                        y_pitch_1 = y_pitch_2
        # print(x_pitch_1)
        x_min = 9999
        x_max = 0
        for i in range(len(pin)):
            if pin[i][0] < x_min:
                x_min = pin[i][0]
            if pin[i][2] > x_max:
                x_max = pin[i][2]
        y_min = 9999
        y_max = 0
        for i in range(len(pin)):
            if pin[i][1] < y_min:
                y_min = pin[i][1]
            if pin[i][3] > y_max:
                y_max = pin[i][3]
        pin_x_num = (x_max - x_min) / x_pitch_1
        pin_x_num = round(pin_x_num)
        pin_y_num = (y_max - y_min) / y_pitch_1
        pin_y_num = round(pin_y_num)
        # print(pin_x_num, pin_y_num)

        pin_num_x_y = np.array([pin_x_num, pin_y_num])
        # print(pin_num_x_y)
        pin_num_x_y = pin_num_x_y.astype(int)
        path = r'yolox_data\pin_num.txt'
        # print("[一行的ball数,一列的ball数]", pin_num_x_y)
        np.savetxt(path, pin_num_x_y)

        txt_path = r'yolox_data\pin.txt'
        np.savetxt(txt_path, pin, delimiter=',')

        average_pitch_x_y = np.array([x_pitch_1, y_pitch_1])
        path = r'yolox_data\average_x_y.txt'
        np.savetxt(path, average_pitch_x_y)


##################################################################问题：没有矫正环节，有些误检测的pin的干扰没有消除

def get_np_array_in_txt(file_path):  # 提取txt中保存的数组，要求：浮点数且用逗号隔开
    """读取 txt 中保存的 numpy 数组。"""

    with open(file_path) as f:
        line = f.readline()
        data_array = []
        while line:
            # num = list(map(int, line.split(',')))
            num = list(map(float, line.split(',')))
            data_array.append(num)
            line = f.readline()
        data_array = np.array(data_array)

    # print(data_array[0][:])
    # print('*' * 50)
    # print(data_array)
    return data_array


def crop_img_save(path_img, path_crop, x_min, y_min, x_max, y_max):
    """按照坐标裁剪图像并落盘。"""
    img = cv2.imread(path_img)
    if y_min < 0:
        y_min = 1
    cropped = img[y_min:y_max, x_min:x_max]  # 裁剪坐标为[y0:y1, x0:x1]必须为整数
    cv2.imwrite(path_crop, cropped)
    print("保存图", path_crop)


def empty_folder(folder_path):
    """清空文件夹中的临时文件。"""
    try:
        shutil.rmtree(folder_path)
    except FileNotFoundError:
        print(f"文件夹 {folder_path} 不存在！无须删除文件夹")


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
    # 当模型检测pad时
    if cls.any():
        bboxes_np = np.array(bboxes)
        location = np.zeros((0, 4))
        for i in range(len(cls)):
            if cls[i] == 3:
                num += 1
                location = np.r_[location, [bboxes_np[i]]]


def begain_output_pin_num(img_path, conf):
    """封装入口：推理并返回 PIN 数量及相关信息。"""
    onnx_inference(img_path, conf)


def show_lost_pin_when_full(pin, pin_num_x, pin_num_y, average_x_pitch, average_y_pitch, pin_x, pin_y):
    """调试函数：展示缺失 PIN 的推断。"""
    #***************old
    pin_map = np.zeros((pin_num_y, pin_num_x), dtype=int)
    min_x = 9999999
    min_y = 9999999
    # 找pin中心点的坐标最小值
    min_x_no = -1
    min_y_no = -1
    for i in range(len(pin)):
        if (pin[i][0] + pin[i][2]) * 0.5 < min_x:
            min_x = (pin[i][0] + pin[i][2]) * 0.5
            min_x_no = i
        if (pin[i][1] + pin[i][3]) * 0.5 < min_y:
            min_y = (pin[i][1] + pin[i][3]) * 0.5
            min_y_no = i
    min_x = pin_x
    min_y = pin_y
    standard_ratio = 0.5  # 能接受的偏离程度
    for i in range(len(pin)):
        x = y = -1
        test_x = round(((pin[i][0] + pin[i][2]) * 0.5 - min_x) / average_x_pitch)
        test_y = round(((pin[i][1] + pin[i][3]) * 0.5 - min_y) / average_y_pitch)
        if abs((pin[i][0] + pin[i][2]) * 0.5 - min_x - test_x * average_x_pitch) / average_x_pitch < standard_ratio:
            x = round(((pin[i][0] + pin[i][2]) * 0.5 - min_x) / average_x_pitch)
        if abs((pin[i][1] + pin[i][3]) * 0.5 - min_y - test_y * average_y_pitch) / average_y_pitch < standard_ratio:
            y = round(((pin[i][1] + pin[i][3]) * 0.5 - min_y) / average_y_pitch)
        if x != -1 and y != -1 and x < pin_num_x and y < pin_num_y:
            pin_map[y][x] = 1


    #**********************old
    #*****************new
    # '''
    # pin按照行分为一组，列分为一组，再判断每组是第几行或者第几列
    # '''
    # lie_pin = 0  # 几列
    # key = np.empty((0,), dtype=int)  # 存储已经被分类成列的pin的序号
    # lie_set = np.full((99, 99), -1)  # 存储列pin
    # lie_set_key = 0
    #
    # # 如果两个pin在横坐标上重合的距离在两边占比都超过50%
    # for i in range(len(pin)):  # 将相同列的pin序号排在一起
    #     if i not in key:
    #         lie_set_set_key = 0
    #         key = np.append(key, i)
    #         lie_set[lie_set_key][lie_set_set_key] = i
    #
    #         lie_set_set_key += 1
    #         for j in range(len(pin)):
    #             if j not in key and j != i:
    #                 if max(pin[i][2], pin[j][2]) - min(pin[i][0], pin[j][0]) < pin[i][2] + pin[j][2] - pin[i][0] - \
    #                         pin[j][0]:
    #                     a = max(pin[i][2], pin[j][2]) - min(pin[i][0], pin[j][0])
    #                     b = pin[i][2] + pin[j][2] - pin[i][0] - pin[j][0]
    #                     if (b - a) / (pin[i][2] - pin[i][0]) > 0.5 and (b - a) / (pin[j][2] - pin[j][0]) > 0.5:
    #                         key = np.append(key, j)
    #                         lie_set[lie_set_key][lie_set_set_key] = j
    #                         lie_set_set_key += 1
    #         lie_set_key += 1
    #
    # hang_pin = 0  # 几行
    # key = np.empty((0,), dtype=int)  # 存储已经被分类成行的pin的序号
    # hang_set = np.full((99, 99), -1)  # 存储行pin
    # hang_set_key = 0
    #
    # for i in range(len(pin)):  # 将相同行的pin序号排在一起
    #     if i not in key:
    #         hang_set_set_key = 0
    #         key = np.append(key, i)
    #         # y_stand = (pin[i][1] + pin[i][3]) * 0.5
    #         hang_set[hang_set_key][hang_set_set_key] = i
    #
    #         hang_set_set_key += 1
    #         for j in range(len(pin)):
    #             if j not in key and j != i:
    #                 if max(pin[i][3], pin[j][3]) - min(pin[i][1], pin[j][1]) < pin[i][3] + pin[j][3] - pin[i][1] - \
    #                         pin[j][1]:
    #                     a = max(pin[i][3], pin[j][3]) - min(pin[i][1], pin[j][1])
    #                     b = pin[i][3] + pin[j][3] - pin[i][1] - pin[j][1]
    #                     if (b - a) / (pin[i][3] - pin[i][1]) > 0.5 and (b - a) / (pin[j][3] - pin[j][1]) > 0.5:
    #                         # y_stand_j = (pin[j][1] + pin[j][3]) * 0.5
    #                         # if abs(y_stand - y_stand_j) < average_y_length:
    #                         key = np.append(key, j)
    #                         hang_set[hang_set_key][hang_set_set_key] = j
    #                         hang_set_set_key += 1
    #         hang_set_key += 1
    #
    # # 找第一列是哪一列
    # lie = np.full(99, -1)  # 记录实际每列在lie_set中的哪一列
    # hang = np.full(99, -1)
    # min_x_no = -1
    # min_y_no = -1
    # min_x = 9999999
    # min_y = 9999999
    # for i in range(len(pin)):
    #     if pin[i][0] < min_x:
    #         min_x = pin[i][0]
    #         min_x_no = i
    #     if pin[i][1] < min_y:
    #         min_y = pin[i][1]
    #         min_y_no = i
    # min_x = pin_x
    # min_y = pin_y
    # if min_x_no != -1:
    #     for i in range(len(lie_set)):
    #         for j in range(len(lie_set)):
    #             if lie_set[i][j] == min_x_no:
    #                 lie[0] = i
    #                 lie_0 = i
    # if min_y_no != -1:
    #     for i in range(len(hang_set)):
    #         for j in range(len(hang_set)):
    #             if hang_set[i][j] == min_y_no:
    #                 hang[0] = i
    #                 hang_0 = i
    # # 顺序寻找之后的2、3、4、5...列和行是lie_set中的哪一组
    # lie_finded = 0  # 记录已经找到第几列
    # lie_to_find = 1  # 记录正在找第几列
    # pin_0 = min_x
    # get_key = 1  # 记录是否找到下一列
    # for j in range(len(lie_set) + 99):
    #     lie_finded += 1
    #     lie_to_find += 1
    #     if get_key != 1:
    #         pin_0 = pin_0 + average_x_pitch
    #     for i in range(len(lie_set)):
    #         get_key = 0  # 记录是否找到下一列
    #         if i != lie_0 and i not in lie and lie_set[i][0] != -1:
    #             if average_x_pitch < (pin[lie_set[i][0]][0] - pin_0) < 2 * average_x_pitch:
    #                 lie[lie_to_find - 1] = i
    #                 pin_0 = pin[lie_set[i][0]][0]
    #                 get_key = 1
    #                 break
    # print("lie",lie)
    # # 顺序寻找之后的2、3、4、5...列和行是lie_set中的哪一组
    # hang_finded = 0  # 记录已经找到第几列
    # hang_to_find = 1  # 记录正在找第几列
    # pin_0 = min_y
    # get_key = 1  # 记录是否找到下一列
    # for j in range(len(hang_set) + 99):
    #     hang_finded += 1
    #     hang_to_find += 1
    #     if get_key != 1:
    #         pin_0 = pin_0 + average_y_pitch
    #     for i in range(len(hang_set)):
    #         get_key = 0  # 记录是否找到下一列
    #         if i != hang_0 and i not in hang and hang_set[i][0] != -1:
    #             if average_y_pitch < (pin[hang_set[i][0]][1] - pin_0) < 2 * average_y_pitch:
    #                 hang[hang_to_find - 1] = i
    #                 pin_0 = pin[hang_set[i][0]][0]
    #                 get_key = 1
    #                 break
    # print("hang", hang)
    #
    # pin_map = np.zeros((pin_num_y, pin_num_x), dtype=int)
    # for i in range(len(pin)):
    #     for j in range(len(hang_set)):
    #         for l in range(len(hang)):
    #             if hang[l] == j:
    #                 x = l
    #         for k in range(len(hang_set[i])):
    #             if hang_set[j][k] == i:
    #                 x1 = x
    #     for j in range(len(lie_set)):
    #         for l in range(len(lie)):
    #             if lie[l] == j:
    #                 y = l
    #         for k in range(len(lie_set[i])):
    #             if lie_set[j][k] == i:
    #                 y1 = y
    #     pin[x1,y1] = 1
    # print("pin", pin)





    #*****************new
    print("################输出bottom视图###############")
    print("pin存在显示'o',不存在以位置信息代替")

    print('   ', end='')
    for i in range(len(pin_map[0])):
        print(i + 1, end='   ')
    print()

    for i in range(len(pin_map)):
        print(chr(65 + i), end='   ')
        for j in range(len(pin_map[i])):
            if pin_map[i][j] == 1:
                print("o", end='   ')
            if pin_map[i][j] == 0:
                if (j + 1) > 9:
                    print(chr(65 + i), j + 1, end=' ', sep='')
                if (j + 1) < 10:
                    print(chr(65 + i), j + 1, end='  ', sep='')
        print()

    return pin_map


def when_pin_num_big_15():
    """处理引脚数大于 15 的特殊逻辑。"""
    show_img_key = 0  # 是否显示过程图片，用来检查错误
    test_no = 0  # test图的编号
    no_key = 10  # 每次检测的最大行列数

    # 读取yolox检测的pin行列数
    pin_num_txt = r'yolox_data\pin_num.txt'
    pin_num_x_y = get_np_array_in_txt(pin_num_txt)
    pin_x_num = int(pin_num_x_y[0][0])  # 行pin数
    pin_y_num = int(pin_num_x_y[1][0])  # 列pin数
    if pin_x_num == 0:
        pin_x_num = pin_y_num
    if pin_y_num == 0:
        pin_y_num = pin_x_num

    pin_x_origin = pin_x_num
    pin_y_origin = pin_y_num

    pin_txt = r'yolox_data\pin.txt'
    pin = get_np_array_in_txt(pin_txt)

    # 寻找图中所有pin的大概范围，以防yolox在没有pin的区域硬生生识别出pin
    pin_map_x_min = 99999
    pin_map_y_min = 99999
    pin_map_x_max = 0
    pin_map_y_max = 0
    for i in range(len(pin)):
        if pin[i][0] < pin_map_x_min:
            pin_map_x_min = int(pin[i][0])
        if pin[i][1] < pin_map_y_min:
            pin_map_y_min = int(pin[i][1])
        if pin[i][2] > pin_map_x_max:
            pin_map_x_max = int(pin[i][2])
        if pin[i][3] > pin_map_y_max:
            pin_map_y_max = int(pin[i][3])

    x_min = 9999
    y_min = 9999
    x_min_last = 0
    x_max_last = 0
    y_min_last = 0
    y_max_last = 0
    conf = 0.35  # yolox检测精度
    error_inter_huanghang = 0

    # 首先根据规定的no_key划分一个初始区域test0
    while (pin_x_num > no_key or pin_y_num > no_key):  # 当检测行列数大于12则需要根据pin存在区域进行分割截图之后分别用yolox检测

        pin_txt = r'yolox_data\pin.txt'
        pin = get_np_array_in_txt(pin_txt)
        average_pitch_x_y_txt = r'yolox_data\average_x_y.txt'
        average_pitch_x_y = get_np_array_in_txt(average_pitch_x_y_txt)
        average_pitch_x = average_pitch_x_y[0][0]
        average_pitch_y = average_pitch_x_y[1][0]
        # 求平均pin直径
        pin_x = 0
        pin_y = 0
        for i in range(len(pin)):
            pin_x = pin_x + pin[i][2] - pin[i][0]
            pin_y = pin_y + pin[i][3] - pin[i][1]
        pin_x = pin_x / len(pin)
        pin_y = pin_y / len(pin)
        weight = (no_key - 1) * average_pitch_x + pin_x  # 作为参考的裁剪的尺寸宽
        length = (no_key - 1) * average_pitch_y + pin_y  # 作为参考的裁剪的尺寸长
        # 找pin中的左上角位置，然后根据weight和length找右上角pin与右下角pin
        # 找左上角裁剪位置
        # x_min = 9999
        # y_min = 9999
        for i in range(len(pin)):
            if pin[i][0] < x_min:
                x_min = pin[i][0]
            if pin[i][1] < y_min:
                y_min = pin[i][1]
        # 找右上角裁剪位置
        x_max = x_min + weight
        for i in range(len(pin)):
            if pin[i][0] < x_max < pin[i][2]:
                x_max = pin[i][2]
                break
        # 找右下角位置
        y_max = y_min + length
        for i in range(len(pin)):
            if pin[i][1] < y_max < pin[i][3]:
                y_max = pin[i][3]
                break
        # 裁剪图片并保存
        # path_img = r"data/bottom.jpg"
        path_img = r"data_bottom_crop/pinmap.jpg"
        path_crop = r"data_bottom_crop/test" + str(test_no) + ".jpg"
        test_no = test_no + 1
        x_min = int(x_min)
        x_max = int(x_max)
        y_min = int(y_min)
        y_max = int(y_max)
        if x_min_last != 0 or y_min_last != 0:
            x_min = x_min_last + x_min
            x_max = x_min_last + x_max
            y_min = y_min_last + y_min
            y_max = y_min_last + y_max
        crop_img_save(path_img, path_crop, max(x_min + x_min_last, pin_map_x_min),
                      max(pin_map_y_min, y_min + y_min_last), min(pin_map_x_max, x_max), min(pin_map_y_max, y_max))
        #################################
        if show_img_key == 1:
            xxxx = cv2.imread(path_crop)
            print(1)
            cv2.imshow(path_crop, xxxx)
            cv2.waitKey(0)
        #################################
        # print("保存了分割的图",path_crop)
        x_min_last = x_min
        x_max_last = x_max
        y_min_last = y_min
        y_max_last = y_max
        # yolox检测pin行列数看是否大于12，大于则再裁剪，小于则按照这个裁剪标准对整个pin存在区域进行裁剪然后识别pin之后将pin的坐标拼接
        # img_path = r"data_bottom_crop/test.jpg"
        img_path = path_crop
        print("*******yolox检测，用来检测test图，以分割出规定行列数的test*******")
        begain_output_pin_num(img_path, conf)
        print("**************")
        # 得到pin行列数，用于循环
        pin_num_txt = r'yolox_data\pin_num.txt'
        pin_num_x_y = get_np_array_in_txt(pin_num_txt)
        pin_x_num = int(pin_num_x_y[0][0])  # 行pin数
        pin_y_num = int(pin_num_x_y[1][0])  # 列pin数
    # 循环裁剪下所有pin区域，直到已经将(x_min,y_min),到(x_max,y_max)区域裁剪完
    if pin_x_origin > no_key or pin_y_origin > no_key:
        x_key = 0
        y_key = 0
        crop_map = np.zeros((pin_y_origin, pin_x_origin))  # 表示裁剪的图片占原来的图片的位置
        pin_map_whole = np.zeros((4 * pin_x_origin, 4 * pin_y_origin))  # 表示最终呈现的pin图，先扩大再裁剪
        pin_map_x_min_last = 0
        pin_map_x_max_last = 0
        pin_map_y_min_last = 0
        pin_map_y_max_last = 0
        pin_map_lie_limate = 0  # pin列的最边缘位置
        pin_map_hang_limate = 0  # pin行的最边缘位置
        map_no = np.array([0, 0])  # 表示裁剪的图所在的位置

        while not (x_key == 1 and y_key == 1):  # x_key = 0 表示
            crop_map[map_no[0]][map_no[1]] = 1
            # 裁剪图，根据位置命名，yolox检测然后输出该位置的pin位置，推理出下一张裁剪图的位置
            # 裁剪图
            # path_img = r"data/bottom.jpg"
            img_path = r"data_bottom_crop/pinmap.jpg"
            path_crop = r"data_bottom_crop/" + str(int(map_no[0])) + "," + str(int(map_no[1])) + ".jpg"
            # crop_img_save(path_img, path_crop, x_min, y_min, x_max, y_max)
            try:
                crop_img_save(path_img, path_crop, max(x_min, pin_map_x_min), max(pin_map_y_min, y_min),
                              min(pin_map_x_max, x_max), min(pin_map_y_max, y_max))
            except:
                print("过程错误，推测已经检测完毕，直接输出pinmap")
                break
            if map_no[1] == 0:
                lie_x_min_same = max(x_min, pin_map_x_min)  # 这一列的切割的图片的x_min坐标一样，保证换行拼接的时候正确
                lie_x_max_same = min(pin_map_x_max, x_max)  #
            hang_y_min_same = max(pin_map_y_min, y_min)  # 这一行的切割的图片的y_min坐标一样，保证拼接的时候正确
            hang_y_max_same = min(pin_map_y_max, y_max)  # 这一行的切割的图片的y_max坐标一样，保证拼接的时候正确
            # yolox检测
            conf = 0.6
            print("**********yolox检测，用来从（number，number）图中输出pin位置************")
            try:
                begain_output_pin_num(path_crop, conf)
            except:
                print("过程错误，推测已经检测完毕，直接输出pinmap")
                break
            print("******************************")
            # 输出pin位置
            ###########################################
            pin_txt = r'yolox_data\pin.txt'
            pin = get_np_array_in_txt(pin_txt)
            pin_num_txt = r'yolox_data\pin_num.txt'
            pin_num_x_y = get_np_array_in_txt(pin_num_txt)
            try:  # 报错因为yolox面对没有检测的目标时索性不会执行检测程序，导致输出的pin_x_num根本没有输出
                pin_num_x = int(pin_num_x_y[0][0])  # 行pin数
                pin_num_y = int(pin_num_x_y[1][0])  # 列pin数
            except:
                pin_num_x = no_key
                pin_num_y = no_key
            average_pitch_x_y_txt = r'yolox_data\average_x_y.txt'
            average_pitch_x_y = get_np_array_in_txt(average_pitch_x_y_txt)
            average_x_pitch = average_pitch_x_y[0][0]
            average_y_pitch = average_pitch_x_y[1][0]
            # 针对局部缺整行整列pin情况，对pin_map_part进行补全

            if pin_num_x == 0 and pin_num_y != 0:
                # 求平均pin直径
                pin_x = 0
                for i in range(len(pin)):
                    pin_x = pin_x + pin[i][2] - pin[i][0]
                pin_x = pin_x / len(pin)
                ratio = 0.5

                image = cv2.imread(path_crop)
                weight = image.shape[1]
                for j in range(1, no_key * 2):
                    if abs(weight - ((j - 1) * average_x_pitch + pin_x)) / average_x_pitch < ratio:
                        pin_num_x = j
            # 边缘缺整行列pin需要补齐，中间缺不需要补齐

            #############################################
            if pin_num_x != 0 and pin_num_y == 0:
                # 求平均pin直径
                pin_y = 0
                for i in range(len(pin)):
                    pin_y = pin_y + pin[i][3] - pin[i][1]
                pin_y = pin_y / len(pin)
                ratio = 0.5

                image = cv2.imread(path_crop)
                length = image.shape[0]
                for j in range(1, no_key * 2):
                    if abs(length - ((j - 1) * average_y_pitch + pin_y)) / average_y_pitch < ratio:
                        pin_num_y = j
            print("未修正前", pin_num_x, pin_num_y)
            # 对行列数再确定一下，防止最后一行一列会有缺整行，整列的情况
            if len(pin) != 0:  # 如果完全空白分割图，直接输出10行10列空白
                pin_x = 0
                pin_y = 0
                for i in range(len(pin)):
                    pin_x = pin_x + pin[i][2] - pin[i][0]
                    pin_y = pin_y + pin[i][3] - pin[i][1]

                pin_x = pin_x / len(pin)
                pin_y = pin_y / len(pin)
                ratio = 0.5
                image = cv2.imread(path_crop)
                weight = image.shape[1]
                length = image.shape[0]
                for j in range(1, no_key * 2):
                    if abs(weight - ((j - 1) * average_x_pitch + pin_x)) / average_x_pitch < ratio:
                        pin_num_x = j
                for j in range(1, no_key * 2):
                    if abs(length - ((j - 1) * average_y_pitch + pin_y)) / average_y_pitch < ratio:
                        pin_num_y = j
                if pin_num_x > no_key:
                    pin_num_x = no_key
                if pin_num_y > no_key:
                    pin_num_y = no_key
            print("修正后pin行列数", pin_num_x, pin_num_y)
            pin_map_part = show_lost_pin_when_full(pin, pin_num_x, pin_num_y, average_x_pitch, average_y_pitch, pin_x,
                                                   pin_y)

            if map_no[1] == 0 and map_no[0] == 0:
                pin_map_lie_min = 0
                pin_map_lie_max = pin_map_part.shape[1]
                pin_map_hang_min = 0
                pin_map_hang_max = pin_map_part.shape[0]
            else:

                pin_map_lie_min = pin_map_lie_max_last
                pin_map_hang_min = pin_map_hang_min_last
                pin_map_lie_max = pin_map_lie_min + pin_map_part.shape[1]  # 终止列数 = 起始列数 + 此次分割图中的列数
                pin_map_hang_max = pin_map_hang_min + pin_map_part.shape[0]  # 终止行数 = 起始行数 + 此次分割图中的行数

            if map_no[1] == 0 and map_no[0] != 0:
                pin_map_lie_min = 0
                pin_map_hang_min = pin_map_hang_max_last
                pin_map_lie_max = pin_map_lie_min + pin_map_part.shape[1]  # 终止列数 = 起始列数 + 此次分割图中的列数
                pin_map_hang_max = pin_map_hang_min + pin_map_part.shape[0]  # 终止行数 = 起始行数 + 此次分割图中的行数

            pin_map_whole[pin_map_hang_min:pin_map_hang_max, pin_map_lie_min:pin_map_lie_max] = pin_map_part  # pin图拼接
            if pin_map_lie_limate < pin_map_lie_max:
                pin_map_lie_limate = pin_map_lie_max
            if pin_map_hang_limate < pin_map_hang_max:
                pin_map_hang_limate = pin_map_hang_max

            pin_map_lie_max_last = pin_map_lie_max
            pin_map_lie_min_last = pin_map_lie_min
            pin_map_hang_min_last = pin_map_hang_min
            pin_map_hang_max_last = pin_map_hang_max

            ###########################################
            # 推理下一张裁剪图位置:首先按照上一张图片的右侧裁剪长相同，宽直到边缘的图再用yolox判断是否行列数多于10，多余则裁剪到10为止，如果为0则需要换行
            img_bottom = cv2.imread(path_img)
            y_img_max, x_img_max = img_bottom.shape[:2]
            # x_img_max = img_weight_length_tongdao[0]#bottom.jpg最右侧x坐标
            # y_img_max = img_weight_length_tongdao[1]#bottom.jpg最下侧y坐标
            # 裁剪上一张裁剪图右侧图
            path_crop = r"data_bottom_crop/test" + str(test_no) + ".jpg"
            test_no = test_no + 1
            # crop_img_save(path_img, path_crop, x_max, y_min, x_img_max, y_max)
            try:
                # 不能确定正确执行的代码
                # crop_img_save(path_img, path_crop, max(x_max, pin_map_x_min), max(pin_map_y_min, y_min),
                #               min(pin_map_x_max, x_img_max), min(pin_map_y_max, y_max))
                crop_img_save(path_img, path_crop, max(x_max, pin_map_x_min), pin_map_y_min,
                              min(pin_map_x_max, x_img_max), pin_map_y_max)

                x_max_last = x_max
                y_min_last = y_min
                ##################################
                if show_img_key == 1:
                    xxxx = cv2.imread(path_crop)
                    print("裁剪的上一张图的右侧图")
                    # print("保存了分割的图", path_crop)
                    cv2.imshow(path_crop, xxxx)
                    cv2.waitKey(0)
                ##################################
                print("*****************yolox检测,看右侧图的pin行列数是否满足要求**************")
                conf = 0.35
                begain_output_pin_num(path_crop, conf)
                print("********************")
                if min(pin_map_x_max, x_img_max) - max(x_max, pin_map_x_min) < pin_x:  # 如果裁剪的图已经宽度比pin直径还小则认为该换行
                    crop_img_save(path_img, path_crop, min(pin_map_x_max, x_img_max), pin_map_y_min,
                                  max(x_max, pin_map_x_min), pin_map_y_max)  # 主动触发异常，运行换行程序
            except:
                try:
                    # crop_img_save(path_img, path_crop, pin_map_x_min, max(pin_map_y_min, y_max),
                    #               pin_map_x_max, pin_map_y_max)
                    crop_img_save(path_img, path_crop, pin_map_x_min, max(pin_map_y_min, y_max),
                                  pin_map_x_max, pin_map_y_max)
                    x_max_last = pin_map_x_max
                    y_min_last = y_max
                    error_inter_huanghang = 1
                    map_no[0] += 1
                    map_no[1] = -1

                    ##################################
                    if show_img_key == 1:
                        xxxx = cv2.imread(path_crop)
                        print("由于裁剪错误，认为该换行，展示下一行的图")
                        # print("保存了分割的图", path_crop)
                        cv2.imshow(path_crop, xxxx)
                        cv2.waitKey(0)
                    ##################################

                    print("*****************yolox检测,看下一行图的pin行列数是否满足要求**************")
                    conf = 0.35
                    begain_output_pin_num(path_crop, conf)
                    print("********************")
                except:
                    print("yolox检测下一行失败，推测已经将所有pin检测完")
                    x_key = 1
                    y_key = 1
                    continue

            # crop_img_save(path_img, path_crop, max(x_max, pin_map_x_min), max(pin_map_y_min, y_min),
            # min(pin_map_x_max, x_img_max), min(pin_map_y_max, y_max))

            # x_max_last = x_max
            # y_min_last = y_min
            # ##################################
            # xxxx = cv2.imread(path_crop)
            # print("裁剪的上一张图的右侧图")
            # cv2.imshow(path_crop, xxxx)
            # cv2.waitKey(0)
            # ##################################
            # yolox检测是否行列数多余12
            # print("*****************yolox检测,看右侧图的pin行列数是否满足要求**************")
            # begain_output_pin_num(path_crop,conf)
            # print("********************")

            # 读取yolox检测的pin行列数
            pin_num_txt = r'yolox_data\pin_num.txt'
            pin_num_x_y = get_np_array_in_txt(pin_num_txt)
            pin_x_num = int(pin_num_x_y[0][0])  # 行pin数
            pin_y_num = int(pin_num_x_y[1][0])  # 列pin数
            if pin_x_num == 0:
                pin_x_num = pin_y_num
            if pin_y_num == 0:
                pin_y_num = pin_x_num
            pin_x_origin = pin_x_num
            pin_y_origin = pin_y_num
            # if pin_x_num > 12 or pin_y_num > 12:#当检测行列数大于12则需要根据pin存在区域进行分割截图之后分别用yolox检测
            huanhang_key = 0  # 0 = 不用换行，1 = 需要换行操作

            # pin_txt = r'yolox_data\pin_num.txt'
            # pin = get_np_array_in_txt(pin_txt)
            xunhuancishu = 1  # 下一个while循环的循环次数计数器
            pin_x_num = no_key + 1  # 无论右侧图检测出pin的行列数，都至少执行一遍下面的循环
            pin_y_num = no_key + 1
            while (pin_x_num > no_key or pin_y_num > no_key):  # 当检测行列数大于12则需要根据pin存在区域进行分割截图之后分别用yolox检测

                pin_txt = r'yolox_data\pin.txt'
                pin = get_np_array_in_txt(pin_txt)
                average_pitch_x_y_txt = r'yolox_data\average_x_y.txt'
                average_pitch_x_y = get_np_array_in_txt(average_pitch_x_y_txt)
                average_pitch_x = average_pitch_x_y[0][0]
                average_pitch_y = average_pitch_x_y[1][0]
                # 求平均pin直径
                pin_x = 0
                pin_y = 0
                for i in range(len(pin)):
                    pin_x = pin_x + pin[i][2] - pin[i][0]
                    pin_y = pin_y + pin[i][3] - pin[i][1]
                pin_x = pin_x / len(pin)
                pin_y = pin_y / len(pin)

                weight = (no_key - 1) * average_pitch_x + pin_x  # 作为参考的裁剪的尺寸宽
                length = (no_key - 1) * average_pitch_y + pin_y  # 作为参考的裁剪的尺寸长
                if pin_x_num > no_key and xunhuancishu > 1:  # 如果第一次裁剪不满足行列数要求，减少裁剪的列数
                    weight_new = weight - (pin_x_num - no_key) * average_x_pitch
                    if weight_new > weight:
                        weight = weight_new

                # 找pin中的左上角位置，然后根据weight和length找右上角pin与右下角pin
                # 找左上角裁剪位置:可能最左侧的一列pin在上一张图的左侧图正好局部缺整列，因此左上角位置需要在整个pin图中找，否则会漏掉这一列
                x_min = 9999
                y_min = 9999
                for i in range(len(pin)):
                    if pin[i][0] < x_min:
                        x_min = pin[i][0]
                    if pin[i][1] < y_min:
                        y_min = pin[i][1]
                xxxx = x_min
                # 找右上角裁剪位置
                x_max = x_min + weight
                for i in range(len(pin)):
                    if pin[i][0] < x_max < pin[i][2]:
                        x_max = pin[i][2]
                        break
                # 找右下角位置
                y_max = y_min + length
                for i in range(len(pin)):
                    if pin[i][1] < y_max < pin[i][3]:
                        y_max = pin[i][3]
                        break
                # 裁剪图片并保存
                # path_img = r"data/bottom.jpg"
                img_path = r"data_bottom_crop/pinmap.jpg"
                path_crop = r"data_bottom_crop/test" + str(test_no) + ".jpg"
                test_no = test_no + 1
                x_min = int(x_min)
                xxxxxxxxx = x_min
                x_max = int(x_max)
                y_min = int(y_min)
                y_max = int(y_max)
                if xunhuancishu == 1 and error_inter_huanghang == 0:
                    x_min = x_max_last + x_min
                    x_max = x_max_last + x_max
                    y_min = y_min_last + y_min
                    y_max = y_min_last + y_max
                    yyy = 1

                if xunhuancishu != 1:
                    x_min = x_min_last + x_min
                    x_max = x_min_last + x_max
                    y_min = y_min_last + y_min
                    y_max = y_min_last + y_max
                    yyy = 2

                if error_inter_huanghang == 1:
                    x_min = x_min + pin_map_x_min
                    y_min = y_min + y_min_last
                    x_max = x_max + pin_map_x_min
                    y_max = y_max + y_min_last
                    error_inter_huanghang = 0
                    yyy = 3
                # crop_img_save(path_img, path_crop, max(x_min, pin_map_x_min), max(pin_map_y_min, y_min),
                #               min(pin_map_x_max, x_max), min(pin_map_y_max, y_max))

                crop_img_save(path_img, path_crop, max(x_min, pin_map_x_min), hang_y_min_same,
                              min(x_max, pin_map_x_max), hang_y_max_same)

                # x_min_last = x_min
                # y_min_last = y_min
                x_min_last = x_min
                y_min_last = hang_y_min_same
                ##################################
                if show_img_key == 1:
                    xxxx = cv2.imread(path_crop)
                    print("针对裁剪的右侧图，循环直到裁剪到规定数量的行列数")
                    # print("保存了分割的图", path_crop)
                    cv2.imshow(path_crop, xxxx)
                    cv2.waitKey(0)
                ##################################
                # yolox检测pin行列数看是否大于12，大于则再裁剪，小于则按照这个裁剪标准对整个pin存在区域进行裁剪然后识别pin之后将pin的坐标拼接
                # img_path = r"data_bottom_crop/test.jpg"

                # yolox检测之前将之前保存的数据清除，避免影响
                pin_num_txt = r'yolox_data\pin_num.txt'
                with open(pin_num_txt, 'a+', encoding='utf-8') as test:
                    test.truncate(0)
                pin_txt = r'yolox_data\pin.txt'
                with open(pin_txt, 'a+', encoding='utf-8') as test:
                    test.truncate(0)

                img_path = path_crop
                print("****************yolox检测，看右侧图行列数是否满足规定*************")
                try:  # 报错因为yolox面对没有检测的目标时报错
                    begain_output_pin_num(img_path, conf)
                except:
                    pin_x_num = no_key
                    pin_y_num = no_key
                print("**************************")
                # 得到pin行列数，用于循环
                pin_num_txt = r'yolox_data\pin_num.txt'
                pin_num_x_y = get_np_array_in_txt(pin_num_txt)
                try:  # 报错因为yolox面对没有检测的目标时索性不会执行检测程序，导致输出的pin_x_num根本没有输出
                    pin_x_num = int(pin_num_x_y[0][0])  # 行pin数
                    pin_y_num = int(pin_num_x_y[1][0])  # 列pin数
                except:
                    pin_x_num = no_key
                    pin_y_num = no_key
                xunhuancishu += 1
                print("用于检测判断是否超过规定pin行列数的行列数", pin_num_x_y)
                if pin_x_num == pin_y_num == 0:  # 识别不到pin则换行
                    huanhang_key = 1
            # 如果裁剪到了小于12的pin图则对裁剪的图用yolox检测
            if huanhang_key == 0:
                map_no[1] = map_no[1] + 1  # 表示裁剪的图所在的位置
            if huanhang_key == 1:
                print("执行换行操作")
                map_no[0] = map_no[0] + 1
                map_no[1] = 0
                x_key = 0  # 表示上一张裁剪图右侧已经没有pin了

                # 将下一行位置推理出来
                # 裁剪出下一行
                path_crop = r"data/bottom.jpg"
                # crop_img_save(path_img, path_crop, pin_map_x_min, y_max, x_img_max, y_img_max)
                try:
                    crop_img_save(path_img, path_crop, max(x_min, pin_map_x_min), max(pin_map_y_min, y_max),
                                  min(pin_map_x_max, x_img_max), min(pin_map_y_max, y_img_max))
                except:
                    print("yolox检测下一行失败，推测已经将所有pin检测完")
                    x_key = 1
                    y_key = 1
                # yolox检测是否行列数多余12
                conf = 0.6
                begain_output_pin_num(path_crop, conf)

                # 读取yolox检测的pin行列数
                pin_num_txt = r'yolox_data\pin_num.txt'
                pin_num_x_y = get_np_array_in_txt(pin_num_txt)
                pin_x_num = int(pin_num_x_y[0][0])  # 行pin数
                pin_y_num = int(pin_num_x_y[1][0])  # 列pin数
                if pin_x_num == pin_y_num == 0:
                    y_key = 0  # 表示上一张截图下侧已经没有pin了
                if not (pin_x_num == pin_y_num == 0):

                    pin_txt = r'yolox_data\pin.txt'
                    pin = get_np_array_in_txt(pin_txt)
                    while (pin_x_num > no_key or pin_y_num > no_key):  # 当检测行列数大于12则需要根据pin存在区域进行分割截图之后分别用yolox检测
                        # pin_txt = r'yolox_data\pin_num.txt'
                        # pin = get_np_array_in_txt(pin_txt)
                        average_pitch_x_y_txt = r'yolox_data\average_x_y.txt'
                        average_pitch_x_y = get_np_array_in_txt(average_pitch_x_y_txt)
                        average_pitch_x = average_pitch_x_y[0][0]
                        average_pitch_y = average_pitch_x_y[1][0]
                        # 求平均pin直径
                        pin_x = 0
                        pin_y = 0
                        for i in range(len(pin)):
                            pin_x = pin_x + pin[i][2] - pin[i][0]
                            pin_y = pin_y + pin[i][3] - pin[i][1]
                        pin_x = pin_x / len(pin)
                        pin_y = pin_y / len(pin)
                        weight = (no_key - 1) * average_pitch_x + pin_x  # 作为参考的裁剪的尺寸宽
                        length = (no_key - 1) * average_pitch_y + pin_y  # 作为参考的裁剪的尺寸长
                        # 找pin中的左上角位置，然后根据weight和length找右上角pin与右下角pin
                        # 找左上角裁剪位置
                        x_min = 9999
                        y_min = 9999
                        for i in range(len(pin)):
                            if pin[i][0] < x_min:
                                x_min = pin[i][0]
                            if pin[i][1] < y_min:
                                y_min = pin[i][1]
                        # 找右上角裁剪位置
                        x_max = x_min + weight
                        for i in range(len(pin)):
                            if pin[i][0] < x_max < pin[i][2]:
                                x_max = pin[i][2]
                                break
                        # 找右下角位置
                        y_max = y_min + length
                        for i in range(len(pin)):
                            if pin[i][1] < y_max < pin[i][3]:
                                y_max = pin[i][3]
                                break
                        # 裁剪图片并保存
                        # path_img = r"data/bottom.jpg"
                        img_path = r"data_bottom_crop/pinmap.jpg"
                        path_crop = r"data_bottom_crop/test" + str(test_no) + ".jpg"
                        test_no = test_no + 1
                        x_min = int(x_min)
                        x_max = int(x_max)
                        y_min = int(y_min)
                        y_max = int(y_max)
                        # crop_img_save(path_img, path_crop, x_min, y_min, x_max, y_max)
                        crop_img_save(path_img, path_crop, max(x_min, pin_map_x_min), max(pin_map_y_min, y_min),
                                      min(pin_map_x_max, x_max), min(pin_map_y_max, y_max))
                        ##################################
                        if show_img_key == 1:
                            xxxx = cv2.imread(path_crop)
                            print(4)
                            cv2.imshow(path_crop, xxxx)
                            cv2.waitKey(0)
                        ##################################
                        # yolox检测pin行列数看是否大于12，大于则再裁剪，小于则按照这个裁剪标准对整个pin存在区域进行裁剪然后识别pin之后将pin的坐标拼接
                        img_path = path_crop
                        begain_output_pin_num(img_path, conf)
                        # 得到pin行列数，用于循环
                        pin_num_txt = r'yolox_data\pin_num.txt'
                        pin_num_x_y = get_np_array_in_txt(pin_num_txt)
                        pin_x_num = int(pin_num_x_y[0][0])  # 行pin数
                        pin_y_num = int(pin_num_x_y[1][0])  # 列pin数

        pin_map = pin_map_whole[0:pin_map_hang_limate, 0:pin_map_lie_limate]  # 分割出完整的pin_map

        pin_num_x_y = np.array([pin_map.shape[0], pin_map.shape[1]])
        pin_num_x_y = pin_num_x_y.astype(int)
        path = r'yolox_data\pin_num.txt'
        print("[一行的ball数,一列的ball数]", pin_num_x_y)
        np.savetxt(path, pin_num_x_y)

        print("################输出bottom视图###############")
        print("pin存在显示'o',不存在以位置信息代替")
        print('     ', end='')
        for i in range(len(pin_map[0])):
            if i <= 8:
                print(i + 1, end='    ')
            if i > 8:
                print(i + 1, end='   ')
        print()
        letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'T', 'U', 'V', 'W',
                  'Y']
        for i in range(len(pin_map)):
            if (i + 1) <= 20:
                print(letter[i], end='    ')
            if 20 <= i:
                print(letter[int(((i + 1) // 20) - 1)] + letter[int(((i + 1) % 20) - 1)], end='   ')
            for j in range(len(pin_map[i])):
                if pin_map[i][j] == 1:
                    print("o", end='    ')
                if pin_map[i][j] == 0 and (i + 1) <= 20:
                    if (j + 1) > 9:
                        print(letter[int(i)], j + 1, end='  ', sep='')
                    if (j + 1) < 10:
                        print(letter[int(i)], j + 1, end='   ', sep='')
                if pin_map[i][j] == 0 and i > 19:
                    if (j + 1) > 9:
                        print(letter[int(((i + 1) // 20) - 1)] + letter[int(((i + 1) % 20) - 1)], j + 1, end=' ',
                              sep='')
                    if (j + 1) < 10:
                        print(letter[int(((i + 1) // 20) - 1)] + letter[int(((i + 1) % 20) - 1)], j + 1, end='  ',
                              sep='')
            print()
    return pin_map


def resize_pinmap(pin_x_num, pin_y_num, filein, fileout):
    """调整 pinmap 图像大小。"""
    img = Image.open(filein)
    # pin排列为10*10的图片分辨率为137最合适，因此将pinmap分辨率调整为max（pin_x_num，pin_y_num）*137/10
    height_1, width_1 = img.size[0], img.size[1]
    width = int(max(pin_x_num, pin_y_num) * 337 / 10)
    height = int(height_1 * width / width_1)
    type = img.format
    out = img.resize((width, height), Image.LANCZOS)
    # 第二个参数：
    # Image.NEAREST ：低质量
    # Image.BILINEAR：双线性
    # Image.BICUBIC ：三次样条插值
    # Image.ANTIALIAS：高质量
    out.save(fileout, type)


def resize_pinmap_200(filein, fileout):
    """将 pinmap 调整为宽度 200。"""
    img = Image.open(filein)
    height_1, width_1 = img.size[0], img.size[1]
    width = int(500)
    height = int(height_1 * width / width_1)
    type = img.format
    out = img.resize((width, height), Image.LANCZOS)
    # 第二个参数：
    # Image.NEAREST ：低质量
    # Image.BILINEAR：双线性
    # Image.BICUBIC ：三次样条插值
    # Image.ANTIALIAS：高质量
    out.save(fileout, type)


def manual_get_boxes(folder_path, save_path, save_name):
    """手动标注检测框以辅助调试。"""
    # 定义文件夹路径和保存路径
    global rect_list
    # 获取文件夹内所有图片文件
    file_list = glob.glob(os.path.join(folder_path, '*.jpg'))
    # 循环处理每张图片
    for i, file_path in enumerate(file_list):
        # 读取图片
        img = cv2.imread(file_path)
        # 获取图片尺寸
        height, width, _ = img.shape
        # 计算缩放比例
        # scale = min(1.0, 1024 / max(height, width))
        scale = 1
        # 缩放图片
        img_resized = cv2.resize(img, None, fx=scale, fy=scale)
        # 创建窗口并显示图片
        cv2.namedWindow('image', 0)
        # cv2.namedWindow('picture', 0)
        cv2.imshow('image', img_resized)
        # 初始化框选区域列表
        rect_list = []
        # 循环框选区域
        while True:
            # 等待用户框选区域
            rect = cv2.selectROI('image', img_resized, False)
            # 计算缩放后的框选区域
            rect_resized = [int(x / scale) for x in rect]
            # 如果没有框选区域，则退出循环
            if rect == (0, 0, 0, 0):
                break
            # 截取选中区域并保存
            crop_img = img[rect_resized[1]:rect_resized[1] + rect_resized[3],
                       rect_resized[0]:rect_resized[0] + rect_resized[2]]
            # crop_zuobiao = np.array([int(rect_resized[1]),int(rect_resized[0]), int(rect_resized[1] + rect_resized[3])
            # , int(rect_resized[0] + rect_resized[2])])
            # crop_zuobiao = np.array([int(rect_resized[1]), int(rect_resized[0]), int( rect_resized[3])
            # , int( rect_resized[2])])
            rect_list.append(rect)
            # print(rect_list)
            cv2.imwrite(os.path.join(save_path, save_name), crop_img)

    selectRec = (rect_list[0][0], rect_list[0][1], rect_list[0][2] + rect_list[0][0], rect_list[0][3] + rect_list[0][1])
    # crop_zuobiao = np.array([rect_resized[1],rect_resized[1] + rect_resized[3],
    #                    rect_resized[0],rect_resized[0] + rect_resized[2]])
    # crop_img_save()
    cv2.destroyAllWindows()  # 关闭弹框

    return selectRec


def begain_output_pin_num_pin_map():
    """封装入口：输出 pinmap 相关资源。"""
    pin_output = 0  # 是否在此py中输出了pin图
    pin_num_key = 15  # 设置pin行列数大于多少采用分割法
    # 保存存pinmap

    # 创建文件夹
    empty_folder('pinmap')
    os.makedirs('pinmap')
    shutil.copy(f'data_bottom_crop/pinmap.jpg', f'pinmap/pinmap.jpg')
    # 2.将pinmap转为4*137*4*137分辨率初步检测pin
    file_path = r"data_bottom_crop/pinmap.jpg"
    ResizeImage_137_4(file_path, file_path)  # 调整图片大小适应yolox
    # yolox检测pin图
    img_path = r"data_bottom_crop/pinmap.jpg"
    conf = 0.3  # yolox的检测精度，第一次用于粗略识别，参数较低，因此可能pin区域会放大
    print("********yolox检测，用来给出pin存在的大概范围*********")
    onnx_inference(img_path, conf)
    print("*********************")
    # 读取yolox检测的pin行列数
    pin_num_txt = r'yolox_data\pin_num.txt'
    pin_num_x_y = get_np_array_in_txt(pin_num_txt)
    pin_x_num = int(pin_num_x_y[0][0])  # 行pin数
    pin_y_num = int(pin_num_x_y[1][0])  # 列pin数
    # 3.pin排列为10*10的图片分辨率为137最合适，因此根据初步pin检测结果将pinmap分辨率调整为max（pin_x_num，pin_y_num）*137/10
    resize_pinmap(pin_x_num, pin_y_num, file_path, file_path)
    # yolox检测pin图
    img_path = r"data_bottom_crop/pinmap.jpg"
    conf = 0.3  # yolox的检测精度，第一次用于粗略识别，参数较低，因此可能pin区域会放大
    print("********yolox检测，用来给出pin存在的大概范围*********")
    onnx_inference(img_path, conf)
    print("*********************")
    # 读取yolox检测的pin行列数
    pin_num_txt = r'yolox_data\pin_num.txt'
    pin_num_x_y = get_np_array_in_txt(pin_num_txt)
    pin_x_num = int(pin_num_x_y[0][0])  # 行pin数
    pin_y_num = int(pin_num_x_y[1][0])  # 列pin数
    if pin_x_num == 0:
        pin_x_num = pin_y_num
    if pin_y_num == 0:
        pin_y_num = pin_x_num
    pin_map = np.zeros((1, 1))
    # 4.根据pin行列数判断是否执行分割识别pin算法
    if pin_x_num > pin_num_key or pin_y_num > pin_num_key:
        # try:
        pin_map = when_pin_num_big_15()
        # except:
        #     print("pinmap输出报错")
        #     pin_map = np.ones((20,20))
        pin_output = 1
    else:
        # 当数量小于15时直接将pinmap分辨率转为200*200，yolox检测pin效果会好一些
        shutil.copy(f'pinmap/pinmap.jpg', f'data_bottom_crop/pinmap.jpg')
        img_path = r"data_bottom_crop/pinmap.jpg"
        onnx_inference(img_path, conf)
        # resize_pinmap_200(img_path, img_path)



    return pin_map, pin_output


if __name__ == '__main__':
    pass
