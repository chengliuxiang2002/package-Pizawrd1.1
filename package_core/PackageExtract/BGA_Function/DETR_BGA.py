import onnxruntime as ort
from PIL import Image, ImageDraw
import numpy as np
import time
import os
import cv2
from pathlib import Path

try:
    from package_core.PackageExtract.yolox_onnx_py.model_paths import yolo_model_path, result_path
except ModuleNotFoundError:  # pragma: no cover - 兼容脚本直接运行
    from pathlib import Path
    def yolo_model_path(*parts):
        return str(Path(__file__).resolve().parents[2] / 'model' / 'yolo_model' / Path(*parts))
    def result_path(*parts):
        return str(Path(__file__).resolve().parents[3] / 'Result' / Path(*parts))

# 全局路径 - 使用统一的路径管理函数
DATA = result_path('Package_extract', 'data')
DATA_BOTTOM_CROP = result_path('Package_extract', 'data_bottom_crop')
DATA_COPY = result_path('Package_extract', 'data_copy')
ONNX_OUTPUT = result_path('Package_extract', 'onnx_output')
OPENCV_OUTPUT = result_path('Package_extract', 'opencv_output')
OPENCV_OUTPUT_LINE = result_path('Package_extract', 'opencv_output_yinXian')
YOLO_DATA = result_path('Package_extract', 'yolox_data')
DETR_OUTPUT = result_path('Package_extract', 'detr_output')
BGA_BOTTOM = result_path('Package_extract', 'bga_bottom')
PINMAP = result_path('Package_extract', 'pinmap')
YOLOX_DATA = result_path('Package_extract', 'yolox_data')

# 使用统一的路径管理加载模型
model_file = yolo_model_path("package_model", "rtdetr_r50vd_best0521_BGA.onnx")
session = ort.InferenceSession(model_file)

def vis(img, boxes, scores, cls_ids, conf, class_names):
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
        # text = ''
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

def DETR_preprocess(image, input_size=1024):
    """支持PIL Image或OpenCV numpy数组的预处理"""
    # 统一转换为PIL Image
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:  # BGR格式
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:  # 已经是RGB或灰度
            image = Image.fromarray(image)

    # 获取原始尺寸
    orig_w, orig_h = image.size

    # 调整大小并填充
    scale = min(input_size / orig_w, input_size / orig_h)
    new_w, new_h = int(scale * orig_w), int(scale * orig_h)
    image = image.resize((new_w, new_h), Image.BILINEAR)

    new_image = Image.new('RGB', (input_size, input_size), (128, 128, 128))
    new_image.paste(image, ((input_size - new_w) // 2, (input_size - new_h) // 2))

    # 转换为模型需要的格式
    image = np.array(new_image).astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)  # HWC to CHW

    return np.expand_dims(image, axis=0), (orig_h, orig_w)  # 返回 (1,3,H,W) 和原始尺寸(h,w)
def DETR_per_img(output_folder, img_path,classes):
    """
    DETR版本的单个图片检测函数
    """
    # save_img_path = os.path.splitext(img_path)[0] + "_cv.png"
    filename = os.path.basename(img_path)  # 获取原始图片名，例如 '3.png'
    save_img_path = os.path.join(output_folder, filename)
    img_image = Image.open(img_path).convert('RGB')
    origin_img = cv2.cvtColor(np.asarray(img_image), cv2.COLOR_RGB2BGR)
    blob, (h, w) = DETR_preprocess(origin_img)

    input_names = [input.name for input in session.get_inputs()]
    # im = img_image.resize((640, 640))
    im = img_image.resize((1024, 1024))
    # 将图像数据转换为numpy数组并归一化
    im_data = np.array(im, dtype=np.float32) / 255.0
    # 调整维度顺序为CHW并添加batch维度
    im_data = np.transpose(im_data, (2, 0, 1))[np.newaxis, ...]
    # size = np.array([[640, 640]], dtype=np.int64)
    size = np.array([[1024, 1024]], dtype=np.int64)
    orig_size = np.array([[h, w]], dtype=np.int64)  # 修复关键点

    ort_inputs = {
        input_names[0]: im_data,
        input_names[1]: size
    }

    outputs = session.run(None, ort_inputs)
    # labels, boxes, scores = outputs

    boxes, scores, labels = DETR_postprocess(outputs,[h,w])



    if len(boxes) > 0:
        origin_img = vis(origin_img, boxes, scores, labels, 0.6, classes)

    img_rgb = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
    Image.fromarray(img_rgb).save(save_img_path)


    return boxes, scores, labels

def DETR_postprocess(outputs, orig_size, score_thresh=0.6):
    """
    DETR后处理，适用于 boxes 已经是 [xmin, ymin, xmax, ymax] 格式的归一化坐标。

    参数:
        outputs: (labels, boxes, scores)，每项 shape = (num_queries, ...)
        orig_size: (height, width)
        score_thresh: 最低置信度阈值
    返回:
        boxes_xyxy: (N, 4) — 原图上的像素坐标
        scores: (N,)
        labels: (N,)
    """
    labels = outputs[0][0]  # shape: (num_queries,)
    boxes = outputs[1][0]  # shape: (num_queries, 4) in [x0,y0,x1,y1] normalized
    scores = outputs[2][0]  # shape: (num_queries,)

    # 过滤低分数
    keep = scores > score_thresh
    labels = labels[keep]
    boxes = boxes[keep]
    scores = scores[keep]

    # 原图尺寸还原
    orig_h, orig_w = orig_size
    h_rate = orig_h/1024
    w_rate = orig_w/1024
    boxes[:, 0] *= w_rate  # xmin
    boxes[:, 1] *= h_rate  # ymin
    boxes[:, 2] *= w_rate  # xmax
    boxes[:, 3] *= h_rate  # ymax

    return boxes.astype(np.int32), scores, labels

def DETR_BGA(img_path, package_classes):
    if package_classes == 'BGA':
        classes = ['BGA_Border', 'BGA_PIN', 'BGA_serial_letter', 'BGA_serial_number', 'multi_value_1', 'multi_value_2',
                   'multi_value_3', 'multi_value_triangle', 'other', 'pairs_inside_col', 'pairs_inside_row',
                   'pairs_outside_col',
                   'pairs_outside_row']


        # 检查文件夹是否存在
        if not os.path.exists(DETR_OUTPUT):
            os.makedirs(DETR_OUTPUT)  # 创建文件夹
            print("文件夹已创建")
        else:
            print("文件夹已存在")
        output_file = DETR_OUTPUT
        bboxes, scores, cls = DETR_per_img(output_file, img_path, classes)
        # np.set_printoptions(threshold=np.inf)
        # print(bboxes)

        # YOLOX_weight = "DFN_SON_0831.onnx"
        if package_classes == 'BGA':
            YOLOX_weight = 'rtdetr_r50vd_best0521_BGA.onnx'
        weight_path = yolo_model_path('package_model', YOLOX_weight)


        other_num = 0
        numbers_num = 0
        serial_num_num = 0
        pin_num = 0
        pad_num = 0
        border_num = 0
        pairs_num = 0
        angle_pairs_num = 0
        BGA_serial_letter_num = 0
        BGA_serial_num_num = 0
        '''
        0'BGA_Border'
        1'BGA_PIN'
        2'BGA_serial_letter'
        3'BGA_serial_number'
        4'multi_value_1'
        5'multi_value_2'
        6'multi_value_3'
        7'multi_value_triangle'
        8'other'
        9'pairs_inside_col'
        10'pairs_inside_row'
        11'pairs_outside_col'
        12'pairs_outside_row'
        '''
        # 获取两种类型的数量
        for i in range(len(cls)):
            if package_classes == 'BGA':
                if cls[i] == 8:
                    other_num += 1
                if cls[i] == 4 or cls[i] == 5 or cls[i] == 6 or cls[i] == 7:
                    numbers_num += 1
                if cls[i] == 3:
                    BGA_serial_num_num += 1
                if cls[i] == 2:
                    BGA_serial_letter_num += 1
                if cls[i] == 9 or cls[i] == 10 or cls[i] == 11 or cls[i] == 12:
                    pairs_num += 1
                if cls[i] == 0:
                    border_num += 1
                if cls[i] == 1:
                    pin_num += 1



        yolox_num = np.empty((numbers_num, 4))  # [x1,x2,x3,x4]
        yolox_other = np.empty((other_num, 4))  # [x1,x2,x3,x4]
        yolox_serial_num = np.empty((serial_num_num, 4))

        yolox_pin = np.empty((pin_num, 4))
        pad = np.empty((pad_num, 4))
        yolox_border = np.empty((border_num, 4))
        yolox_pairs = np.empty((pairs_num, 5))
        angle_pairs = np.empty((angle_pairs_num, 4))
        yolox_BGA_serial_num = np.empty((BGA_serial_num_num, 4))
        yolox_BGA_serial_letter = np.empty((BGA_serial_letter_num, 4))

        j = 0
        k = 0
        l = 0
        m = 0
        n = 0
        o = 0
        p = 0
        q = 0
        r = 0
        s = 0
        for i in range(len(cls)):
            if package_classes == 'BGA':
                if cls[i] == 8:
                    yolox_other[k][0] = bboxes[i][0]
                    yolox_other[k][1] = bboxes[i][1]
                    yolox_other[k][2] = bboxes[i][2]
                    yolox_other[k][3] = bboxes[i][3]
                    k = k + 1
                if cls[i] == 4 or cls[i] == 5 or cls[i] == 6 or cls[i] == 7:
                    yolox_num[j][0] = bboxes[i][0]
                    yolox_num[j][1] = bboxes[i][1]
                    yolox_num[j][2] = bboxes[i][2]
                    yolox_num[j][3] = bboxes[i][3]
                    j = j + 1
                if cls[i] == 3:
                    yolox_BGA_serial_num[s][0] = bboxes[i][0]
                    yolox_BGA_serial_num[s][1] = bboxes[i][1]
                    yolox_BGA_serial_num[s][2] = bboxes[i][2]
                    yolox_BGA_serial_num[s][3] = bboxes[i][3]
                    s = s + 1
                if cls[i] == 2:
                    yolox_BGA_serial_letter[r][0] = bboxes[i][0]
                    yolox_BGA_serial_letter[r][1] = bboxes[i][1]
                    yolox_BGA_serial_letter[r][2] = bboxes[i][2]
                    yolox_BGA_serial_letter[r][3] = bboxes[i][3]
                    r = r + 1
                if cls[i] == 9 or cls[i] == 10 or cls[i] == 11 or cls[i] == 12:
                    yolox_pairs[p][0] = bboxes[i][0]
                    yolox_pairs[p][1] = bboxes[i][1]
                    yolox_pairs[p][2] = bboxes[i][2]
                    yolox_pairs[p][3] = bboxes[i][3]
                    if cls[i] == 9 or cls[i] == 10:
                        yolox_pairs[p][4] = 1
                    else:
                        yolox_pairs[p][4] = 0
                    p = p + 1
                if cls[i] == 0:
                    yolox_border[o][0] = bboxes[i][0]
                    yolox_border[o][1] = bboxes[i][1]
                    yolox_border[o][2] = bboxes[i][2]
                    yolox_border[o][3] = bboxes[i][3]
                    o = o + 1
                if cls[i] == 1:
                    yolox_pin[m][0] = bboxes[i][0]
                    yolox_pin[m][1] = bboxes[i][1]
                    yolox_pin[m][2] = bboxes[i][2]
                    yolox_pin[m][3] = bboxes[i][3]
                    m = m + 1

    return yolox_pairs, yolox_num, yolox_serial_num, yolox_pin, yolox_other, pad, yolox_border, angle_pairs, yolox_BGA_serial_num, yolox_BGA_serial_letter
