import json
import os
import sys
import cv2
import time
import math
import copy
import re
import numpy as np
import pyclipper
from shapely.geometry import Polygon
from PIL import Image, ImageDraw
import warnings
import onnxruntime
from package_core.PDF_Processed.ocr.utils.upline import uplineCoordinate, isExistUpline

try:
    from package_core.PackageExtract.yolox_onnx_py.model_paths import ocr_model_path, model_path
except ModuleNotFoundError:  # pragma: no cover - 兼容脚本直接运行
    from pathlib import Path
    def ocr_model_path(*parts):
        return str(Path(__file__).resolve().parents[3] / 'model' / 'ocr_model' / Path(*parts))
    def model_path(*parts):
        return str(Path(__file__).resolve().parents[3] / 'model' / Path(*parts))



warnings.filterwarnings("ignore")


# PalldeOCR 检测模块 需要用到的图片预处理类
class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        img = data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)

        assert isinstance(img,
                          np.ndarray), "invalid input 'img' in NormalizeImage"
        data['image'] = (
                                img.astype('float32') * self.scale - self.mean) / self.std
        return data


class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
        data['image'] = img.transpose((2, 0, 1))
        return data


class KeepKeys(object):
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list


class DetResizeForTest(object):
    def __init__(self, **kwargs):
        super(DetResizeForTest, self).__init__()
        self.resize_type = 0
        self.limit_side_len = kwargs['limit_side_len']
        self.limit_type = kwargs.get('limit_type', 'min')

    def __call__(self, data):
        img = data['image']
        src_h, src_w, _ = img.shape
        img, [ratio_h, ratio_w] = self.resize_image_type0(img)
        data['image'] = img
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def resize_image_type0(self, img):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        limit_side_len = self.limit_side_len
        h, w, _ = img.shape

        # 1. 计算缩放比例（避免比例为0）
        if max(h, w) > limit_side_len:
            ratio = float(limit_side_len) / max(h, w)  # 用max确保不超过限制
        else:
            ratio = 1.0  # 尺寸正常时不缩放

        # 2. 计算resize后尺寸，强制不小于32（避免为0）
        resize_h = int(round(h * ratio))
        resize_w = int(round(w * ratio))
        resize_h = max(resize_h, 32)  # 最小高度32
        resize_w = max(resize_w, 32)  # 最小宽度32
        # 调整为32的整数倍
        resize_h = (resize_h + 31) // 32 * 32  # 向上取整到32的倍数
        resize_w = (resize_w + 31) // 32 * 32

        # 3. 安全执行resize（增加异常捕获）
        try:
            img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            print(f"cv2.resize失败: 原图尺寸={img.shape}, 目标尺寸=({resize_w}, {resize_h}), 错误={e}")
            return None, (None, None)

        # 4. 计算实际缩放比例
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return img, [ratio_h, ratio_w]


### 检测结果后处理过程（得到检测框）
class DBPostProcess(object):
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self,
                 thresh=0.3,
                 box_thresh=0.7,
                 max_candidates=1000,
                 unclip_ratio=2.0,
                 use_dilation=False,
                 **kwargs):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = 3
        self.dilation_kernel = None if not use_dilation else np.array(
            [[1, 1], [1, 1]])

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores

    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self, outs_dict, shape_list):
        pred = outs_dict
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask,
                                                   src_w, src_h)

            boxes_batch.append({'points': boxes})
        return boxes_batch


## 根据推理结果解码识别结果
class process_pred(object):
    def __init__(self, character_dict_path=None, character_type='ch', use_space_char=False):
        self.character_str = ''
        with open(character_dict_path, 'rb') as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip('\n').strip('\r\n')
                self.character_str += line
        if use_space_char:
            self.character_str += ' '
        dict_character = list(self.character_str)

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        result_list = []
        ignored_tokens = [0]
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]:
                        continue
                try:
                    char_list.append(self.character[int(text_index[batch_idx][idx])])
                except IndexError:
                    char_list.append(self.character[int(text_index[batch_idx][idx]) - 1])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def __call__(self, preds, label=None):
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label


class det_rec_functions(object):
    def __init__(self, image, use_large=False):
        self.img = image.copy()
        # 使用统一的路径管理加载模型
        self.small_rec_file = ocr_model_path('onnx_rec', 'rec.onnx')
        self.large_rec_file = 'onnx_rec/ABInet.onnx'  # TODO: 确认这个路径是否正确

        self.det_file = ocr_model_path('onnx_det', '0529det_model.onnx')
        self.onet_det_session = onnxruntime.InferenceSession(self.det_file)
        if use_large:
            self.onet_rec_session = onnxruntime.InferenceSession(self.large_rec_file)
        else:
            self.onet_rec_session = onnxruntime.InferenceSession(self.small_rec_file)
        self.infer_before_process_op, self.det_re_process_op = self.get_process()
        # 使用统一的路径管理加载字典文件
        self.postprocess_op = process_pred(ocr_model_path('ppocr_keys_v1.txt'), 'ch', True)
    ## 图片预处理过程
    def transform(self, data, ops=None):
        """ transform """
        if ops is None:
            ops = []
        for op in ops:
            data = op(data)
            if data is None:
                return None
        return data

    def create_operators(self, op_param_list, global_config=None):
        """ create operators based on the config """
        assert isinstance(op_param_list, list), ('operator config should be a list')
        ops = []
        for operator in op_param_list:
            assert isinstance(operator, dict) and len(operator) == 1, "yaml format error"
            op_name = list(operator)[0]
            param = {} if operator[op_name] is None else operator[op_name]
            if global_config is not None:
                param.update(global_config)
            op = eval(op_name)(**param)
            ops.append(op)
        return ops

    ### 检测框的后处理
    def order_points_clockwise(self, pts):
        """ reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py """
        xSorted = pts[np.argsort(pts[:, 0]), :]
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost
        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    ### 定义图片前处理过程，和检测结果后处理过程
    def get_process(self):
        det_db_thresh = 0.3
        det_db_box_thresh = 0.5
        max_candidates = 2000
        unclip_ratio = 1.6
        use_dilation = True

        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': 2500,
                'limit_type': 'max'
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]

        infer_before_process_op = self.create_operators(pre_process_list)
        det_re_process_op = DBPostProcess(det_db_thresh, det_db_box_thresh, max_candidates, unclip_ratio, use_dilation)
        return infer_before_process_op, det_re_process_op

    def sorted_boxes(self, dt_boxes):
        """ Sort text boxes in order from top to bottom, left to right """
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                    (_boxes[i + 1][0][0] < _boxes[i][0][0]):
                tmp = _boxes[i]
                _boxes[i] = _boxes[i + 1]
                _boxes[i + 1] = tmp
        return _boxes

    ### 图像输入预处理
    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = [int(v) for v in "3, 48, 100".split(",")]
        assert imgC == img.shape[2]

        # 确保 max_wh_ratio 有效
        max_wh_ratio = max(max_wh_ratio, 0.05)

        imgW = int((32 * max_wh_ratio))

        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))

        # --- 【修复代码开始】 ---
        # 强制 resized_w 至少为 1，否则 cv2.resize 会报错或生成空数组
        if resized_w <= 0:
            resized_w = 1
        # --- 【修复代码结束】 ---
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    ## 推理检测图片中的部分（支持传入子图像）
    def get_boxes(self, name, show, image1, input_img=None):
        """
        检测文本框（支持传入子图像，默认用self.img）
        :param input_img: 可选，传入的子图像（BGR格式）
        """
        # 优先使用传入的子图像，否则用初始化时的原图
        img_ori = input_img if input_img is not None else self.img
        img_part = img_ori.copy()
        data_part = {'image': img_part}

        # 预处理（处理可能的转换失败）
        data_part = self.transform(data_part, self.infer_before_process_op)
        if data_part is None:
            print(f"警告：{name} 图像预处理失败，返回空检测框")
            return []
        img_part, shape_part_list = data_part

        # 模型推理
        img_part = np.expand_dims(img_part, axis=0)
        shape_part_list = np.expand_dims(shape_part_list, axis=0)
        inputs_part = {self.onet_det_session.get_inputs()[0].name: img_part}
        try:
            outs_part = self.onet_det_session.run(None, inputs_part)
        except Exception as e:
            print(f"检测模型运行错误 ({name}): {e}")
            return []

        # 后处理得到检测框
        post_res_part = self.det_re_process_op(outs_part[0], shape_part_list)
        dt_boxes_part = post_res_part[0]['points']
        dt_boxes_part = self.filter_tag_det_res(dt_boxes_part, img_ori.shape)
        dt_boxes_part = self.sorted_boxes(dt_boxes_part)

        # 检测框延申（原逻辑保留）
        det_boxs = []
        arr_List = np.array(dt_boxes_part)
        for det_box in arr_List:
            det_box[1:3, 0] += 2
            det_boxs.append(det_box)

        # 显示检测结果（子图像单独保存）
        if show:
            det_lists = []
            for det in det_boxs:
                a_list = [list(d) for d in det]
                det_lists.append(a_list)
            # 转换BGR到RGB（PIL显示需要）
            image_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_rgb)
            draw = ImageDraw.Draw(image)
            # 绘制四边形检测框
            for det_list in det_lists:
                draw.polygon([tuple(p) for p in det_list], outline=(0, 0, 255), width=3)
            # # 保存检测结果
            # if not os.path.exists('det_sign'):
            #     os.makedirs('det_sign')
            # save_path = f'det_sign/{name}_det.png'
            # # if os.path.exists(save_path):
            # #     save_path = f'det_sign/{name}_det_1.png'
            # image.save(save_path)
            # print(f"检测结果保存: {save_path}")

        return det_boxs

    ### 根据bounding box得到单元格图片
    def get_rotate_crop_image(self, img, points):
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))

        # --- 【修复代码开始】 ---
        # 强制最小宽高为1，防止 resize 或 warp 报错
        img_crop_width = max(img_crop_width, 1)
        img_crop_height = max(img_crop_height, 1)
        # --- 【修复代码结束】 ---

        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        return dst_img

    ### 单张图片推理
    def get_img_res(self, onnx_model, img, process_op):
        h, w = img.shape[:2]
        img = self.resize_norm_img(img, w * 1.0 / h)
        img = img[np.newaxis, :]
        inputs = {onnx_model.get_inputs()[0].name: img}
        outs = onnx_model.run(None, inputs)
        result = process_op(outs[0])
        return result

    def preprocess_image_cv2(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (64, 64))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        img_standardized = (img_transposed - mean) / std
        return img_standardized

    ### 识别文本框（原逻辑保留）
    def recognition_img(self, dt_boxes, name, Is_crop):
        img_ori = self.img
        img = img_ori.copy()
        img_list = []
        i = 0
        for box in dt_boxes:
            tmp_box = copy.deepcopy(box)
            img_crop = self.get_rotate_crop_image(img, tmp_box)

            # --- 【修复代码开始】 ---
            # 检查裁剪出来的图片是否有效
            if img_crop is None or img_crop.shape[0] < 1 or img_crop.shape[1] < 1:
                # 如果图片高度或宽度为0，跳过不处理，补一个空图占位或直接忽略
                # 这里为了保持索引对齐，建议放入一个极小的纯白图片，或者直接 continue (取决于你的业务逻辑)
                # 方案A：直接跳过（推荐，能过滤噪点）
                continue

                # 进一步过滤：如果长宽比极度畸变（例如一条细线），也容易导致模型报错
            h, w = img_crop.shape[:2]
            if w == 0 or h == 0:
                continue
            # --- 【修复代码结束】 ---

            if Is_crop:
                crop_dir = f'dataset_crop/{name}'
                if not os.path.exists(crop_dir):
                    os.makedirs(crop_dir)
                crop = Image.fromarray(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
                crop.save(f'{crop_dir}/{name}_{i}.jpg')
                i += 1
            img_list.append(img_crop)

        ## 识别小图片
        results = []
        results_info = []
        for pic in img_list:
            # --- 【修复代码开始】 ---
            # 再次保险：防止处理过程中出现空对象
            if pic is None or pic.size == 0:
                continue

            try:
                res = self.get_img_res(self.onet_rec_session, pic, self.postprocess_op)
            except Exception as e:
                # print(f"识别单张小图失败，跳过。错误信息: {e}, 图片尺寸: {pic.shape}")
                continue
            # --- 【修复代码结束】 ---

            res[0] = list(res[0])

            # 上划线处理（原逻辑保留）
            if isExistUpline(pic) and uplineCoordinate(pic):
                upline_coordinate, cropheight = uplineCoordinate(pic)
                if cropheight < 0.5 * pic.shape[0]:
                    if len(upline_coordinate) == 1:
                        # 边界调整
                        upline_coordinate_left = max(0, upline_coordinate[0][0] - 3)
                        upline_coordinate_right = min(pic.shape[1], upline_coordinate[0][2] + 3)
                        upline_crop = pic[cropheight:, upline_coordinate_left:upline_coordinate_right]

                        # 过滤无效裁剪
                        if upline_crop.shape[0] / upline_crop.shape[1] > 6:
                            res_upline = None
                        else:
                            res_upline = self.get_img_res(self.onet_rec_session, upline_crop, self.postprocess_op)
                        res_ori_upline = self.get_img_res(self.onet_rec_session, pic[cropheight:, :],
                                                          self.postprocess_op)

                        # 合并上划线识别结果
                        if res_upline is not None and res_upline[0][0] == res_ori_upline[0][0]:
                            res[0][0] = ''.join(['!' + char for char in res_upline[0][0]])
                            res[0][0] = re.sub(r'!+', '!', res[0][0])
                        elif res_upline is not None:
                            # 处理左右扩展区域
                            extend_crop_left = pic[cropheight:, :upline_coordinate[0][0]]
                            extend_crop_right = pic[cropheight:, upline_coordinate[0][2]:]
                            res_left = \
                            self.get_img_res(self.onet_rec_session, extend_crop_left, self.postprocess_op)[0][0] if (
                                        extend_crop_left.shape[1] > 0 and extend_crop_left.shape[0] /
                                        extend_crop_left.shape[1] <= 6) else ""
                            res_right = \
                            self.get_img_res(self.onet_rec_session, extend_crop_right, self.postprocess_op)[0][0] if (
                                        extend_crop_right.shape[1] > 0 and extend_crop_right.shape[0] /
                                        extend_crop_right.shape[1] <= 6) else ""
                            res[0][0] = res_left + ''.join(['!' + char for char in res_upline[0][0]]) + res_right
                            res[0][0] = re.sub(r'!+', '!', res[0][0])
                # print("upline:", res[0][0])

            results.append(res[0])
            results_info.append(res)
        return results, results_info






def list_to_tuple(data):
    if isinstance(data, list):
        return tuple(list_to_tuple(item) for item in data)
    else:
        return data


def ONNX_Use(image, name):
    ocr_sys = det_rec_functions(image)
    t1 = time.time()
    dt_boxes = ocr_sys.get_boxes(name, show=True, image1=image)
    t2 = time.time()
    # print(f'此图检测时间:{t2 - t1}')

    # 检测框格式转换
    b = []
    for det_list in dt_boxes:
        a = [[int(d[0]), int(d[1])] for d in det_list]
        b.append(a)
    results, results_info = ocr_sys.recognition_img(dt_boxes, name, Is_crop=False)
    text_list = [res[0][0] for res in results_info]

    # 结果字典整理
    b = list_to_tuple(b)
    text_save = dict(zip(b, text_list))
    write_txt = []
    for key, values in text_save.items():
        txt1 = {"transcription": values, "points": key, "difficult": 'false'}
        write_txt.append(txt1)
    return write_txt


def resize_image(image):
    width, height, c = image.shape
    if width > 2000 and height > 2000:
        new_height = int(height / 2.5)
        new_width = int(width / 2.5)
        return cv2.resize(image, (new_height, new_width), interpolation=cv2.INTER_LINEAR)
    else:
        return image


def Run_onnx(image_path, name):
    image = cv2.imread(image_path)
    if image is None or image.size == 0:
        print(f"错误：无法读取图像 {image_path}，图像为空或路径无效")
        return [], []  # 返回空结果，终止流程
    write = ONNX_Use(image, name)
    texts = [item['transcription'] for item in write]
    boxes = [[list(point) for point in item['points']] for item in write]
    return boxes, texts


def Run_onnx1(image, name):
    write = ONNX_Use(image, name)
    texts = [item['transcription'] for item in write]
    boxes = [[list(point) for point in item['points']] for item in write]
    return boxes, texts



if __name__ == '__main__':
    # 1. 加载图像
    image_path = r'D:\workspace\PackageWizard1.1\Result\Package_view\page\bottom.jpg'  # 替换为你的图像路径
    # image_path = 'imgs/test3/1.png'  # 替换为你的图像路径
    t1=time.time()
    res = Run_onnx(image_path,'t')

    print(res)
    print(time.time()-t1)


