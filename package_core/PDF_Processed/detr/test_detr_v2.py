import os
import cv2
import numpy as np
import onnxruntime as ort
import time
import sys
from pathlib import Path
from tqdm import tqdm
import json
from PIL import Image
from package_core.PDF_Processed.ocr.det_text import Run_onnx1


# ====================== 1. 配置参数 (Centralized Configuration) ======================

class DetectorConfig:
    """
    将所有与模型和推理行为直接相关的配置项集中管理。
    """
    ONNX_PATH ='model/DETR_model/rtdetr_r50vd_best1001_package.onnx'
    INPUT_SIZE = 640
    CONF_THRESHOLD = 0.6
    CLASS_NAMES = ['BGA', 'BOTTOMVIEW', 'DETAIL', 'DFN_SON', 'Detail', 'Form', 'Note', 'Package_title', 'QFN', 'QFP', 'SIDEVIEW',
                   'SOP', 'Side', 'TOPVIEW', 'Top', 'package']


# ====================== 2. 全局常量 (Global Constants) ======================
_PREDEFINED_COLORS = np.array([
    [0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933],
    [0.635, 0.078, 0.184], [0.300, 0.300, 0.300], [0.600, 0.600, 0.600],
    [1.000, 0.000, 0.000], [1.000, 0.500, 0.000], [0.749, 0.749, 0.000],
    [0.000, 1.000, 0.000], [0.000, 0.000, 1.000], [0.667, 0.000, 1.000]
]).astype(np.float32)


# ====================== 3. 核心功能函数 (Core Functions) ======================

def preprocess_image(image: np.ndarray, input_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """对输入的图像进行预处理，以匹配模型要求。"""
    original_height, original_width = image.shape[:2]
    resized_image = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    image_tensor = image_rgb.astype(np.float32) / 255.0
    image_tensor = np.transpose(image_tensor, (2, 0, 1))
    return np.expand_dims(image_tensor, axis=0), (original_height, original_width)


def postprocess_results(outputs: list, original_size: tuple[int, int], input_size: int, conf_threshold: float) -> tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """对模型输出进行后处理，将坐标还原到原始图像尺寸。"""
    labels, boxes, scores = outputs[0][0], outputs[1][0], outputs[2][0]
    mask = scores > conf_threshold
    boxes, scores, labels = boxes[mask], scores[mask], labels[mask]
    if len(boxes) == 0: return np.array([]), np.array([]), np.array([])
    original_height, original_width = original_size
    scale_w, scale_h = original_width / input_size, original_height / input_size
    boxes[:, [0, 2]] *= scale_w
    boxes[:, [1, 3]] *= scale_h
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_height)
    return boxes.astype(np.int32), scores, labels


def visualize_detections(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
                         class_names: list) -> np.ndarray:
    """在图像上绘制检测结果，使用预定义的全局颜色列表。"""
    for i in range(len(boxes)):
        box, label_id, score = boxes[i], int(labels[i]), scores[i]
        color_rgb = _PREDEFINED_COLORS[label_id % len(_PREDEFINED_COLORS)]
        color_bgr = [int(c * 255) for c in reversed(color_rgb)]
        class_name = class_names[label_id] if label_id < len(class_names) else f"ID:{label_id}"
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color_bgr, 2)
        text = f'{class_name}: {score:.2f}'
        font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(image, (box[0], box[1] - text_height - baseline), (box[0] + text_width, box[1]), color_bgr, -1)
        cv2.putText(image, text, (box[0], box[1] - baseline), font, font_scale, (255, 255, 255), thickness)
    return image


# ====================== 4. 主推理类 (Main Inference Class) ======================

class RTDETR_Detector:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.session = ort.InferenceSession(self.config.ONNX_PATH,
                                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        print(f"模型已加载，使用设备: {self.session.get_providers()[0]}")
        self.warm_up()

    def warm_up(self):
        """执行一次虚拟推理以预热GPU。"""
        print("正在进行GPU预热...")
        dummy_input = np.zeros((1, 3, self.config.INPUT_SIZE, self.config.INPUT_SIZE), dtype=np.float32)
        dummy_size = np.array([[self.config.INPUT_SIZE, self.config.INPUT_SIZE]], dtype=np.int64)
        _ = self.session.run(None, {self.input_names[0]: dummy_input, self.input_names[1]: dummy_size})
        print("预热完成。")

    def detect(self, image_path: str, save_result: bool = False, verbose: bool = True):
        """对单个图像进行目标检测。增加verbose参数控制打印输出。"""
        if not os.path.exists(image_path):
            if verbose: print(f"文件未找到: {image_path}")
            return None, None, None, None

        original_image = cv2.imread(image_path)
        if original_image is None:
            if verbose: print(f"无法读取图像: {image_path}")
            return None, None, None, None

        image_tensor, original_size = preprocess_image(original_image, self.config.INPUT_SIZE)
        ort_inputs = {self.input_names[0]: image_tensor,
                      self.input_names[1]: np.array([[self.config.INPUT_SIZE, self.config.INPUT_SIZE]], dtype=np.int64)}
        outputs = self.session.run(None, ort_inputs)
        boxes, scores, labels = postprocess_results(outputs, original_size, self.config.INPUT_SIZE,
                                                    self.config.CONF_THRESHOLD)

        if verbose: print(f"在 '{os.path.basename(image_path)}' 中检测到 {len(boxes)} 个目标。")

        if save_result:
            vis_image = visualize_detections(original_image.copy(), boxes, scores, labels, self.config.CLASS_NAMES)
            save_path = os.path.splitext(image_path)[0] + "_result.png"
            cv2.imwrite(save_path, vis_image)
            if verbose: print(f"结果已保存到: {save_path}")

        return original_image, boxes, scores, labels


# ====================== 5. 批量处理函数 (Batch Processing Function) ======================
# def batch_process_folder(detector: RTDETR_Detector, folder_path: str, save_results: bool = True):
#     """
#     对指定文件夹中的所有图片进行批量检测。
#
#     Args:
#         detector (RTDETR_Detector): 已初始化的检测器对象。
#         folder_path (str): 包含图片的文件夹路径。
#         save_results (bool): 是否为每张图片保存可视化结果。
#     """
#     if not os.path.isdir(folder_path):
#         print(f"错误：提供的路径不是一个有效的目录: {folder_path}")
#         return
#
#     image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
#     image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
#                    f.lower().endswith(image_extensions)]
#
#     if not image_paths:
#         print(f"在目录 '{folder_path}' 中未找到任何支持的图片文件。")
#         return
#
#     print(f"\n开始批量处理目录 '{folder_path}' 中的 {len(image_paths)} 张图片...")
#
#     start_time = time.time()
#
#     # 使用tqdm创建进度条
#     for image_path in tqdm(image_paths, desc="批量检测中"):
#         # 调用detect，但设置verbose=False以保持控制台清洁
#         detector.detect(image_path, save_result=save_results, verbose=False)
#
#     end_time = time.time()
#
#     total_time = end_time - start_time
#     num_images = len(image_paths)
#     avg_time_per_image = total_time / num_images if num_images > 0 else 0
#     fps = 1 / avg_time_per_image if avg_time_per_image > 0 else 0
#
#     # 打印性能总结报告
#     print("\n--- 批量处理完成 ---")
#     print(f"处理图片总数: {num_images}")
#     print(f"总耗时: {total_time:.2f} 秒")
#     print(f"平均每张图片耗时: {avg_time_per_image:.4f} 秒")
#     print(f"等效处理速度 (FPS): {fps:.2f} 帧/秒")
#


def batch_process_folder(detector: RTDETR_Detector, folder_path: str, save_results: bool = True):
    """
    批量检测文件夹图片，记录版面分析/OCR时间，不区分大小写匹配关键字
    """
    # 目录有效性快速判断
    if not os.path.isdir(folder_path):
        print(f"错误：无效目录: {folder_path}")
        return

    # 筛选图片文件（简化推导式）
    img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    img_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                 if f.lower().endswith(img_exts)]

    if not img_paths:
        print(f"目录 '{folder_path}' 无有效图片")
        return

    # 核心配置（合并变量定义）
    target_labels = {"figure_title", "paragraph_title", "text", "table_title"}
    keywords = ["BGA", "DFN", "SON", "QFP", "QFN", "SOP", "FCSP","SOIC",
                "TOPVIEW", "TOP VIEW", "SIDEVIEW", "SIDE VIEW", "DETAIL"]
    keywords_lower = [kw.lower() for kw in keywords]
    results, time_stats = [], []

    print(f"\n开始处理 '{folder_path}' 中的 {len(img_paths)} 张图片...")
    total_start = time.time()

    # 批量处理主逻辑
    for img_path in tqdm(img_paths, desc="检测中"):
        img_name = os.path.basename(img_path)
        img_base = os.path.splitext(img_name)[0]

        # 版面分析计时与执行
        layout_start = time.time()
        orig_img, boxes, _, labels = detector.detect(img_path, save_result=save_results, verbose=False)
        layout_time = time.time() - layout_start
        ocr_total_time = 0.0

        # 无有效检测结果则记录时间并跳过
        if orig_img is None or not boxes.size:
            time_stats.append({
                'img_name': img_name,
                'layout_analysis_time': layout_time,
                'ocr_time': 0.0,
                'total_time': layout_time
            })
            continue

        # 处理每个目标框
        for i, (box, label) in enumerate(zip(boxes, labels)):
            # 筛选目标标签
            label_type = detector.config.CLASS_NAMES[int(label)] if int(label) < len(
                detector.config.CLASS_NAMES) else f"未知标签({int(label)})"
            if label_type not in target_labels:
                continue

            # 坐标处理（简化赋值）
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_img.shape[1], x2), min(orig_img.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # OCR处理与计时
            ocr_start = time.time()
            _, texts = Run_onnx1(orig_img[y1:y2, x1:x2], f"{img_base}_{i}")
            ocr_text = " ".join(filter(None, texts))
            print(ocr_text)
            ocr_total_time += time.time() - ocr_start

            # 大小写不敏感匹配关键字
            ocr_text_lower = ocr_text.lower()
            matched_kw = next((kw for kw, kw_lower in zip(keywords, keywords_lower) if kw_lower in ocr_text_lower),
                              None)
            if matched_kw:
                results.append({
                    'img_name': img_name,
                    'label_type': label_type,
                    'keyword': matched_kw,
                    'coordinates': (x1, y1, x2, y2),
                    'type': 'ocr',
                    'content': ocr_text.strip()
                })

        # 记录当前图片时间统计
        time_stats.append({
            'img_name': img_name,
            'layout_analysis_time': layout_time,
            'ocr_time': ocr_total_time,
            'total_time': layout_time + ocr_total_time
        })

    # 整体性能计算
    total_time = time.time() - total_start
    avg_time = total_time / len(img_paths) if img_paths else 0

    # 结果保存（简化文件操作）
    if results:
        json.dump(results, open(os.path.join(folder_path, 'ocr_keyword_results.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
        print(f"\n已保存OCR结果到: ocr_keyword_results.json")

    json.dump(time_stats, open(os.path.join(folder_path, 'processing_time_stats.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    print(f"已保存时间统计到: processing_time_stats.json")


    return results, time_stats

# ====================== 6. 单张图片处理函数  ======================
def process_single_image(image_path, detector):
    """
    对单张图片进行版面分析，仅返回指定标签区域的坐标和类别信息
    """
    results = []
    target_labels = {"figure_title", "paragraph_title", "table_title"}

    # 执行版面分析
    original_image, boxes, _, labels = detector.detect(image_path, save_result=True, verbose=True)
    if original_image is None or not boxes.size:
        return results
    for i, (box, label) in enumerate(zip(boxes, labels)):
        label_idx = int(label)
        if label_idx >= len(detector.config.CLASS_NAMES):
            continue

        class_name = detector.config.CLASS_NAMES[label_idx]
        if class_name not in target_labels:
            continue

        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(original_image.shape[1], x2), min(original_image.shape[0], y2)

        if x2 > x1 and y2 > y1:
            result = {
                'label_type': class_name,
                'coordinates': (x1, y1, x2, y2)
            }
            results.append(result)

            # print(f"目标 {i + 1} ({class_name}): 坐标 {x1, y1, x2, y2}")

    return results

# ====================== 7. 主程序入口 (Main Execution) ======================

if __name__ == "__main__":
    # 初始化配置和检测器
    config = DetectorConfig()
    detector = RTDETR_Detector(config)

    # # # --- 场景一：测试单张图片并打印详细信息 ---
    image_path = r"D:\20250822\PackageWizard1.0\current_page.png"  # 替换为你的图片路径
    ocr_results = process_single_image(image_path, detector)
    print(ocr_results)



    # print("\n--- 测试单张图片 ---")
    # single_image_path = r"test/1.png"
    # if os.path.exists(single_image_path):
    #     start_time = time.time()
    #     _, boxes, scores, labels = detector.detect(single_image_path, save_result=True, verbose=True)
    #     end_time = time.time()
    #     print("\n--- 检测结果 ---")
    #     if len(boxes) > 0:
    #         for i in range(len(boxes)):
    #             print(f"目标 {i + 1}: "
    #                   f"类别='{config.CLASS_NAMES[labels[i]]}', "
    #                   f"置信度={scores[i]:.4f}, "
    #                   f"边界框={boxes[i]}")
    #     else:
    #         print("未检测到任何目标。")
    #
    #     print(f"\n推理耗时: {end_time - start_time:.4f} 秒")
    # else:
    #     print(f"单张测试图片未找到: {single_image_path}")

    # --- 场景二：批量测试整个文件夹 ---
    # test_folder_path = "1"
    # batch_process_folder(detector, test_folder_path, save_results=True)