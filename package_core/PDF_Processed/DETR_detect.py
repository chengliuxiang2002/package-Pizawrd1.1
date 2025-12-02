"""使用YOLOv13进行封装图分类识别"""
import re
import fitz
import shutil
import cv2
import numpy as np
import time
import json
import os
import glob
from ultralytics import YOLO
from pathlib import Path

# 导入统一路径管理
try:
    from package_core.PackageExtract.yolox_onnx_py.model_paths import result_path
except ModuleNotFoundError:
    from pathlib import Path
    def result_path(*parts):
        return str(Path(__file__).resolve().parents[2] / 'Result' / Path(*parts))

IMAGE_PATH = result_path('PDF_extract', 'detr_img')
SAVE_IMG_PATH = result_path('PDF_extract', 'detr_result')
ZOOM = (3, 3)

# YOLO模型配置
MODEL_PATH = "model/yolo_model/PDF_processed/best.onnx"  # 修改为你的模型路径
CONF_THRESHOLD = 0.6

# 类别配置（保持与DETR相同）
VOC_CLASSES = ['BGA', 'BOTTOMVIEW', 'DETAIL', 'DFN_SON', 'Detail', 'Form', 'Note', 'Package_title', 'QFN', 'QFP',
               'SIDEVIEW', 'SOP', 'Side', 'TOPVIEW', 'Top', 'package']
DETR_KEYVIEW_CLASSES = ['BGA', 'DFN_SON', 'QFP', 'QFN', 'SOP']
DETR_VIEW_CLASSES = ['Top', 'Side', 'Detail', 'Form']

# 全局数据存储变量（保持不变）
source_data = []
source_package_data = []
source_keyview_data = []
source_Top_data = []
source_Side_data = []
source_Detail_data = []
source_Note_data = []
source_Form_data = []
source_Package_title_data = []
source_TOPVIEW_data = []
source_BOTTOMVIEW_data = []
source_SIDEVIEW_data = []
source_DETAIL_data = []

# 工具函数（保持不变）
def remove_dir(dir_path):
    """删除dir_path文件夹（包括其所有子文件夹及文件）"""
    shutil.rmtree(dir_path)

def create_dir(dir_path):
    """创建dir_path空文件夹（若存在该文件夹则清空该文件夹）"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)

def pdf2img(pdf_path, pages):
    """将pages列表中的页转为图片"""
    create_dir(IMAGE_PATH)
    with fitz.open(pdf_path) as doc:
        for i in range(len(pages)):
            page = doc[pages[i]]
            mat = fitz.Matrix(ZOOM[0], ZOOM[1])
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pix.save(os.path.join(IMAGE_PATH, f"{pages[i] + 1}.png"))

def natural_sort_key(s):
    """使用正则匹配文件名中的数字"""
    int_part = re.search(r'\d+', s).group()
    return int(int_part)

def init_data_storage():
    """初始化数据存储"""
    global source_data, source_package_data, source_keyview_data
    global source_Top_data, source_Side_data, source_Detail_data
    global source_Note_data, source_Form_data, source_Package_title_data
    global source_TOPVIEW_data, source_BOTTOMVIEW_data, source_SIDEVIEW_data, source_DETAIL_data

    source_data = []
    source_package_data = []
    source_keyview_data = []
    source_Top_data = []
    source_Side_data = []
    source_Detail_data = []
    source_Note_data = []
    source_Form_data = []
    source_Package_title_data = []
    source_TOPVIEW_data = []
    source_BOTTOMVIEW_data = []
    source_SIDEVIEW_data = []
    source_DETAIL_data = []


def process_yolov13_detection():
    """
    调用YOLOv13检测（使用Ultralytics接口）
    """
    global source_data, source_package_data, source_keyview_data
    global source_Top_data, source_Side_data, source_Detail_data
    global source_Note_data, source_Form_data, source_Package_title_data
    global source_TOPVIEW_data, source_BOTTOMVIEW_data, source_SIDEVIEW_data, source_DETAIL_data

    yolov13_start = time.time()

    # 创建保存结果的目录
    create_dir(SAVE_IMG_PATH)

    # 检查图片文件夹是否存在
    if not os.path.exists(IMAGE_PATH) or not os.listdir(IMAGE_PATH):
        print(f"❌ 图片文件夹为空或不存在: {IMAGE_PATH}")
        return

    # 加载YOLO模型
    print("🔄 加载YOLOv13模型...")
    model = YOLO(MODEL_PATH)

    # 获取图片列表
    image_paths = glob.glob(os.path.join(IMAGE_PATH, "*.*"))
    image_paths = [p for p in image_paths if p.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff'))]

    # 进行推理
    print("🔄 开始推理...")
    results = model.predict(
        source=IMAGE_PATH,
        conf=CONF_THRESHOLD,
        save=True,
        project=SAVE_IMG_PATH,
        name="predictions",
        exist_ok=True,
        verbose=False
    )

    # 处理检测结果
    for i, result in enumerate(results):
        if not result.boxes:
            continue

        # 获取图片文件名和页码
        img_path = result.path
        img_name = os.path.basename(img_path)
        page_num = int(os.path.splitext(img_name)[0]) - 1  # 转换为0-based页码

        # 获取检测框信息
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for j in range(len(boxes)):
            box = boxes[j]
            score = confidences[j]
            label_id = class_ids[j]

            # 获取类别名称
            class_name = result.names[label_id] if label_id in result.names else f"ID:{label_id}"

            # 过滤置信度小于0.6并且标签名称为package的检测结果
            if class_name == 'package' and score < 0.6:
                continue

            # 调整坐标到原始PDF尺寸（考虑ZOOM缩放）
            adjusted_pos = [
                int(box[0] / ZOOM[0]),
                int(box[1] / ZOOM[1]),
                int(box[2] / ZOOM[0]),
                int(box[3] / ZOOM[1])
            ]

            # 存储检测结果到source_data
            source_data.append({
                'page': page_num,
                'type': class_name,
                'detr_type': class_name,
                'pos': adjusted_pos,
                'conf': float(score)
            })

            # 分类存储不同类型的数据
            store_detection_by_type(page_num, class_name, adjusted_pos, float(score))

    yolov13_end = time.time()
    print(f"✅ YOLOv13检测完成，耗时: {yolov13_end - yolov13_start:.4f}秒")


def store_detection_by_type(page_num, class_name, pos, score):
    """根据类别分类存储检测结果"""
    global source_package_data, source_keyview_data, source_Top_data
    global source_Side_data, source_Detail_data, source_Note_data
    global source_Form_data, source_Package_title_data, source_TOPVIEW_data
    global source_BOTTOMVIEW_data, source_SIDEVIEW_data, source_DETAIL_data

    detection_item = {
        'page': page_num,
        'type': class_name,
        'detr_type': class_name,
        'pos': pos,
        'conf': score,
        'match_state': -1
    }

    if class_name == 'package':
        source_package_data.append(detection_item)
    elif class_name == 'Top':
        source_Top_data.append(detection_item)
    elif class_name == 'Side':
        source_Side_data.append(detection_item)
    elif class_name == 'Detail':
        source_Detail_data.append(detection_item)
    elif class_name == 'Note':
        source_Note_data.append(detection_item)
    elif class_name == 'Form':
        source_Form_data.append(detection_item)
    elif class_name == 'Package_title':
        source_Package_title_data.append(detection_item)
    elif class_name == 'TOPVIEW':
        source_TOPVIEW_data.append(detection_item)
    elif class_name == 'BOTTOMVIEW':
        source_BOTTOMVIEW_data.append(detection_item)
    elif class_name == 'SIDEVIEW':
        source_SIDEVIEW_data.append(detection_item)
    elif class_name == 'DETAIL':
        source_DETAIL_data.append(detection_item)
    elif class_name in DETR_KEYVIEW_CLASSES:  # 关键特征视图
        source_keyview_data.append(detection_item)



def _sort_views_by_position(views):
    """按位置对视图进行排序：先上下后左右"""
    if not views:
        return []

    # 按Y坐标分组（使用整数除法来创建分组）
    groups = {}
    for view in views:
        y_group = view['pos'][1] // 20
        if y_group not in groups:
            groups[y_group] = []
        groups[y_group].append(view)

    # 按Y坐标排序分组（从上到下），然后每组内按X坐标排序（从左到右）
    result = []
    for y_group in sorted(groups.keys()):
        sorted_group = sorted(groups[y_group], key=lambda x: x['pos'][0])
        result.extend(sorted_group)

    return result

def find_pages_with_features_but_no_package():
    """找出特征视图数量与package数量不匹配的页面"""
    global source_data
    page_stats = {}

    # 统计每个页面的元素数量和特征
    for item in source_data:
        page_num = item['page']
        item_type = item['type']

        if page_num not in page_stats:
            page_stats[page_num] = {
                'package_count': 0,
                'keyview_count': 0,
                'view_features_count': 0,
                'has_package': False
            }

        if item_type == 'package':
            page_stats[page_num]['package_count'] += 1
            page_stats[page_num]['has_package'] = True
        elif item_type in DETR_KEYVIEW_CLASSES:
            page_stats[page_num]['keyview_count'] += 1
        elif item_type in DETR_VIEW_CLASSES:
            page_stats[page_num]['view_features_count'] += 1

    target_pages = []

    for page_num, stats in page_stats.items():
        # 如果特征视图数量与package数量不相同，则根据原逻辑判断是否返回页码
        if stats['keyview_count'] != stats['package_count']:
            # 原逻辑：有关键特征视图、有其他视图特征、但没有package标签
            if (stats['keyview_count'] > 0 and
                    stats['view_features_count'] > 1):
                target_pages.append(page_num)

    return sorted(target_pages)


def add_minimum_bounding_boxes_for_target_pages():
    """为目标页面添加最小外接矩形框"""
    global source_data, source_package_data
    # 首先获取target_pages
    target_pages = find_pages_with_features_but_no_package()

    if not target_pages:
        return 0

    page_data = {}

    for page_num in target_pages:
        page_data[page_num] = {
            'all_rects': [],  # 存储所有需要计算外框的矩形
            'keyview_count': 0,
            'top_views': [],
            'keyviews': [],
            'package_count': 0
        }

    for item in source_data:
        page_num = item['page']
        if page_num in target_pages:
            rect = item['pos']
            item_type = item['type']

            # 收集所有需要计算外框的矩形（包括关键特征和视图类）
            if item_type in DETR_KEYVIEW_CLASSES or item_type in DETR_VIEW_CLASSES:
                page_data[page_num]['all_rects'].append(rect)

            # 统计关键特征视图数量和位置
            if item_type in DETR_KEYVIEW_CLASSES:
                page_data[page_num]['keyview_count'] += 1
                page_data[page_num]['keyviews'].append({
                    'type': item_type,
                    'pos': rect
                })
            # 收集Top视图
            elif item_type == 'Top':
                page_data[page_num]['top_views'].append({
                    'type': item_type,
                    'pos': rect
                })
            # 收集package
            elif item_type == 'package':
                page_data[page_num]['package_count'] += 1

    # 为每个目标页面计算最小外接矩形框
    added_count = 0
    # existing_package_pages = {item['page'] for item in source_package_data}

    for page_num, data in page_data.items():
        # 使用所有相关矩形来计算，而不仅仅是rects
        if not data['all_rects']:
            continue

        all_rects = data['all_rects']
        keyview_count = data['keyview_count']
        package_count = data['package_count']
        top_views = data['top_views']
        keyviews = data['keyviews']

        # 情况1：只存在一个特征视图时，计算所有矩形框的最小外框
        if keyview_count == 1:
            # 只有一个关键特征视图，计算所有相关矩形框的最小外接矩形
            bounding_rect = [
                min(rect[0] for rect in all_rects),
                min(rect[1] for rect in all_rects),
                max(rect[2] for rect in all_rects),
                max(rect[3] for rect in all_rects)
            ]

            # 添加到source_package_data
            source_package_data.append({
                'page': page_num,
                'type': 'package',
                'detr_type': 'package',
                'pos': bounding_rect,
                'conf': 1,
                'match_state': -1
            })
            added_count += 1

        # 情况2：原逻辑 - 关键特征视图数量 > package数量
        elif keyview_count > package_count:
            # 计算需要补充的package数量
            need_package_count = keyview_count - package_count

            if need_package_count == 1 and keyview_count == 1:
                # 只有一个关键特征视图，计算所有相关矩形框的最小外接矩形
                bounding_rect = [
                    min(rect[0] for rect in all_rects),
                    min(rect[1] for rect in all_rects),
                    max(rect[2] for rect in all_rects),
                    max(rect[3] for rect in all_rects)
                ]

                # 添加到source_package_data
                source_package_data.append({
                    'page': page_num,
                    'type': 'package',
                    'detr_type': 'package',
                    'pos': bounding_rect,
                    'conf': 1,
                    'match_state': -1
                })
                added_count += 1

            elif need_package_count >= 1 and keyview_count > 1:
                # 多个关键特征视图，需要匹配Top视图
                sorted_top_views = _sort_views_by_position(top_views)
                sorted_keyviews = _sort_views_by_position(keyviews)

                # 为每个Top视图和对应的关键特征视图计算最小外接矩形
                for i, top_view in enumerate(sorted_top_views):
                    if i < len(sorted_keyviews) and i < need_package_count:
                        top_rect = top_view['pos']
                        keyview_rect = sorted_keyviews[i]['pos']

                        # 计算这两个矩形的最小外接矩形
                        bounding_rect = [
                            min(top_rect[0], keyview_rect[0]),
                            min(top_rect[1], keyview_rect[1]),
                            max(top_rect[2], keyview_rect[2]),
                            max(top_rect[3], keyview_rect[3])
                        ]

                        # 添加到source_package_data
                        source_package_data.append({
                            'page': page_num,
                            'type': 'package',
                            'detr_type': 'package',
                            'pos': bounding_rect,
                            'conf': 1,
                            'match_state': -1
                        })
                        added_count += 1

    return added_count


def detect_components(pdf_path, pages):
    """
    完整的组件检测流程（使用DETR模型）
    """
    # 1. 初始化数据存储
    init_data_storage()

    # 2. 转换PDF为图片
    pdf2img(pdf_path, pages)

    # 3. 调用DETR检测
    process_yolov13_detection()

    # 4. 为需要的页面添加最小外接矩形框
    add_minimum_bounding_boxes_for_target_pages()

    # 5. 保存检测结果到JSON文件
    results = {
        'source_data': source_data,
        'source_package_data': source_package_data,
        'source_keyview_data': source_keyview_data,
        'source_Top_data': source_Top_data,
        'source_Side_data': source_Side_data,
        'source_Detail_data': source_Detail_data,
        'source_Note_data': source_Note_data,
        'source_Form_data': source_Form_data,
        'source_Package_title_data': source_Package_title_data,
        'source_TOPVIEW_data': source_TOPVIEW_data,
        'source_BOTTOMVIEW_data': source_BOTTOMVIEW_data,
        'source_SIDEVIEW_data': source_SIDEVIEW_data,
        'source_DETAIL_data': source_DETAIL_data
    }

    # 5. 保存到JSON文件
    # with open('Result/PDF_extract/detr_detection_results0.json', 'w', encoding='utf-8') as f:
    #     json.dump(results, f, ensure_ascii=False, indent=2,
    #               default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)

    # print("检测结果已保存到 detr_detection_results.json")
    return results
