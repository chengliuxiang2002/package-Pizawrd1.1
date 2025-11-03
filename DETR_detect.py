"""使用DETR进行封装图分类识别"""
import re
import fitz
import shutil
from package_core.PDF_Processed.detr.test_detr_v2 import *

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

# 使用DETR的类别配置
CONFIG = DetectorConfig()
VOC_CLASSES = ['BGA', 'BOTTOMVIEW', 'DETAIL', 'DFN_SON', 'Detail', 'Form', 'Note', 'Package_title', 'QFN', 'QFP',
               'SIDEVIEW', 'SOP', 'Side', 'TOPVIEW', 'Top', 'package']
DETR_KEYVIEW_CLASSES = ['BGA', 'DFN_SON', 'QFP', 'QFN', 'SOP']
DETR_VIEW_CLASSES = ['Top', 'Side', 'Detail', 'Form']
# 全局数据存储变量
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


def expand_and_crop_image(box, image, expansion_factor_x, expansion_factor_y):
    height, width = image.shape[:2]

    # 框的坐标
    x1, y1, x2, y2 = box

    # 计算框的中心点和宽度/高度
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width_box = x2 - x1
    height_box = y2 - y1

    # 计算扩大后的框的尺寸
    expanded_width = int(width_box * expansion_factor_x)
    expanded_height = int(height_box * expansion_factor_y)

    # 计算扩大后的框的左上角和右下角坐标
    expanded_x1 = max(0, center_x - expanded_width / 2)
    expanded_y1 = max(0, center_y - expanded_height / 2)
    expanded_x2 = min(width, center_x + expanded_width / 2)
    expanded_y2 = min(height, center_y + expanded_height / 2)

    # 确保框的坐标是整数
    expanded_x1, expanded_y1, expanded_x2, expanded_y2 = map(int, (expanded_x1, expanded_y1, expanded_x2, expanded_y2))

    # 裁剪图片
    cropped_img = image[expanded_y1:expanded_y2, expanded_x1:expanded_x2]

    rectangle = expanded_x1, expanded_y1, expanded_x2, expanded_y2

    return rectangle, cropped_img


def straight_line(img, horizontalSize, verticalSize):
    src_img0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    src_img2 = cv2.bitwise_not(src_img0)

    thresh, AdaptiveThreshold = cv2.threshold(src_img2, 5, 255, 0)

    horizontal = AdaptiveThreshold.copy()
    vertical = AdaptiveThreshold.copy()

    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalSize))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    mask = horizontal + vertical
    return mask


def calculate_iou(box_a, box_b):
    # 确定矩形的 (x1, y1, x2, y2)
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    # 计算交集面积
    interArea = max(xB - xA + 1, 0) * max(yB - yA + 1, 0)

    # 计算并集面积
    boxAArea = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    boxBArea = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    unionArea = boxAArea + boxBArea - interArea

    # 避免除数为0的情况
    if unionArea == 0:
        return 0

    # 计算IoU
    iou = interArea / unionArea
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def table_rectangle(image, rect):
    typ = [None, None, None, None, None]
    rectangle, img = expand_and_crop_image(rect, image, 2, 2)
    (rx1, ry1, rx2, ry2) = rectangle
    (bx1, by1, bx2, by2) = rect
    new_bx1 = bx1 - rx1
    new_by1 = by1 - ry1
    new_bx2 = bx2 - rx1
    new_by2 = by2 - ry1
    box = (new_bx1, new_by1, new_bx2, new_by2)
    src_img0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    src_img1 = cv2.GaussianBlur(src_img0, (3, 3), 0)
    src_img2 = cv2.bitwise_not(src_img1)
    thresh, AdaptiveThreshold = cv2.threshold(src_img2, 5, 255, 0)
    x1, y1, x2, y2 = box
    y1 = int(y1)
    y2 = int(y2)
    x1 = int(x1)
    x2 = int(x2)
    box_area = AdaptiveThreshold[y1:y2, x1:x2]
    white_area = np.count_nonzero(box_area)
    box_area = (y2 - y1) * (x2 - x1)
    if white_area / box_area > 0.75:
        typ[0] = "Background"
        typ[1] = "no frame"
        typ[2] = "no outer frame"
        typ[3] = "no Horizontal lines"
        typ[4] = "no vertical lines"
        mask = straight_line(img, int(image.shape[1] / 20), int(image.shape[0] / 30))
        kernel = np.ones((6, 6), np.uint8)
        binary = cv2.dilate(mask, kernel)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        screened_rectangles = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            screened_rectangles.append([x, y, x + w, y + h])

        best_match = None
        max_iou = 0
        for screen_rect in screened_rectangles:
            iou = calculate_iou(box, screen_rect)
            if iou > max_iou:
                max_iou = iou
                best_match = screen_rect
    else:
        typ[0] = "no Background"
        mask = straight_line(img, int(image.shape[1] / 20), int(image.shape[0] / 30))
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        screened_rectangles = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            cnt_area = cv2.contourArea(cnt)
            ratio = cnt_area / area
            screened_rectangles.append([x, y, x + w, y + h, ratio])

        match = None
        max_iou = 0
        rect_ratio = 0
        for screen_rect_with_ratio in screened_rectangles:
            screen_rect = screen_rect_with_ratio[:4]
            iou = calculate_iou(box, screen_rect)
            if iou > max_iou:
                max_iou = iou
                match = screen_rect
                rect_ratio = screen_rect_with_ratio[-1]

        # match_area = (match[2] - match[0]) * (match[3] - match[1])
        # box_area = (box[2] - box[0]) * (box[3] - box[1])

        if max_iou > 0.5:
            typ[1] = "frame"
            best_match = match
            if rect_ratio >= 0.9:
                typ[2] = "outer frame"
            else:
                typ[2] = "no outer frame"

            need1, expanded_rectangle1 = expand_and_crop_image(best_match, img, 1.05, 1.05)
            # 设定长度阈值，即行中白色像素的最小数量
            length_threshold = expanded_rectangle1.shape[1] / 2

            src_img1 = cv2.cvtColor(expanded_rectangle1, cv2.COLOR_BGR2GRAY)
            src_img2 = cv2.bitwise_not(src_img1)
            thresh, AdaptiveThreshold = cv2.threshold(src_img2, 5, 255, 0)
            horizontal = AdaptiveThreshold.copy()
            horizontalSize = int(horizontal.shape[1] / 10)
            horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
            horizontal = cv2.erode(horizontal, horizontalStructure)
            horizontal = cv2.dilate(horizontal, horizontalStructure)

            sss = []
            for y in range(horizontal.shape[0]):
                row = horizontal[y, :]
                non_zero_count = np.count_nonzero(row)  # 计算非零像素的数量
                # 如果当前行的长度超过阈值
                if non_zero_count > length_threshold:
                    sss.append(y)
            if sss:
                count = 1
            else:
                count = 0
            for i in range(1, len(sss)):
                if sss[i] - sss[i - 1] > 1:  # 如果当前元素与前一个元素的差值大于1
                    count += 1  # 增加组的数量
            if count <= 5:
                typ[3] = "no Horizontal lines"
            else:
                typ[3] = "Horizontal lines"

            typ[4] = "vertical lines"

        else:
            typ[1] = "no frame"
            typ[2] = "no outer frame"
            typ[3] = "Horizontal lines"
            typ[4] = "no vertical lines"
            need, expanded_rectangle = expand_and_crop_image(box, mask, 1.2, 1.2)

            # 设定长度阈值，即行中白色像素的最小数量
            length_threshold = expanded_rectangle.shape[1] / 2

            # 初始化变量来存储结果
            long_lines_y_max = 0
            long_lines_y_min = expanded_rectangle.shape[0]

            # 遍历每一行
            for y in range(expanded_rectangle.shape[0]):
                row = expanded_rectangle[y, :]
                non_zero_count = np.count_nonzero(row)  # 计算非零像素的数量

                # 如果当前行的长度超过阈值
                if non_zero_count > length_threshold:
                    # 更新纵坐标的最大值和最小值
                    long_lines_y_max = max(long_lines_y_max, y)
                    long_lines_y_min = min(long_lines_y_min, y)

            cv2.line(expanded_rectangle, (expanded_rectangle.shape[1] // 2, long_lines_y_max),
                     (expanded_rectangle.shape[1] // 2, long_lines_y_min), (255), 1)
            contours2, hierarchy2 = cv2.findContours(expanded_rectangle, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            fff = []
            for cnt in contours2:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                fff.append([x, y, x + w, y + h, area])
            sorted_lst = sorted(fff, key=lambda x: x[4], reverse=True)
            match = sorted_lst[0][:4]

            best_match = [match[0] + need[0], match[1] + need[1], match[2] + need[0], match[3] + need[1]]

    (cx1, cy1, cx2, cy2) = best_match
    new_cx1 = cx1 + rx1
    new_cy1 = cy1 + ry1
    new_cx2 = cx2 + rx1
    new_cy2 = cy2 + ry1
    best_match_box = [new_cx1, new_cy1, new_cx2, new_cy2]
    return best_match_box


def correction_selected_coordinates(img, rect):
    """返回表格矫正后的坐标"""
    new_rect = table_rectangle(img, rect)
    return new_rect


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


def process_detr_detection():
    """
    调用DETR检测
    """
    global source_data, source_package_data, source_keyview_data
    global source_Top_data, source_Side_data, source_Detail_data
    global source_Note_data, source_Form_data, source_Package_title_data
    global source_TOPVIEW_data, source_BOTTOMVIEW_data, source_SIDEVIEW_data, source_DETAIL_data

    detr_start = time.time()
    create_dir(SAVE_IMG_PATH)
    img_item_list = os.listdir(IMAGE_PATH)
    img_item_list = sorted(img_item_list, key=natural_sort_key)

    # 初始化DETR检测器
    detector = RTDETR_Detector(CONFIG)

    for i in range(len(img_item_list)):
        img_item = img_item_list[i]
        img_path = os.path.join(IMAGE_PATH, img_item)

        # 读取图像
        original_image = cv2.imread(img_path)
        if original_image is None:
            print(f"无法读取图像: {img_path}")
            continue

        # 使用DETR进行检测
        image_tensor, original_size = preprocess_image(original_image, CONFIG.INPUT_SIZE)
        ort_inputs = {
            detector.input_names[0]: image_tensor,
            detector.input_names[1]: np.array([[CONFIG.INPUT_SIZE, CONFIG.INPUT_SIZE]], dtype=np.int64)
        }

        outputs = detector.session.run(None, ort_inputs)
        boxes, scores, labels = postprocess_results(
            outputs, original_size, CONFIG.INPUT_SIZE, CONFIG.CONF_THRESHOLD
        )

        # 处理检测结果
        page_num = int(os.path.splitext(img_item)[0])

        for j in range(len(boxes)):
            box = boxes[j]
            score = scores[j]
            label_id = int(labels[j])
            class_name = VOC_CLASSES[label_id] if label_id < len(VOC_CLASSES) else f"ID:{label_id}"

            # 过滤置信度小于0.6并且标签名称为package的检测结果
            if class_name == 'package' and score < 0.6:
                continue

            # 存储检测结果
            source_data.append({
                'page': page_num - 1,
                'type': class_name,
                'detr_type': class_name,
                'pos': [int(box[0] / ZOOM[0]),
                        int(box[1] / ZOOM[1]),
                        int(box[2] / ZOOM[0]),
                        int(box[3] / ZOOM[1])],
                'conf': score
            })
            # 分类存储不同类型的数据
            if class_name == 'package':

                source_package_data.append({
                    'page': page_num - 1,
                    'type': class_name,
                    'detr_type': class_name,
                    'pos':  [int(box[0] / ZOOM[0]),
                        int(box[1] / ZOOM[1]),
                        int(box[2] / ZOOM[0]),
                        int(box[3] / ZOOM[1])],
                    'conf': score,
                    'match_state': -1
                })
            elif class_name == 'Top':
                source_Top_data.append({
                    'page': page_num - 1,
                    'type': class_name,
                    'detr_type': class_name,
                    'pos':  [int(box[0] / ZOOM[0]),
                        int(box[1] / ZOOM[1]),
                        int(box[2] / ZOOM[0]),
                        int(box[3] / ZOOM[1])],
                    'conf': score,
                    'match_state': -1
                })
            elif class_name == 'Side':
                source_Side_data.append({
                    'page': page_num - 1,
                    'type': class_name,
                    'detr_type': class_name,
                    'pos':  [int(box[0] / ZOOM[0]),
                        int(box[1] / ZOOM[1]),
                        int(box[2] / ZOOM[0]),
                        int(box[3] / ZOOM[1])],
                    'conf': score,
                    'match_state': -1
                })
            elif class_name == 'Detail':
                source_Detail_data.append({
                    'page': page_num - 1,
                    'type': class_name,
                    'detr_type': class_name,
                    'pos': [int(box[0] / ZOOM[0]),
                        int(box[1] / ZOOM[1]),
                        int(box[2] / ZOOM[0]),
                        int(box[3] / ZOOM[1])],
                    'conf': score,
                    'match_state': -1
                })
            elif class_name == 'Note':
                source_Note_data.append({
                    'page': page_num - 1,
                    'type': class_name,
                    'detr_type': class_name,
                    'pos':  [int(box[0] / ZOOM[0]),
                        int(box[1] / ZOOM[1]),
                        int(box[2] / ZOOM[0]),
                        int(box[3] / ZOOM[1])],
                    'conf': score,
                    'match_state': -1
                })
            elif class_name == 'Form':
                source_Form_data.append({
                    'page': page_num - 1,
                    'type': class_name,
                    'detr_type': class_name,
                    'pos':  [int(box[0] / ZOOM[0]),
                        int(box[1] / ZOOM[1]),
                        int(box[2] / ZOOM[0]),
                        int(box[3] / ZOOM[1])],
                    'conf': score,
                    'match_state': -1
                })
            elif class_name == 'Package_title':
                source_Package_title_data.append({
                    'page': page_num - 1,
                    'type': class_name,
                    'pos':  [int(box[0] / ZOOM[0]),
                        int(box[1] / ZOOM[1]),
                        int(box[2] / ZOOM[0]),
                        int(box[3] / ZOOM[1])],
                    'conf': score,
                    'match_state': -1
                })
            elif class_name == 'TOPVIEW':
                source_TOPVIEW_data.append({
                    'page': page_num - 1,
                    'type': class_name,
                    'pos':  [int(box[0] / ZOOM[0]),
                        int(box[1] / ZOOM[1]),
                        int(box[2] / ZOOM[0]),
                        int(box[3] / ZOOM[1])],
                    'conf': score,
                    'match_state': -1
                })
            elif class_name == 'BOTTOMVIEW':
                source_BOTTOMVIEW_data.append({
                    'page': page_num - 1,
                    'type': class_name,
                    'pos':  [int(box[0] / ZOOM[0]),
                        int(box[1] / ZOOM[1]),
                        int(box[2] / ZOOM[0]),
                        int(box[3] / ZOOM[1])],
                    'conf': score,
                    'match_state': -1
                })
            elif class_name == 'SIDEVIEW':
                source_SIDEVIEW_data.append({
                    'page': page_num - 1,
                    'type': class_name,
                    'pos':  [int(box[0] / ZOOM[0]),
                        int(box[1] / ZOOM[1]),
                        int(box[2] / ZOOM[0]),
                        int(box[3] / ZOOM[1])],
                    'conf': score,
                    'match_state': -1
                })
            elif class_name == 'DETAIL':
                source_DETAIL_data.append({
                    'page': page_num - 1,
                    'type': class_name,
                    'pos':  [int(box[0] / ZOOM[0]),
                        int(box[1] / ZOOM[1]),
                        int(box[2] / ZOOM[0]),
                        int(box[3] / ZOOM[1])],
                    'conf': score,
                    'match_state': -1
                })
            else:  # 关键特征视图（BGA, DFN_SON, QFP, QFN, SOP）
                source_keyview_data.append({
                    'page': page_num - 1,
                    'pos':  [int(box[0] / ZOOM[0]),
                        int(box[1] / ZOOM[1]),
                        int(box[2] / ZOOM[0]),
                        int(box[3] / ZOOM[1])],
                    'type': class_name,
                    'detr_type': class_name,
                    'conf': score,
                    'match_state': -1
                })

        # 保存可视化结果（可选）
        # if len(boxes) > 0:
        #     vis_image = visualize_detections(original_image.copy(), boxes, scores, labels, VOC_CLASSES)
        #     save_path = os.path.join(SAVE_IMG_PATH, img_item)
        #     cv2.imwrite(save_path, vis_image)

    detr_end = time.time()
    print(f"DETR检测耗时: {detr_end - detr_start:.4f}秒")


# def _sort_views_by_position(views):
#     """按位置对视图进行排序（从左到右，从上到下）"""
#     return sorted(views, key=lambda x: (x['pos'][0], x['pos'][1]))
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
    process_detr_detection()

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
