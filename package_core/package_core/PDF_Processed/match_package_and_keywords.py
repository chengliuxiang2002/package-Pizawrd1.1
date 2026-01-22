import json
import cv2
import copy
import tempfile
import numpy as np
from shapely.geometry import Polygon
from package_core.PDF_Processed.DETR_detect import detect_components
from package_core.PDF_Processed.tools.search_keywords_in_editable_page import *
from package_core.PDF_Processed.tools.find_packagedata_from_title import *
from package_core.PDF_Processed.tools.merge_image import *
from package_core.PDF_Processed.ocr.det_text import Run_onnx1

# 导入统一路径管理
try:
    from package_core.PackageExtract.yolox_onnx_py.model_paths import result_path
except ModuleNotFoundError:
    from pathlib import Path
    def result_path(*parts):
        return str(Path(__file__).resolve().parents[2] / 'Result' / Path(*parts))

# 关键字
TYPE_KEYWORDS = ["BGA", "DFN", "SON", "QFP", "QFN", "SOP", "SOT", "SOIC", "LFCSP", "BALL GRID ARRAY",
                 "Quad Flat Package", "Quad Flatpack"]
VIEW_KEYWORDS = ["TOPVIEW", "SIDEVIEW", "DETAIL"]

def get_detr_package_title_results(detr_result, page_num):
    """从DETR结果中获取指定页码的Package_title结果"""
    if detr_result is None:
        return []
    package_title_results = []
    if 'source_Package_title_data' in detr_result:
        for item in detr_result['source_Package_title_data']:
            if item.get('page') == page_num:
                package_title_results.append({
                    'page': item['page'],
                    'keyword': 'Package_title',
                    'coordinates': item['pos'],
                    'type': 'detr',
                    'content': 'Package_title'
                })

    return package_title_results


def get_detr_package_results(detr_result, page_num):
    """获取DETR检测的package结果"""
    if detr_result is None:
        return []
    package_results = []
    if 'source_package_data' in detr_result:
        for item in detr_result['source_package_data']:
            if item.get('page') == page_num:
                package_results.append({
                    'page': item['page'],
                    'keyword': 'package',
                    'coordinates': item['pos'],
                    'type': 'detr',
                    'content': 'package',
                    'conf': item.get('conf', 0.0)
                })
    return package_results

def get_detr_view_results(detr_result,page_num):
    """从DETR结果中获取指定页码的TOPVIEW, BOTTOMVIEW,SIDEVIEW, DETAIL结果"""
    if detr_result is None:
        return []

    view_results = []
    # 获取TOPVIEW数据
    if 'source_TOPVIEW_data' in detr_result:
        for item in detr_result['source_TOPVIEW_data']:
            if item.get('page') == page_num:
                view_results.append({
                    'page': item['page'],
                    'keyword': 'TOPVIEW',
                    'coordinates': item['pos'],
                    'type': 'detr',
                    'content': 'TOPVIEW'
                })

    # 获取SIDEVIEW数据
    if 'source_SIDEVIEW_data' in detr_result:
        for item in detr_result['source_SIDEVIEW_data']:
            if item.get('page') == page_num:
                view_results.append({
                    'page': item['page'],
                    'keyword': 'SIDEVIEW',
                    'coordinates': item['pos'],
                    'type': 'detr',
                    'content': 'SIDEVIEW'
                })
    # 获取BOTTOMVIEW数据
    if 'source_BOTTOMVIEW_data' in detr_result:
        for item in detr_result['source_BOTTOMVIEW_data']:
            if item.get('page') == page_num:
                view_results.append({
                    'page': item['page'],
                    'keyword': 'BOTTOMVIEW',
                    'coordinates': item['pos'],
                    'type': 'detr',
                    'content': 'BOTTOMVIEW'
                })
    # 获取DETAIL数据
    if 'source_DETAIL_data' in detr_result:
        for item in detr_result['source_DETAIL_data']:
            if item.get('page') == page_num:
                view_results.append({
                    'page': item['page'],
                    'keyword': 'DETAIL',
                    'coordinates': item['pos'],
                    'type': 'detr',
                    'content': 'DETAIL'
                })

    return view_results

# def get_detr_keyview_results(detr_result, page_num):
#     """获取DETR检测的关键特征视图结果(BGA, DFN_SON, QFP, QFN, SOP等)"""
#     if detr_result is None:
#         return []
#     keyview_results = []
#     if 'source_keyview_data' in detr_result:
#         for item in detr_result['source_keyview_data']:
#             if item.get('page') == page_num:
#                 keyview_results.append({
#                     'page': item['page'],
#                     'keyword': item.get('type', ''),
#                     'coordinates': item['pos'],
#                     'type': 'detr',
#                     'content': item.get('type', ''),
#                     'conf': item.get('conf', 0.0)
#                 })
#     return keyview_results


def match_keywords_for_all_pages(pdf_path, page_list, detection_results):
    """
    替代旧的 run_keyword_matching 循环。
    接收原始DETR结果，返回一个被修改后的新副本。
    """
    # 创建一个深拷贝以避免修改原始输入字典
    modified_results = copy.deepcopy(detection_results)

    keywords = ["BGA", "DFN", "SON", "QFP", "QFN", "SOP", "SOT", "SOIC", "BALL GRID ARRAY", "Plastic Quad Flat Package",
                "Quad Flatpack", "TOPVIEW", "SIDEVIEW", "BOTTOMVIEW", "TOP VIEW", "SIDE VIEW", "BOTTOM VIEW", "TOP", "SIDE",
                "BOTTOM", "VIEW", "DETAIL"]

    # 清空package_baseinfo.json
    with open('package_baseinfo.json', 'w', encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False, indent=2)

    for page_num in page_list:
        # 注意：现在将 modified_results 传入
        page_results = process_page_keywords(pdf_path, page_num, keywords, modified_results)
        match_package_with_type(pdf_path, page_num, page_results, modified_results)
        match_package_with_view(page_num, page_results, modified_results)
        remove_title(page_num, page_results)

    return modified_results

def get_rects_d(rect1_coords, rect2_coords):
    """输入两个矩形坐标，返回这两个矩形之间的最短距离"""
    rect1 = Polygon([(rect1_coords[0], rect1_coords[1]),
                     (rect1_coords[0], rect1_coords[3]),
                     (rect1_coords[2], rect1_coords[3]),
                     (rect1_coords[2], rect1_coords[1])])

    rect2 = Polygon([(rect2_coords[0], rect2_coords[1]),
                     (rect2_coords[0], rect2_coords[3]),
                     (rect2_coords[2], rect2_coords[3]),
                     (rect2_coords[2], rect2_coords[1])])

    return rect1.distance(rect2)


def get_rects(rect1_coords, rect2_coords):
    """输入两个矩形坐标，返回这两个矩形中心点之间的距离"""
    rect1_center_x = (rect1_coords[0] + rect1_coords[2]) / 2
    rect1_center_y = (rect1_coords[1] + rect1_coords[3]) / 2

    rect2_center_x = (rect2_coords[0] + rect2_coords[2]) / 2
    rect2_center_y = (rect2_coords[1] + rect2_coords[3]) / 2

    distance = ((rect1_center_x - rect2_center_x) ** 2 +
                (rect1_center_y - rect2_center_y) ** 2) ** 0.5

    return distance


def merge_image(region_images, gap: int = 20):
    """合并区域图像，使用左下角算法进行布局，确保合并后的顺序与输入的region_images一致"""
    # 创建临时目录保存区域图像
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_paths = []
        for i, img in enumerate(region_images):
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            temp_path = os.path.join(temp_dir, f"region_{i}.png")
            pil_img.save(temp_path)
            temp_paths.append((i, temp_path))

        # 1. 创建ImageBlock列表，同时记录原始索引
        blocks_with_index = []
        for orig_idx, path in temp_paths:
            block = ImageBlock(path)
            blocks_with_index.append((orig_idx, block))

        # 2. 提取图像块用于左下角布局算法
        blocks = [b for (i, b) in blocks_with_index]
        canvas_w, canvas_h = bottom_left_layout(blocks, gap)

        # 3. 执行合并并建立原始索引到合并后坐标的映射
        canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
        index_to_mapping = {}

        for (orig_idx, block) in blocks_with_index:
            canvas.paste(block.img, (block.x, block.y))
            merged_x0 = block.x
            merged_y0 = block.y
            merged_x1 = block.x + block.width
            merged_y1 = block.y + block.height
            index_to_mapping[orig_idx] = [merged_x0, merged_y0, merged_x1, merged_y1]

        # 4. 按原始输入顺序重建region_mappings
        region_mappings = [index_to_mapping[i] for i in range(len(region_images))]
        merged_cv2 = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

    return merged_cv2, region_mappings


def is_page_editable(pdf_path, page_num):
    """判断 PDF 指定页面是否可编辑"""
    doc = fitz.open(pdf_path)
    if 0 <= page_num <= len(doc):
        page = doc[page_num]

        text = page.get_text()
        if text.strip():
            return True
        else:
            image_list = page.get_images(full=True)
            if len(image_list) > 0:
                return False
            else:
                return False
    else:
        print(f"页面编号 {page_num} 超出范围")
        return False


def process_non_editable_page_with_package_titles(pdf_path,detr_result, page_num, keywords):
    """处理不可编辑页面：获取Package_title结果，裁剪图片，合并后OCR处理，与DETR视图关键字合并"""

    # 1. 获取DETR检测结果中的Package_title数据
    package_title_results = get_detr_package_title_results(detr_result, page_num)

    if not package_title_results:
        view_results = get_detr_view_results(detr_result, page_num)
        print("get_detr_view_results返回:", view_results)
        return view_results

    # 2. 提取页面图像
    with fitz.open(pdf_path) as doc:
        page = doc[page_num]
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 3. 根据Package_title坐标裁剪图像区域
    region_images = []
    region_origin_info = []
    for item in package_title_results:
        coords = item['coordinates']
        x0, y0, x1, y1 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        cropped_img = img[y0:y1, x0:x1]
        region_images.append(cropped_img)
        region_origin_info.append({
            'type': 'package_title',
            'orig_coords': (x0, y0, x1, y1),
            'item': item
        })

    if region_images:
        # 4. 合并图像
        merged_img, region_mappings = merge_image(region_images, gap=20)

        # 5. 对合并后的图像进行OCR处理
        merged_boxes, merged_texts = Run_onnx1(merged_img, f"merged_package_titles_page_{page_num}")
        print("OCR识别结果:", merged_texts)

        # 6. 将OCR结果分配到对应的原始图像（使用矩形相交检测）
        original_image_ocr_results = []
        ocr_results = []

        for idx, region_map in enumerate(region_mappings):
            merged_x0, merged_y0, merged_x1, merged_y1 = region_map
            region_rect = (merged_x0, merged_y0, merged_x1, merged_y1)

            origin_info = region_origin_info[idx]
            orig_x0, orig_y0, orig_x1, orig_y1 = origin_info['orig_coords']

            region_texts = []
            region_boxes_pdf = []
            region_boxes_merged = []

            for box, text in zip(merged_boxes, merged_texts):
                box_x_coords = [p[0] for p in box]
                box_y_coords = [p[1] for p in box]
                box_x0, box_x1 = min(box_x_coords), max(box_x_coords)
                box_y0, box_y1 = min(box_y_coords), max(box_y_coords)
                text_box_rect = (box_x0, box_y0, box_x1, box_y1)

                distance = get_rects_d(region_rect, text_box_rect)
                if int(distance) == 0:
                    region_texts.append(text)
                    region_boxes_pdf.append((orig_x0, orig_y0, orig_x1, orig_y1))
                    region_boxes_merged.append(text_box_rect)

            # 保存当前原始图像的OCR结果
            original_image_ocr_results.append({
                "texts": region_texts,
                "boxes_pdf": region_boxes_pdf,
                "boxes_merged": region_boxes_merged,
                "original_index": idx,
                "merged_region": region_rect
            })

        # 7. 处理OCR结果，映射回原始区域
        for idx, origin_info in enumerate(region_origin_info):
            orig_x0, orig_y0, orig_x1, orig_y1 = origin_info['orig_coords']

            region_ocr_data = original_image_ocr_results[idx]
            region_ocr_texts = region_ocr_data["texts"]
            region_ocr_boxes_pdf = region_ocr_data["boxes_pdf"]

            for text, box in zip(region_ocr_texts, region_ocr_boxes_pdf):
                if not text:
                    continue

                cleaned_text = text.strip()
                for keyword in keywords:
                    if keyword.lower() in cleaned_text.lower():
                        ocr_results.append({
                            'page': page_num,
                            'keyword': keyword,
                            'coordinates': (orig_x0, orig_y0, orig_x1, orig_y1),
                            'type': 'ocr',
                            'content': cleaned_text,
                            'original_content': text
                        })
                        break

        # 8. 合并OCR结果和DETR视图关键字结果
        view_results = get_detr_view_results(detr_result,page_num)
        ocr_results.extend(view_results)
        return ocr_results

    # 如果没有图像区域可处理，直接返回DETR视图关键字结果
    return get_detr_view_results(detr_result,page_num)


def process_page_keywords(pdf_path, page_num, keywords, detr_result):
    """
    判断页面是否可编辑，获取所有关键字，去重，合并视图关键字，并进行类型和视图关键字匹配
    """
    is_editable = is_page_editable(pdf_path, page_num)

    all_title_results = []
    if is_editable:
        results = search_keywords_in_editable_page(pdf_path, page_num, keywords)
        found_keywords = set(result['keyword'] for result in results)
        view_keywords_present = any(kw in found_keywords for kw in
                                    ["TOP", "SIDE", "VIEW", "TOP VIEW", "SIDE VIEW", "TOPVIEW", "SIDEVIEW"])

        # 处理可编辑但不存在view关键字,合并可编辑页面结果和DETR检测结果中的视图关键字
        if not view_keywords_present:
            detr_view_results = get_detr_view_results(detr_result, page_num)
            results.extend(detr_view_results)
    else:
        # 处理不可编辑页面
        results = process_non_editable_page_with_package_titles(pdf_path, detr_result,page_num, keywords)

    all_title_results.extend(results)

    # 坐标完全相同的数据去重
    unique_results = []
    seen_coords = set()
    for item in all_title_results:
        coords_tuple = tuple(item['coordinates'])
        if coords_tuple not in seen_coords:
            unique_results.append(item)
            seen_coords.add(coords_tuple)
    all_results = unique_results

    return all_results


def match_package_with_type(pdf_path, page_num, all_results, detr_result):
    """将package与类型关键字匹配"""
    # 定义类型关键字列表
    type_keywords = TYPE_KEYWORDS

    # 1. 获取当前页面的所有类型关键字
    editable_keywords = [item for item in all_results
                         if item['page'] == page_num
                         and item['keyword'] in type_keywords
                         and item.get('type') == 'text']

    if editable_keywords:
        type_keywords_list = editable_keywords
    else:
        ocr_keywords = [item for item in all_results
                        if item['page'] == page_num
                        and item['keyword'] in type_keywords
                        and item.get('type') == 'ocr']
        type_keywords_list = ocr_keywords

    if not type_keywords_list:
        print(f"页面 {page_num} 未找到类型关键字")
        return

    # 2. 从DETR检测结果获取package数据
    package_list = [item for item in detr_result.get('source_package_data', [])
                    if item.get('page') == page_num]

    if not package_list:
        print(f"页面 {page_num} 未找到DETR package数据")
        return

    # 3. 为每个package找到对应的类型关键字并修改内部视图名称
    for i, package in enumerate(package_list):
        package_rect = package['pos']
        print(f"处理第 {i + 1} 个package: {package_rect}")

        # 计算与所有类型关键字的距离
        distances = []
        for type_kw in type_keywords_list:
            dist = get_rects(package_rect, type_kw['coordinates'])
            distances.append((dist, type_kw))
        distances.sort(key=lambda x: x[0])
        if distances:
            closest_type = distances[0][1]
            if closest_type in type_keywords_list:
                type_keywords_list.remove(closest_type)
                package['type'] = closest_type['keyword']
                print(f"Package {i + 1} 匹配到类型: {closest_type['keyword']}")

            content = closest_type.get('content', '')
            info = clean_result(check_keywords_and_numbers(content))
            result_dict = {
                'pdf': os.path.basename(pdf_path),
                'page_num': page_num,
                'content': content,
                'package_type': closest_type.get('keyword', ''),
                'pin': info.get('pin') if info.get('pin') is not None else None,
                'length': info.get('length') if info.get('length') is not None else None,
                'width': info.get('width') if info.get('width') is not None else None,
                'height': info.get('height') if info.get('height') is not None else None,
                'horizontal_pin': info.get('horizontal_pin') if info.get('horizontal_pin') is not None else None,
                'vertical_pin': info.get('vertical_pin') if info.get('vertical_pin') is not None else None,
            }

            # 写入json文档，标准JSON数组格式
            output_path = 'package_baseinfo.json'
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = []

            data.append(result_dict)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)


def match_package_with_view(page_num, all_results, detr_result):
    """检查package中的视图与视图关键字匹配"""
    # 定义视图关键字列表
    view_keywords = VIEW_KEYWORDS

    # 1.存储含有视图关键字的数据
    view_keywords_list = [item for item in all_results
                          if item['page'] == page_num
                          and any(keyword in item['keyword'] for keyword in view_keywords)]

    package_list = [item for item in detr_result.get('source_package_data', [])
                    if item.get('page') == page_num]

    # 2.为每个package处理视图匹配
    for package in package_list:
        pkg_y0 = package['pos'][1]
        pkg_y1 = package['pos'][3]
        h = pkg_y1 - pkg_y0
        filtered_view_keywords_list = [item for item in view_keywords_list if
                                       pkg_y0 - h * 0.1 <= item['coordinates'][3] <= pkg_y1 + h * 0.1]
        filtered_view_keywords_list.sort(key=lambda x: x['coordinates'][1])

        current_package_rect = package['pos']
        current_page = package['page']

        # 获取Top视图
        top_views = []
        if 'source_Top_data' in detr_result:
            for item in detr_result['source_Top_data']:
                if (item.get('page') == current_page and
                        get_rects_d(current_package_rect, item['pos']) == 0):
                    top_views.append({
                        'part_name': 'Top',
                        'rect': item['pos'],
                        'page': item['page']
                    })
        # 获取Side视图
        side_views = []
        if 'source_Side_data' in detr_result:
            for item in detr_result['source_Side_data']:
                if (item.get('page') == current_page and
                        get_rects_d(current_package_rect, item['pos']) == 0):
                    side_views.append({
                        'part_name': 'Side',
                        'rect': item['pos'],
                        'page': item['page']
                    })
        # 获取Detail视图
        detail_views = []
        if 'source_Detail_data' in detr_result:
            for item in detr_result['source_Detail_data']:
                if (item.get('page') == current_page and
                        get_rects_d(current_package_rect, item['pos']) == 0):
                    detail_views.append({
                        'part_name': 'Detail',
                        'rect': item['pos'],
                        'page': item['page']
                    })

        # 获取关键特征视图
        keyview_views = []
        if 'source_keyview_data' in detr_result:
            for item in detr_result['source_keyview_data']:
                if (item.get('page') == current_page and
                        get_rects_d(current_package_rect, item['pos']) == 0 and
                        item.get('type', '') in ['QFP', 'SOP']):
                    keyview_views.append({
                        'part_name': item.get('type', ''),
                        'rect': item['pos'],
                        'page': item['page']
                    })
        all_views = top_views + side_views + detail_views + keyview_views

        view_list = [part for part in all_views]
        view_list.sort(key=lambda x: x['rect'][1])

        if filtered_view_keywords_list:
            for part in view_list:
                ref_rect = part['rect']
                x0 = ref_rect[0]
                x1 = ref_rect[2]
                y0 = ref_rect[1]
                y1 = ref_rect[3]
                view_center_x = (x0 + x1) / 2
                view_center_y = (y0 + y1) / 2

                filtered_parts = [
                    part for part in view_list
                    if x0 <= ((part['rect'][0] + part['rect'][2]) / 2) <= x1
                       and ((part['rect'][1] + part['rect'][3]) / 2) > y1
                ]
                filtered_parts.sort(
                    key=lambda part: (part['rect'][1] + part['rect'][3]) / 2
                )
                filtered_views = []
                if filtered_parts:
                    first_fp_y0 = filtered_parts[0]['rect'][1]
                    for view in filtered_view_keywords_list:
                        coords = view['coordinates']
                        center_x = (coords[0] + coords[2]) / 2
                        center_y = (coords[1] + coords[3]) / 2
                        if x0 <= center_x <= x1 and center_y < first_fp_y0:
                            filtered_views.append(view)
                else:
                    for view in filtered_view_keywords_list:
                        coords = view['coordinates']
                        center_x = (coords[0] + coords[2]) / 2
                        if x0 <= center_x <= x1:
                            filtered_views.append(view)

                # 存储上方和下方的视图关键字
                top_keywords = []
                bottom_keywords = []

                for view in filtered_views:
                    # 计算视图关键字的中心点和边界
                    coords = view['coordinates']
                    kw_x0, kw_y0, kw_x2, kw_y2 = coords
                    kw_center_x = (kw_x0 + kw_x2) / 2
                    kw_center_y = (kw_y0 + kw_y2) / 2

                    # 关键字必须完全在视图的x0-x1水平范围内
                    kw_fully_in_view_x_range = (x0 <= kw_x0 and kw_x2 <= x1)

                    # 计算两个中心点之间的距离
                    distance = ((view_center_x - kw_center_x) ** 2 + (view_center_y - kw_center_y) ** 2) ** 0.5

                    # 计算相对于y轴的角度（以视图中心点为原点）
                    dx = kw_center_x - view_center_x  # x方向差值
                    dy = kw_center_y - view_center_y  # y方向差值
                    angle = math.atan2(dy, dx)
                    angle = math.degrees(angle)
                    if angle < 0:
                        angle += 360  # 将负角度转换为0-360度范围

                    # 上方区域的限制
                    if (225 <= angle <= 315 and  # 角度范围
                            kw_fully_in_view_x_range):  # 关键字的左右边界必须在视图x0-x1范围内

                        top_keywords.append((distance, view, angle))

                    # 下方区域的限制
                    elif (45 <= angle <= 135 and  # 角度范围
                          kw_fully_in_view_x_range):  # 关键字的左右边界必须在视图x0-x1范围内

                        bottom_keywords.append((distance, view, angle))

                # 优先选择上方的视图关键字
                if top_keywords:
                    top_keywords.sort(key=lambda x: x[0])
                    closest_view = top_keywords[0][1]
                    print(f"视图 '{part.get('part_name')}' 匹配到上方关键字 '{closest_view['keyword']}'")

                elif bottom_keywords:
                    bottom_keywords.sort(key=lambda x: x[0])
                    closest_view = bottom_keywords[0][1]
                    print(f"视图 '{part.get('part_name')}' 匹配到下方关键字 '{closest_view['keyword']}'")

                else:
                    continue

                for item in detr_result.get('source_Top_data', []) + \
                            detr_result.get('source_Side_data', []) + \
                            detr_result.get('source_Detail_data', []) + \
                            detr_result.get('source_keyview_data', []):
                    if (item.get('page') == current_page and
                            item.get('pos') == part.get('rect') and
                            item.get('type') == part.get('part_name')):
                        # 更新 detr_result 中的数据
                        item['type'] = closest_view['keyword']
                        filtered_view_keywords_list.remove(closest_view)
                        break


def remove_title(page_num, results):
    """
    删除该页所有results区域并保存图片
    """
    IMAGE_PATH = result_path('PDF_extract', 'detr_img')
    OUTPUT_PATH = result_path('PDF_extract', 'detr_img')

    # 输入和输出文件路径
    input_file = os.path.join(IMAGE_PATH, f"{page_num + 1}.png")
    output_file = os.path.join(OUTPUT_PATH, f"{page_num + 1}.png")

    if not os.path.exists(input_file):
        print(f"图片文件不存在: {input_file}")
        return None

    # 打开图片
    original_img = Image.open(input_file)
    img_array = np.array(original_img)

    # 删除所有results区域
    for item in results:
        coordinates = item.get('coordinates', [])
        if len(coordinates) >= 4:
            x1, y1 = int(coordinates[0]) * 3, int(coordinates[1]) * 3
            x2, y2 = int(coordinates[2]) * 3, int(coordinates[3]) * 3

            # 边界检查
            x1 = max(0, min(x1, img_array.shape[1] - 1))
            y1 = max(0, min(y1, img_array.shape[0] - 1))
            x2 = max(0, min(x2, img_array.shape[1] - 1))
            y2 = max(0, min(y2, img_array.shape[0] - 1))

            # 设置为白色
            img_array[y1:y2 + 1, x1:x2 + 1] = [255, 255, 255]

    # 创建输出目录并保存图片
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    processed_pil = Image.fromarray(img_array)
    processed_pil.save(output_file)
    print(f"已保存处理后的图片: {output_file}")

    return output_file