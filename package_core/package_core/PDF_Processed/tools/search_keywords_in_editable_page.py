import fitz
import os
import re
from PIL import Image
from collections import defaultdict



def is_page_editable(pdf_path, page_num):
    """判断 PDF 指定页面是否可编辑"""
    doc = fitz.open(pdf_path)
    if 0 <= page_num <= len(doc):
        page=doc[page_num]

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

# 处理每个关键字的搜索结果
def rect_overlap_ratio(box1, box2):
    x0_1, y0_1, x1_1, y1_1 = box1
    x0_2, y0_2, x1_2, y1_2 = box2
    inter_x0 = max(x0_1, x0_2)
    inter_y0 = max(y0_1, y0_2)
    inter_x1 = min(x1_1, x1_2)
    inter_y1 = min(y1_1, y1_2)
    inter_w = max(0, inter_x1 - inter_x0)
    inter_h = max(0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    min_area = min(area1, area2)
    if min_area == 0:
        return 0
    return inter_area / min_area

def find_line_boundaries(y0, y1):
    """确定行的上下边界"""
    # line_height = y1 - y0
    # line_top = y0 - line_height * 0.5
    # line_bottom = y1 + line_height * 0.5
    line_top=y0
    line_bottom=y1
    return line_top, line_bottom

def get_full_line_text(page,  line_top, line_bottom):
    """获取整行文本"""
    page_width = page.rect.width
    full_line = page.get_text("text", clip=(0, line_top, page_width, line_bottom))
    return full_line.strip()

def clean_excel_text(text):
    # 移除所有非法的控制字符（除常用换行、回车、tab外）
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)


def search_keywords_in_editable_page(pdf_path, page_num, keywords):
    """在可编辑的页面中搜索关键字并获取其标签和坐标"""
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    keyword_results = {}
    page_dict = page.get_text("dict")

    # 获取每个关键字的坐标
    for keyword in keywords:
        instances = page.search_for(keyword)
        for inst in instances:
            x0, y0, x1, y1 = map(int, inst)

            line_top, line_bottom = find_line_boundaries(y0, y1)
            line_text = get_full_line_text(page, line_top, line_bottom)
            line_text = clean_excel_text(line_text)

            font_size = 0
            for span in page.get_text("dict", clip=(x0, y0, x1, y1))["blocks"]:
                if "lines" in span:
                    for line in span["lines"]:
                        if "spans" in line:
                            for span_info in line["spans"]:
                                if keyword.lower() in span_info["text"].lower():
                                    font_size = span_info["size"]
                                    break

            result = {
                'page': page_num,
                'keyword': keyword,
                'coordinates': (x0, y0, x1, y1),
                'type': 'text',
                'content': line_text,
                'font_size': font_size,
                'area': (x1 - x0) * (y1 - y0)
            }

            if keyword not in keyword_results:
                keyword_results[keyword] = []
            keyword_results[keyword].append(result)

    overlap_threshold = 0.5
    intermediate_results = []

    # 排序+重叠检测
    for keyword, results in keyword_results.items():
        # 先按字体大小降序排序
        results.sort(key=lambda x: (-x['font_size'], x['area']))

        filtered_results = []
        for res in results:
            overlap = False
            for kept in filtered_results:
                ratio = rect_overlap_ratio(res['coordinates'], kept['coordinates'])
                if ratio > overlap_threshold:
                    # 重叠时保留面积较小的结果
                    if res['area'] < kept['area']:
                        filtered_results.remove(kept)
                        filtered_results.append(res)
                    overlap = True
                    break
            if not overlap:
                filtered_results.append(res)

        intermediate_results.extend(filtered_results)

    # 处理最终结果
    processed_results = []
    for result in intermediate_results:
        result.pop('font_size', None)
        result.pop('area', None)
        if result['keyword'].lower() == 'side view':
            result['keyword'] = 'SIDEVIEW'
        elif result['keyword'].lower() == 'top view':
            result['keyword'] = 'TOPVIEW'
        processed_results.append(result)

    doc.close()
    return processed_results


if __name__ == "__main__":
    # 测试代码
    pdf_path = r"D:\20250822\PackageWizard1.0_F1\PDF_Processed\PDF\sn74auc1g04.pdf"  # 替换为实际的PDF文件路径
    page_num = 13 # 要处理的页码
    keywords = ["BGA", "DFN", "SON", "QFP", "QFN", "SOP","SOT", "Quad Flat Package","TOPVIEW","TOP VIEW", "SIDEVIEW","SIDE VIEW","TOP","SIDE","VIEW","DETAIL"]

    is_editable = is_page_editable(pdf_path, page_num)
    if is_editable:
        # 测试可编辑页面处理
        results = search_keywords_in_editable_page(pdf_path, page_num, keywords)
        print("可编辑页面结果:", results)
        print(len(results))

    # else:
    #     # 测试不可编辑页面处理
    #     results = process_non_editable_page(pdf_path, page_num, detector)
    #     print("不可编辑页面结果:", results)  # 可视化结果

