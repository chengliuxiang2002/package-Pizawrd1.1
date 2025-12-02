# PDF_Processed_main.py (Refactored)
import fitz
from package_core.PDF_Processed.watermark_remove import watermark_remove
from package_core.PDF_Processed.extract_package_page import extract_package_page_list
from package_core.PDF_Processed.DETR_detect import detect_components
from package_core.PDF_Processed.match_package_and_keywords import match_keywords_for_all_pages
from package_core.PDF_Processed.match_package_and_views import process_package_matching
import numpy as np

# 在类外部定义颜色常量
PACKAGE_COLOR = (1, 0, 0)  # 红色
NOTE_COLOR = (0, 1, 1)  # 青色
TOP_COLOR = (0, 1, 0)  # 绿色
SIDE_COLOR = (0, 0, 1)  # 蓝色
DETAIL_COLOR = (1, 1, 0)  # 黄色
FORM_COLOR = (0.5, 0.5, 0.5)  # 灰色
KEYVIEW_COLOR = (1, 0, 1)  # 紫色


class PackageDetectionPipeline:
    """
    一个清晰的、分步骤的封装图检测管道。
    数据在步骤之间显式传递，没有全局状态。
    """

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def step1_preprocess_pages(self):
        """步骤1: 去水印并筛选出需要处理的页面列表。"""
        print("步骤1: 去除水印并筛选页面...")
        watermark_remove(self.pdf_path)

        with fitz.open(self.pdf_path) as doc:
            page_count = doc.page_count

        page_list = extract_package_page_list(self.pdf_path)

        # 扩展页面搜索范围
        expanded_page_list = set()
        for page_num in page_list:
            expanded_page_list.update([page_num - 1, page_num, page_num + 1])

        # 过滤无效页码并排序
        valid_page_list = sorted(list(filter(lambda num: 0 <= num < page_count, expanded_page_list)))
        print(f"最终处理页面列表: {valid_page_list} (共 {len(valid_page_list)} 页)")
        return valid_page_list

    def step2_run_detr_detection(self, page_list):
        """步骤2: 对指定的页面列表运行DETR模型检测。"""
        print("\n步骤2: 开始DETR组件检测...")
        if not page_list:
            print("警告: 没有需要处理的页面。")
            return None

        # detect_components 返回原始的检测结果
        detection_results = detect_components(self.pdf_path, page_list)
        return detection_results

    def step3_match_keywords(self, detection_results, page_list):
        """步骤3: 将关键字与DETR检测结果进行匹配，返回一个被修改和增强后的结果。"""
        print("\n步骤3: 开始关键字与视图匹配...")
        if not detection_results:
            print("警告: 没有DETR结果可供匹配。")
            return None

        # 这个新函数将处理所有页面的关键字匹配，并返回修改后的结果
        # 它不再依赖全局变量
        modified_results = match_keywords_for_all_pages(self.pdf_path, page_list, detection_results)
        return modified_results

    def step4_group_package_components(self, modified_detr_results):
        """步骤4: 将所有视图和组件组合成最终的封装图对象。"""
        print("\n步骤4: 开始组合封装图组件...")
        if not modified_detr_results:
            print("警告: 没有可供组合的结果。")
            return None, None, None

        # process_package_matching 接收修改后的结果作为输入
        package_data, data2, have_page, modified_detr_results= process_package_matching(self.pdf_path, modified_detr_results)
        return package_data, data2, have_page, modified_detr_results