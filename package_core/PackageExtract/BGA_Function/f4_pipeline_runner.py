"""封装 F4.6-F4.9 流程的便捷调用入口。"""

from __future__ import annotations
from typing import Iterable
import sys
import os

# 获取当前脚本所在目录的绝对路径
current_script_path = os.path.abspath(__file__)
# 计算项目根目录（PackageWizard1.1）的路径：从当前脚本目录向上退3级
# （当前脚本在 BGA_Function/ 下，上级是 PackageExtract/，再上级是 package_core/，再上级就是根目录）
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
# 将根目录添加到Python的搜索路径中
sys.path.append(root_dir)
# # 打印关键信息用于排查
# print("当前脚本路径：", current_script_path)
# print("计算出的根目录：", root_dir)
# print("Python搜索路径：", sys.path)  # 查看root_dir是否已被添加
from package_core.PackageExtract import common_pipeline
import os
from package_core.PackageExtract.function_tool import (
    get_BGA_parameter_data
)
from package_core.PackageExtract.BGA_Function.pre_extract import (
    other_match_dbnet,
    pin_match_dbnet,
    angle_match_dbnet,
    num_match_dbnet,
    num_direction,
    match_triple_factor
)
# from package_core.PackageExtract.BGA_Function import fill_triple_factor

# 导入统一路径管理
try:
    from package_core.PackageExtract.yolox_onnx_py.model_paths import result_path
except ModuleNotFoundError:
    from pathlib import Path
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
def run_f4_pipeline(
    image_root: str,
    package_class: str,
    key: int = 0,
    test_mode: int = 0,
):
    """串联执行 F4 阶段的主要函数，返回参数列表与中间结果。

    :param image_root: 存放 ``top/bottom/side/detailed`` 视图图片的目录。
    :param package_class: 封装类型，例如 ``"QFP"``、``"BGA"``。
    :param key: 与历史实现一致的流程参数，用于控制 OCR 清洗策略。
    :param test_mode: 传递给 ``find_pairs_length`` 的调试开关。
    :param view_names: 自定义视图顺序；默认为 ``common_pipeline.DEFAULT_VIEWS``。
    :returns: ``dict``，包含 ``L3`` 数据、参数候选列表以及 ``nx``/``ny``。
    """

    # 从 image_root 获取视图名称（支持目录和图片文件）
    if os.path.exists(image_root):
        views_items = []
        for item in os.listdir(image_root):
            item_path = os.path.join(image_root, item)
            if os.path.isfile(item_path) and item.lower().endswith(('.jpg', '.jpeg', '.png')):
                # 去掉文件扩展名作为视图名称
                view_name = os.path.splitext(item)[0]
                views_items.append(view_name)
        views: Iterable[str] = views_items
    else:
        views: Iterable[str] = common_pipeline.DEFAULT_VIEWS
    print("views:", views)
    ## 初始化合并L1L2构建L3
    print("开始测试初始L3集合")
    print(f'图片路径{image_root}')
    L3 = common_pipeline.get_data_location_by_yolo_dbnet(image_root, package_class, view_names=views)


    ## F4.1-F4.4
    print("开始测试F4.1")
    L3 = other_match_dbnet.other_match_boxes_by_overlap(L3)
    ## F4.2
    print("开始测试F4.2")
    L3 = pin_match_dbnet.PINnum_find_matching_boxes(L3)
    print("开始测试F4.3")
    L3 = angle_match_dbnet.angle_find_matching_boxes(L3)
    print("开始测试F4.4")
    L3 = num_match_dbnet.num_match_size_boxes(L3)
    ## F4.45（添加方向字段）
    print("开始测试F4.45")
    L3 = num_direction.add_direction_field_to_yolox_nums(L3)
    ## F4.6
    print("开始测试F4.6")
    L3 = common_pipeline.enrich_pairs_with_lines(L3, image_root, test_mode)
    ## F4.7
    print("开始测试F4.7")
    triple_factor = match_triple_factor.match_arrow_pairs_with_yolox(L3, image_root)
    print("*****triple_factor*****:", triple_factor)
    ## （整理尺寸线与文本，生成初始配对候选）
    L3 = common_pipeline.preprocess_pairs_and_text(L3, key)
    ## F4.5
    L3 = common_pipeline.run_svtr_ocr(L3)
    L3 = common_pipeline.normalize_ocr_candidates(L3, key)
    ## 将 OCR 识别结果填入 triple_factor
    # print("开始填充 triple_factor 中的 OCR 识别结果")
    # triple_factor = fill_triple_factor.fill_triple_factor_with_ocr(L3, triple_factor)
    # print("*****填充后的 triple_factor*****:", triple_factor)
    ## F4.8
    L3 = common_pipeline.extract_pin_serials(L3, package_class)
    L3 = common_pipeline.match_pairs_with_text(L3, key)
    ## F4.9
    L3 = common_pipeline.finalize_pairs(L3)
    parameters, nx, ny = common_pipeline.compute_qfp_parameters(L3)


    # L3 = common_pipeline.remove_other_annotations(L3)
    # L3 = common_pipeline.enrich_pairs_with_lines(L3, image_root, test_mode)
    # L3 = common_pipeline.preprocess_pairs_and_text(L3, key)
    # L3 = common_pipeline.run_svtr_ocr(L3)
    # L3 = common_pipeline.normalize_ocr_candidates(L3, key)
    # L3 = common_pipeline.extract_pin_serials(L3, package_class)
    # L3 = common_pipeline.match_pairs_with_text(L3, key)
    # L3 = common_pipeline.finalize_pairs(L3)
    # parameters, nx, ny = common_pipeline.compute_qfp_parameters(L3)

    parameter_list = get_BGA_parameter_data(parameters, nx, ny)
    print(f"get_BGA_parameter_data 完成, 返回参数列表长度: {len(parameter_list)}")
    print(parameters)
    # parameter_list = alter_QFP_parameter_data(parameter_list)

    return parameter_list

if __name__ == "__main__":
    run_f4_pipeline(
        image_root="../../../Result/Package_extract/data",
        package_class="BGA",
        key=0,
        test_mode=0
    )
    # 格式：python 脚本路径 >> 输出文件名.txt
# python -u "d:\cc\PackageWizard1.1\package_core\PackageExtract\BGA_Function\f4_pipeline_runner.py" >> console_output.txt
