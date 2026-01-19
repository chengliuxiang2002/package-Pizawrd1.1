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

# print("root_dir =", root_dir)
sys.path.append(root_dir)
# ✅ 关键：把工作目录切到项目根目录（让所有相对路径都按根目录解析）
os.chdir(root_dir)

print("cwd =", os.getcwd())
print("root_dir =", root_dir)
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
    match_triple_factor,
    targeted_ocr
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


from pathlib import Path

log_path = Path("test_data.txt")
# 把 print 的输出复制一份到文件
def tee_print(*args, sep=" ", end="\n"):
    msg = sep.join(map(str, args)) + end
    print(*args, sep=sep, end=end)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(msg)

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
    # print("*****triple_factor*****:", triple_factor)
    ## （整理尺寸线与文本，生成初始配对候选）
    L3 = common_pipeline.preprocess_pairs_and_text(L3, key)
    ## F4.5
    #==================================
    # 5. [Step 1] 先进行 Triple Factor 匹配
    # 此时主要利用 YOLOX 框的位置和箭头进行关联，还没做 OCR
    # print(">> 开始 F4.7 (Match Triple Factor - Location Based)")
    # triple_factor_results = match_triple_factor.match_arrow_pairs_with_yolox(L3, image_root)
    # # 6. [Step 2] 执行定向 OCR (Targeted OCR)
    # # 利用 Triple Factor 里的 YOLOX 大框去跑识别，获取完整的 "0.80x 12=9.60"
    # triple_factor_results = targeted_ocr.run_ocr_on_yolox_locations(triple_factor_results, image_root)
    # # 7. [Step 3] 数据回填 (Overwrite L3)
    # # 用识别好的完整数据，替换掉 L3 里原本可能存在的碎片化 DBNet 数据
    # # 这样后续的 normalize_ocr_candidates 就会处理我们清洗好的数据
    # L3 = targeted_ocr.update_L3_with_yolox_ocr(L3, triple_factor_results)

    L3 = common_pipeline.run_svtr_ocr(L3)
    L3 = common_pipeline.normalize_ocr_candidates(L3, key)
    ## F4.8
    # 0111注释（方便测试）
    L3 = common_pipeline.extract_pin_serials(L3, package_class)
    L3 = common_pipeline.match_pairs_with_text(L3, key)
    ## F4.9
    L3 = common_pipeline.finalize_pairs(L3)
    parameters, nx, ny = common_pipeline.compute_BGA_parameters(L3)

    parameter_list = get_BGA_parameter_data(parameters, nx, ny)
    tee_print(f"get_BGA_parameter_data 完成, 返回参数列表长度: {len(parameter_list)}")
    tee_print(parameter_list)
    tee_print("最终参数结果：",parameters)
    # parameter_list = alter_QFP_parameter_data(parameter_list)

    return parameter_list

if __name__ == "__main__":
    run_f4_pipeline(
       
        image_root=r"D:\\BaiduNetdiskDownload\\PackageWizard1.1\\Result\\Package_extract\\data",
        package_class="BGA",
        key=0,
        test_mode=0
    )
    # 格式：python 脚本路径 >> 输出文件名.txt
# python -u "D:\BaiduNetdiskDownload\PackageWizard1.1\package_core\PackageExtract\BGA_Function\BGA_extract.py" >> console_output.txt
