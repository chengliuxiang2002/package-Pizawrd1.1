"""封装 F4.6-F4.9 流程的便捷调用入口 (经过并行优化)。"""

from __future__ import annotations
from typing import Iterable
import sys
import os
import threading
from concurrent.futures import ThreadPoolExecutor

# 获取当前脚本所在目录的绝对路径
current_script_path = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
sys.path.append(root_dir)

from package_core.PackageExtract import common_pipeline
from package_core.PackageExtract import function_tool # 导入以进行 Monkey Patch
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

# 导入统一路径管理
try:
    from package_core.PackageExtract.yolox_onnx_py.model_paths import result_path
except ModuleNotFoundError:
    from pathlib import Path
    def result_path(*parts):
        return str(Path(__file__).resolve().parents[3] / 'Result' / Path(*parts))

# --- 线程安全补丁 ---
# 为了防止并行修改 L3 列表时发生冲突，我们给 recite_data 加上线程锁
_l3_lock = threading.Lock()
_original_recite_data = function_tool.recite_data

def _thread_safe_recite_data(L3, key, data):
    """线程安全的 recite_data 包装器"""
    with _l3_lock:
        _original_recite_data(L3, key, data)

# 应用补丁：将 function_tool 中的 recite_data 替换为带锁版本
function_tool.recite_data = _thread_safe_recite_data
# ------------------

def run_f4_pipeline(
    image_root: str,
    package_class: str,
    key: int = 0,
    test_mode: int = 0,
):
    """
    串联执行 F4 阶段的主要函数 (优化版：并行执行几何匹配与线条检测)。
    """

    # 1. 视图扫描
    if os.path.exists(image_root):
        views_items = []
        for item in os.listdir(image_root):
            item_path = os.path.join(image_root, item)
            if os.path.isfile(item_path) and item.lower().endswith(('.jpg', '.jpeg', '.png')):
                view_name = os.path.splitext(item)[0]
                views_items.append(view_name)
        views: Iterable[str] = views_items
    else:
        views: Iterable[str] = common_pipeline.DEFAULT_VIEWS
    print("views:", views)

    # 2. 初始化 L3 (YOLO + DBNet 数据获取) - 此步骤需串行，因为是数据源头
    print("开始测试初始L3集合")
    print(f'图片路径{image_root}')
    L3 = common_pipeline.get_data_location_by_yolo_dbnet(image_root, package_class, view_names=views)

    # 3. 定义并行任务
    # 任务 A: 文本框与分类框的几何匹配 (CPU 密集型)
    def task_geometry_matching():
        print(">> [线程A] 开始几何匹配链路 (F4.1-F4.45)")
        # F4.1: Other
        other_match_dbnet.other_match_boxes_by_overlap(L3)
        # F4.2: PIN
        pin_match_dbnet.PINnum_find_matching_boxes(L3)
        # F4.3: Angle
        angle_match_dbnet.angle_find_matching_boxes(L3)
        # F4.4: Num Match
        num_match_dbnet.num_match_size_boxes(L3)
        # F4.45: Num Direction
        num_direction.add_direction_field_to_yolox_nums(L3)
        print("<< [线程A] 几何匹配链路完成")

    # 任务 B: 线条与箭头检测 (OpenCV/IO 密集型)
    def task_line_processing():
        print(">> [线程B] 开始线条处理链路 (F4.6)")
        # F4.6: Enrich Pairs
        common_pipeline.enrich_pairs_with_lines(L3, image_root, test_mode)
        print("<< [线程B] 线条处理链路完成")

    # 4. 执行并行任务
    print("=== 启动并行处理 ===")
    with ThreadPoolExecutor(max_workers=2) as executor:
        # 同时提交两个任务
        # 注意：L3 是列表(可变对象)，在线程间共享，recite_data 已加锁保护
        f1 = executor.submit(task_geometry_matching)
        f2 = executor.submit(task_line_processing)

        # 等待两个任务都完成
        f1.result()
        f2.result()
    print("=== 并行处理结束 ===")

    # 5. 后续串行流程 (依赖于上述两个任务的结果)
    # F4.7: 需要 F4.45 的 direction 和 F4.6 的 length
    print("开始测试F4.7 (Match Triple Factor)")
    triple_factor = match_triple_factor.match_arrow_pairs_with_yolox(L3, image_root)
    print("*****triple_factor*****:", triple_factor)

    # 预处理文本配对
    L3 = common_pipeline.preprocess_pairs_and_text(L3, key)

    # F4.5: OCR (耗时操作，目前单独运行)
    # 提示：如果 run_svtr_ocr 内部实现了 batch 处理，这里效率会很高
    L3 = common_pipeline.run_svtr_ocr(L3)
    L3 = common_pipeline.normalize_ocr_candidates(L3, key)

    # F4.8: 提取 PIN 序列 & 最终配对
    L3 = common_pipeline.extract_pin_serials(L3, package_class)
    L3 = common_pipeline.match_pairs_with_text(L3, key)

    # F4.9: 清理与计算参数
    L3 = common_pipeline.finalize_pairs(L3)
    parameters, nx, ny = common_pipeline.compute_qfp_parameters(L3)

    # 生成最终参数列表
    parameter_list = get_BGA_parameter_data(parameters, nx, ny)
    print(f"get_BGA_parameter_data 完成, 返回参数列表长度: {len(parameter_list)}")
    print(parameters)

    return parameter_list

if __name__ == "__main__":
    run_f4_pipeline(
        image_root="../../../Result/Package_extract/data",
        package_class="BGA",
        key=0,
        test_mode=0
    )