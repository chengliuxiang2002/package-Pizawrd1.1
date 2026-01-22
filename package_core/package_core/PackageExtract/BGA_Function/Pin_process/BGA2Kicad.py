import os
import time
from typing import List

from package_core.PackageExtract.BGA_Function.Pin_process.BGA_extract_pins import BGA_get_PIN


def generate_kicad_file(
        file_path: str,
        module_name: str,
        col_list: List[str],
        row_list: List[str],
        pitch: float,
        pad_size: float
):
    """
    所见即所得版生成器：
    list[0] 永远放在几何左上角。
    """
    num_cols = len(col_list)
    num_rows = len(row_list)

    # 中心索引
    center_col_idx = (num_cols - 1) / 2.0
    center_row_idx = (num_rows - 1) / 2.0

    # 估算尺寸
    body_w = (num_cols - 1) * pitch + 2.0
    body_h = (num_rows - 1) * pitch + 2.0

    timestamp = hex(int(time.time())).upper().replace('0X', '')

    lines = []
    lines.append(f"(module {module_name} (layer F.Cu) (tedit {timestamp})")
    lines.append(f'  (descr "{module_name} (Visual Match)")')
    lines.append(f'  (tags "BGA {num_rows}x{num_cols}")')
    lines.append('  (attr smd)')

    # --- 绘制丝印 ---
    w_half = body_w / 2
    h_half = body_h / 2
    lines.append(f'  (fp_line (start {-w_half} {-h_half}) (end {w_half} {-h_half}) (layer F.SilkS) (width 0.12))')
    lines.append(f'  (fp_line (start {w_half} {-h_half}) (end {w_half} {h_half}) (layer F.SilkS) (width 0.12))')
    lines.append(f'  (fp_line (start {w_half} {h_half}) (end {-w_half} {h_half}) (layer F.SilkS) (width 0.12))')
    lines.append(f'  (fp_line (start {-w_half} {h_half}) (end {-w_half} {-h_half}) (layer F.SilkS) (width 0.12))')

    # 画一个小三角/圆圈在左上角 (Pin 1 Position)
    # 在这个模式下，左上角对应的就是 row_list[0] 和 col_list[0]
    # 我们标记物理左上角
    lines.append(
        f'  (fp_circle (center {-w_half} {-h_half}) (end {-w_half + 0.2} {-h_half}) (layer F.SilkS) (width 0.15))')

    # --- 生成焊盘 (Pads) ---
    for r_idx, r_name in enumerate(row_list):
        # Y坐标: r_idx=0 (list第一个元素) -> 结果为负 (Top/KiCad上方)
        y_pos = (r_idx - center_row_idx) * pitch

        for c_idx, c_name in enumerate(col_list):
            # X坐标: c_idx=0 (list第一个元素) -> 结果为负 (Left/KiCad左侧)
            x_pos = (c_idx - center_col_idx) * pitch

            pad_name = f"{r_name}{c_name}"

            lines.append(
                f'  (pad {pad_name} smd circle (at {x_pos:.3f} {y_pos:.3f}) '
                f'(size {pad_size} {pad_size}) (layers F.Cu F.Paste F.Mask))'
            )

    lines.append(')')

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"✅ [文件生成] 已保存: {os.path.abspath(file_path)}")


# ================= 调用示例 =================
# 在你的 if __name__ == "__main__": 中调用

if __name__ == "__main__":
    # image_path = r"D:\workspace\PackageWizard1.1\Result/Package_view/page/bottom.jpg"
    image_path = r"D:\workspace\RT-DETR-Test\images\test\3.jpg"
    # 可通过cluster_size_threshold参数调整触发阈值（默认20）
    sorted_h_text, sorted_v_text, X, Y, final_col_list, final_row_list, a1_corner,_ = BGA_get_PIN(image_path, visualize=True)

    # 只有当检测成功时才生成
    if X and Y:
        # 定义物理参数
        PITCH_MM = 0.8
        PAD_MM = 0.45

        # 自动命名
        module_name = f"BGA_{X}x{Y}_{final_row_list[0]}{final_col_list[0]}_{final_row_list[-1]}{final_col_list[-1]}"
        save_path = f"{module_name}.kicad_mod"

        # 生成
        generate_kicad_file(
            file_path=save_path,
            module_name=module_name,
            col_list=final_col_list,  # ['1', '2' ... '8']
            row_list=final_row_list,  # ['A', 'B' ... 'H']
            pitch=PITCH_MM,
            pad_size=PAD_MM
        )

