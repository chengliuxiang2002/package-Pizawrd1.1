import os
import cv2
import numpy as np
import onnxruntime as rt
from package_core.PackageExtract.BGA_Function.Pin_process.predict import extract_pin_coords, extract_border_coords, \
    process_single_image
from package_core.PackageExtract.yolox_onnx_py.model_paths import model_path


def filter_valid_son_pins(pin_coords, border_coords):
    """
    筛选SON有效PIN（仅保留Border内部的PIN，外部为无效）
    SON核心特征：引脚位于元件本体（Border框内），外部无有效引脚
    :param pin_coords: 所有PIN坐标列表 [[x1,y1,x2,y2], ...]
    :param border_coords: Border坐标 [left, top, right, bottom]
    :return: 有效PIN坐标列表、有效PIN总数
    """
    left, top, right, bottom = border_coords
    valid_pins = []

    for pin in pin_coords:
        x1, y1, x2, y2 = pin
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # SON有效PIN判定：中心点在Border内部（包含边界）
        if left <= center_x <= right and top <= center_y <= bottom:
            valid_pins.append(pin)

    total_valid = len(valid_pins)
    return valid_pins, total_valid


def calculate_overlap_ratio(box1, box2):
    """通用：计算两个框的重叠率（以小框面积为参考）"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    if inter_area == 0:
        return 0.0, False

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    small_area = min(area1, area2)
    overlap_ratio = inter_area / small_area
    is_small_in_large = (inter_area == small_area)

    return overlap_ratio, is_small_in_large


def deduplicate_pins(pin_coords, overlap_threshold=0.8, conf_scores=None):
    """通用：PIN去重（保留核心逻辑，不修改）"""
    if not isinstance(pin_coords, list) or len(pin_coords) == 0:
        print("警告：PIN坐标列表为空或格式错误，直接返回原数据")
        return pin_coords
    for idx, pin in enumerate(pin_coords):
        if not isinstance(pin, (list, tuple)) or len(pin) != 4:
            raise ValueError(f"错误：第{idx}个PIN坐标格式错误！应为[x1,y1,x2,y2]，实际为{pin}")

    if len(pin_coords) <= 1:
        return pin_coords

    if conf_scores is not None and len(conf_scores) == len(pin_coords):
        sorted_pairs = sorted(zip(pin_coords, conf_scores), key=lambda x: x[1], reverse=True)
        sorted_pins = [pair[0] for pair in sorted_pairs]
    else:
        sorted_pins = sorted(pin_coords, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)

    deduplicated = []
    for current_pin in sorted_pins:
        is_duplicate = False
        current_area = (current_pin[2] - current_pin[0]) * (current_pin[3] - current_pin[1])

        for kept_pin in deduplicated:
            kept_area = (kept_pin[2] - kept_pin[0]) * (kept_pin[3] - kept_pin[1])
            overlap_ratio, is_small_in_large = calculate_overlap_ratio(current_pin, kept_pin)

            if (overlap_ratio >= overlap_threshold or is_small_in_large) and current_area <= kept_area:
                is_duplicate = True
                break

        if not is_duplicate:
            deduplicated.append(current_pin)

    # print(f"PIN去重完成：原始{len(pin_coords)}个 → 去重后{len(deduplicated)}个")
    return deduplicated


def visualize_son_total_pins(image_path, border_coords, valid_pins, total_pins, output_path):
    """简化可视化：只绘制Border和所有有效PIN（统一颜色），显示总数"""
    # 读取图像（支持中文路径）
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"警告：无法读取图像 {image_path}")
        return

    # 颜色配置（简化：Border红、有效PIN绿）
    colors = {
        'border': (0, 0, 255),  # 红色：Border框
        'valid': (0, 255, 0)  # 绿色：有效PIN（SON内部）
    }
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    box_thickness = 2

    # 1. 绘制Border框（红色粗线）
    if border_coords:
        left, top, right, bottom = border_coords
        cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)),
                      colors['border'], thickness=3)
        cv2.putText(img, 'SON_Border', (int(left) + 10, int(top) + 25),
                    font, font_scale, colors['border'], font_thickness)

    # 2. 绘制所有有效PIN（统一绿色，标注序号）
    for i, pin in enumerate(valid_pins):
        x1, y1, x2, y2 = map(int, pin)
        # 绘制PIN框
        cv2.rectangle(img, (x1, y1), (x2, y2), colors['valid'], box_thickness)
        # 标注PIN序号（避免超出图像）
        label_text = f"PIN_{i + 1}"
        label_pos = (x1 + 5, y1 + 20) if y1 + 20 < img.shape[0] else (x1 + 5, y1 - 10)
        # 绘制标签背景（提高可读性）
        text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
        cv2.rectangle(img, (label_pos[0], label_pos[1] - text_size[1] - 3),
                      (label_pos[0] + text_size[0], label_pos[1] + 3),
                      colors['valid'], -1)
        cv2.putText(img, label_text, label_pos, font, font_scale,
                    (255, 255, 255), font_thickness)

    # 3. 绘制总数统计（突出显示）
    stats_text = [
        f"SON Total Valid Pins: {total_pins}",
        f"Detected & Deduplicated"
    ]
    y_offset = 40
    for text in stats_text:
        # 黑色描边+白色填充，突出显示
        cv2.putText(img, text, (20, y_offset), font, font_scale + 0.1,
                    (0, 0, 0), font_thickness + 1)
        cv2.putText(img, text, (20, y_offset), font, font_scale + 0.1,
                    (255, 255, 255), font_thickness)
        y_offset += 30

    # 保存图像（支持中文路径）
    cv2.imencode('.png', img)[1].tofile(output_path)
    print(f"SON可视化结果已保存至：{output_path}")

    # 显示图像（可选）
    cv2.imshow('SON Total Pins', img)
    print("按任意键关闭图像窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_SON_res(img_path):
    """加载SON模型并推理（逻辑不变）"""
    std_h, std_w = 640, 640
    conf_thres = 0.2
    iou_thres = 0.2
    class_config = ['Border', 'PIN', 'PIN_Number']
    ONNX_MODEL_PATH = model_path("yolo_model", "pin_detect", "QFN_pin_detect.onnx")  # 替换为你的SOP模型路径

    # 加载模型
    try:
        sess = rt.InferenceSession(ONNX_MODEL_PATH)
        print(f"成功加载SON模型：{ONNX_MODEL_PATH}")
    except Exception as e:
        print(f"错误：无法加载SON模型 {ONNX_MODEL_PATH} - {str(e)}")
        exit(1)

    res  = process_single_image(
        img_path=img_path,
        output_dir="",
        sess=sess,
        std_h=std_h,
        std_w=std_w,
        class_config=class_config,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        show_image=True
    )
    return res


def SON_extract_pins(img_path):
    """核心函数：仅提取SON有效PIN的总数（无分边）"""
    try:
        # 1. 模型推理
        res = get_SON_res(img_path)
        # 2. 提取原始PIN和Border坐标
        raw_pin_coords = extract_pin_coords(res)
        border_coords = extract_border_coords(res)
        # 3. PIN去重（避免重复检测导致总数错误）
        deduplicated_pins = deduplicate_pins(raw_pin_coords)
        # 4. 筛选SON有效PIN（Border内部）
        if border_coords:
            valid_pins, total_valid_pins = filter_valid_son_pins(deduplicated_pins, border_coords)
        else:
            valid_pins = deduplicated_pins
            total_valid_pins = len(valid_pins)

        # 输出最终总数（核心结果）
        # print(f"\n========== SON引脚统计结果 ==========")
        # print(f"原始检测PIN数：{len(raw_pin_coords)}")
        # print(f"去重后PIN数：{len(deduplicated_pins)}")
        print(f"SON有效PIN总数：{total_valid_pins}")
        # print(f"======================================")

        # 5. 可视化（可选，如需关闭可注释）
        # visualization_output_path = "son_total_pins_visualization.png"
        # visualize_son_total_pins(img_path, border_coords, valid_pins, total_valid_pins, visualization_output_path)

        return int(total_valid_pins/2)
    except Exception as e:
        print(f"PIN处理过程发生错误，{str(e)}")
        return ""



if __name__ == "__main__":
    # 替换为你的SON图像路径
    img_path = r"D:\workspace\PackageWizard1.1\Result\Package_view\page\bottom.jpg"
    pin_nums = SON_extract_pins(img_path)
    print("pin总数：", pin_nums)