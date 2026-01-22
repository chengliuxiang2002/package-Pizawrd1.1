import os
import cv2
import numpy as np
import onnxruntime as rt
from package_core.PackageExtract.BGA_Function.Pin_process.predict import extract_pin_coords, extract_border_coords, \
    process_single_image
from package_core.PackageExtract.yolox_onnx_py.model_paths import model_path, result_path


def filter_valid_son_pins(pin_coords, border_coords):
    """
    筛选SON有效PIN（仅保留Border内部的PIN，外部为无效）
    SON核心特征：引脚位于元件本体的某一侧或两侧，而非分散在四边
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


def classify_son_pins_by_single_side(pin_coords, border_coords):
    """
    修正版：SON引脚按**四个独立侧**分类，不合并侧
    :param pin_coords: 有效PIN坐标列表
    :param border_coords: Border坐标 [left, top, right, bottom]
    :return: 分类结果 {'top': {'pins':[], 'centers':[]}, ..., 'max_side': 'top'}
    """
    left, top, right, bottom = border_coords
    classified = {
        'top': {'pins': [], 'centers': []},
        'bottom': {'pins': [], 'centers': []},
        'left': {'pins': [], 'centers': []},
        'right': {'pins': [], 'centers': []},
        'max_side': ''  # PIN数量最多的侧
    }

    for pin in pin_coords:
        x1, y1, x2, y2 = pin
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 计算到四个侧的距离
        dist_to_top = center_y - top
        dist_to_bottom = bottom - center_y
        dist_to_left = center_x - left
        dist_to_right = right - center_x

        # 找到距离最近的侧（即归属侧）
        min_dist = min(dist_to_top, dist_to_bottom, dist_to_left, dist_to_right)
        if min_dist == dist_to_top:
            classified['top']['pins'].append(pin)
            classified['top']['centers'].append((center_x, center_y))
        elif min_dist == dist_to_bottom:
            classified['bottom']['pins'].append(pin)
            classified['bottom']['centers'].append((center_x, center_y))
        elif min_dist == dist_to_left:
            classified['left']['pins'].append(pin)
            classified['left']['centers'].append((center_x, center_y))
        else:
            classified['right']['pins'].append(pin)
            classified['right']['centers'].append((center_x, center_y))

    # 找出PIN数量最多的侧
    side_counts = {
        'top': len(classified['top']['pins']),
        'bottom': len(classified['bottom']['pins']),
        'left': len(classified['left']['pins']),
        'right': len(classified['right']['pins'])
    }
    classified['max_side'] = max(side_counts, key=side_counts.get)
    print(f"各侧PIN数量: {side_counts}, 选择PIN最多的侧: {classified['max_side']}")

    return classified


def get_son_adjacent_pin_pair_from_single_side(classified_pins):
    """
    修正版：从**同一侧**提取相邻PIN对
    :param classified_pins: classify_son_pins_by_single_side 返回的字典
    :return: 相邻PIN对 [pin1, pin2]（无则None）
    """
    # 获取PIN数量最多的侧
    target_side = classified_pins['max_side']
    target_pins = classified_pins[target_side]['pins']
    target_centers = classified_pins[target_side]['centers']

    # 该侧PIN数量不足2个，无法提取
    if len(target_pins) < 2:
        print(f"警告：{target_side}侧仅{len(target_pins)}个PIN，无法提取相邻对")
        return None

    # 根据侧的类型选择排序依据
    if target_side in ['top', 'bottom']:
        # 水平侧（top/bottom）：按x坐标排序（从左到右）
        combined = sorted(zip(target_pins, target_centers), key=lambda k: k[1][0])
    else:
        # 垂直侧（left/right）：按y坐标排序（从上到下）
        combined = sorted(zip(target_pins, target_centers), key=lambda k: k[1][1])

    # 取中间位置的两个相邻PIN（避免边缘畸变）
    mid_idx = len(combined) // 2
    idx1 = max(0, mid_idx - 1)
    idx2 = idx1 + 1

    adjacent_pair = [combined[idx1][0], combined[idx2][0]]
    print(f"从{target_side}侧提取相邻PIN对：索引 {idx1} 和 {idx2}")
    return adjacent_pair


def save_son_simple_txt(pin_pair, output_path):
    """
    仅保存相邻两个PIN的坐标信息，无其他冗余内容
    格式：[x1,y1,x2,y2],[x1,y1,x2,y2]  或  None
    :param pin_pair: 相邻PIN对 [pin1, pin2]
    :param output_path: 保存路径
    """
    def fmt_pin(p):
        return "[" + ", ".join([f"{v:.2f}" for v in p]) + "]"

    try:
        # 自动创建目录
        dir_name = os.path.dirname(output_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            print(f"目录不存在，已自动创建: {dir_name}")

        with open(output_path, 'w', encoding='utf-8') as f:
            if pin_pair:
                p1_str = fmt_pin(pin_pair[0])
                p2_str = fmt_pin(pin_pair[1])
                f.write(f"{p1_str},{p2_str}\n")
            else:
                f.write("None\n")

        print(f"SON相邻PIN对已保存至: {output_path}")
    except Exception as e:
        print(f"SON TXT保存失败: {str(e)}")


def visualize_son_selected_pair(image_path, border_coords, valid_pins, pin_pair, target_side, output_path):
    """
    可视化：高亮**同一侧**的相邻PIN对
    """
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"警告：无法读取图像用于可视化 {image_path}")
        return

    colors = {
        'border': (0, 0, 255),
        'background_pin': (100, 100, 100),
        'highlight': (255, 0, 255),
        'connector': (0, 255, 255)
    }
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness_thin = 1
    thickness_thick = 2

    # 绘制Border
    if border_coords:
        left, top, right, bottom = map(int, border_coords)
        cv2.rectangle(img, (left, top), (right, bottom), colors['border'], thickness_thick)
        cv2.putText(img, 'SON_Border', (left, top - 10), font, font_scale, colors['border'], thickness_thick)

    # 绘制所有有效PIN
    for pin in valid_pins:
        x1, y1, x2, y2 = map(int, pin)
        cv2.rectangle(img, (x1, y1), (x2, y2), colors['background_pin'], thickness_thin)

    # 高亮选中的相邻PIN对
    get_center = lambda p: (int((p[0] + p[2]) / 2), int((p[1] + p[3]) / 2))
    if pin_pair:
        p1, p2 = pin_pair
        c1 = get_center(p1)
        c2 = get_center(p2)

        cv2.rectangle(img, (int(p1[0]), int(p1[1])), (int(p1[2]), int(p1[3])), colors['highlight'], thickness_thick)
        cv2.rectangle(img, (int(p2[0]), int(p2[1])), (int(p2[2]), int(p2[3])), colors['highlight'], thickness_thick)
        cv2.line(img, c1, c2, colors['connector'], thickness=2)
        cv2.circle(img, c1, 4, colors['highlight'], -1)
        cv2.circle(img, c2, 4, colors['highlight'], -1)

        # 计算间距
        if target_side in ['top', 'bottom']:
            pitch = abs(c1[0] - c2[0])
            pitch_name = "X-Pitch"
        else:
            pitch = abs(c1[1] - c2[1])
            pitch_name = "Y-Pitch"
        text_pos = (int((c1[0] + c2[0]) / 2) - 40, int((c1[1] + c2[1]) / 2) - 20)
        cv2.putText(img, f"{pitch_name}:{pitch:.1f}", text_pos, font, font_scale, colors['connector'], thickness_thick)

    # 绘制统计信息
    stats_text = [
        f"SON Valid Pins: {len(valid_pins)}",
        f"Selected Side: {target_side}"
    ]
    y_offset = 40
    for text in stats_text:
        cv2.putText(img, text, (20, y_offset), font, font_scale + 0.1, (0, 0, 0), thickness_thick + 1)
        cv2.putText(img, text, (20, y_offset), font, font_scale + 0.1, (255, 255, 255), thickness_thick)
        y_offset += 30

    cv2.imencode('.png', img)[1].tofile(output_path)
    print(f"SON可视化结果已保存至：{output_path}")


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
    """通用：PIN去重"""
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

    return deduplicated


def get_SON_res(img_path):
    """加载SON模型并推理"""
    std_h, std_w = 640, 640
    conf_thres = 0.3
    iou_thres = 0.4
    class_config = ['Border', 'PIN', 'PIN_Number']
    ONNX_MODEL_PATH = model_path("yolo_model", "pin_detect", "QFN_pin_detect.onnx")  # 替换为SON专用模型

    try:
        sess = rt.InferenceSession(ONNX_MODEL_PATH)
        print(f"成功加载SON模型：{ONNX_MODEL_PATH}")
    except Exception as e:
        print(f"错误：无法加载SON模型 {ONNX_MODEL_PATH} - {str(e)}")
        exit(1)

    res = process_single_image(
        img_path=img_path,
        output_dir="",
        sess=sess,
        std_h=std_h,
        std_w=std_w,
        class_config=class_config,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        show_image=False
    )
    return res


def SON_extract_pins(img_path):
    """核心函数：从同一侧提取SON相邻PIN对"""
    try:
        res = get_SON_res(img_path)
        raw_pin_coords = extract_pin_coords(res)
        border_coords = extract_border_coords(res)
        deduplicated_pins = deduplicate_pins(raw_pin_coords)

        if border_coords:
            valid_pins, total_valid_pins = filter_valid_son_pins(deduplicated_pins, border_coords)
        else:
            valid_pins = deduplicated_pins
            total_valid_pins = len(valid_pins)

        print(f"\n========== SON引脚统计结果 ==========")
        print(f"原始检测PIN数：{len(raw_pin_coords)}")
        print(f"去重后PIN数：{len(deduplicated_pins)}")
        print(f"SON有效PIN总数：{total_valid_pins}")
        print(f"======================================")

        # 从同一侧提取相邻PIN对
        adj_pair = None
        target_side = ""
        if border_coords and len(valid_pins) >= 2:
            # 按独立侧分类
            classified_pins = classify_son_pins_by_single_side(valid_pins, border_coords)
            target_side = classified_pins['max_side']
            # 从PIN最多的侧提取相邻对
            adj_pair = get_son_adjacent_pin_pair_from_single_side(classified_pins)
            # 保存结果
            txt_output_path = result_path("Package_view", "pin", "SON_adjacent_pin.txt")
            save_son_simple_txt(adj_pair, txt_output_path)
            # 可视化
            # vis_output_path = os.path.splitext(img_path)[0] + "_son_selected_pair.png"
            # visualize_son_selected_pair(img_path, border_coords, valid_pins, adj_pair, target_side, vis_output_path)
        else:
            print("警告：无Border或有效PIN数不足2个，跳过相邻PIN对提取")

        return int(total_valid_pins / 2)
    except Exception as e:
        print(f"PIN处理过程发生错误，{str(e)}")
        return ""


if __name__ == "__main__":
    img_path = r"D:\workspace\PackageWizard1.1\Result\Package_view\page\bottom.jpg"
    pin_nums = SON_extract_pins(img_path)
    print("pin数（单边）：", pin_nums)