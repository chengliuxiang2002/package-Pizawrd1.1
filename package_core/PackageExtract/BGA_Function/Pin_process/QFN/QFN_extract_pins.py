import os
import cv2
import numpy as np
import onnxruntime as rt

from package_core.PackageExtract.BGA_Function.Pin_process.predict import extract_pin_coords, extract_border_coords, \
    process_single_image
from package_core.PackageExtract.yolox_onnx_py.model_paths import model_path, result_path


def classify_pins_by_border(pin_coords, border_coords):
    """
    根据Border将PIN分边（上下左右），并过滤无效PIN
    :param pin_coords: 所有PIN的坐标列表，格式为[[x1,y1,x2,y2], ...]
    :param border_coords: Border坐标，格式为[left, top, right, bottom]
    :return: 分边结果字典，包含上下左右边的PIN、无效PIN及各PIN的中心点
    """
    left, top, right, bottom = border_coords
    classified = {
        'top': {'pins': [], 'centers': []},  # 上边PIN及中心点
        'bottom': {'pins': [], 'centers': []},  # 下边PIN及中心点
        'left': {'pins': [], 'centers': []},  # 左边PIN及中心点
        'right': {'pins': [], 'centers': []},  # 右边PIN及中心点
        'invalid': {'pins': [], 'centers': []}  # 无效PIN及中心点
    }

    for pin in pin_coords:
        x1, y1, x2, y2 = pin
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 过滤无效PIN
        if not (left <= center_x <= right and top <= center_y <= bottom):
            classified['invalid']['pins'].append(pin)
            classified['invalid']['centers'].append((center_x, center_y))
            continue

        # 计算到四边距离
        dist_to_top = center_y - top
        dist_to_bottom = bottom - center_y
        dist_to_left = center_x - left
        dist_to_right = right - center_x

        # 确定归属边
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

    return classified


def calculate_overlap_ratio(box1, box2):
    """
    计算两个框的重叠率（以小框面积为参考）
    :param box1: 框1 [x1,y1,x2,y2]
    :param box2: 框2 [x1,y1,x2,y2]
    :return: overlap_ratio（重叠率=交集面积÷小框面积），is_small_in_large（是否小框在大框内）
    """
    # 计算交集坐标
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # 交集面积（无交集则为0）
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    if inter_area == 0:
        return 0.0, False

    # 计算两个框的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 确定小框面积（以小框为参考）
    small_area = min(area1, area2)
    large_area = max(area1, area2)

    # 重叠率=交集面积÷小框面积（反映小框被大框覆盖的比例）
    overlap_ratio = inter_area / small_area

    # 判断是否小框完全在大框内（交集面积=小框面积）
    is_small_in_large = (inter_area == small_area)

    return overlap_ratio, is_small_in_large


def deduplicate_pins(pin_coords, overlap_threshold=0.8, conf_scores=None):
    """
    优化版PIN去重：以小框面积为参考的重叠率≥0.8，即判定重复（更贴合去重需求）
    :param pin_coords: PIN坐标列表（二维列表）
    :param overlap_threshold: 重叠率阈值（默认0.8，小框80%以上面积与大框重叠则去除）
    :param conf_scores: 可选，置信度列表（高置信度优先保留）
    :return: 去重后的PIN坐标列表
    """
    # 输入格式校验
    if not isinstance(pin_coords, list) or len(pin_coords) == 0:
        print("警告：PIN坐标列表为空或格式错误，直接返回原数据")
        return pin_coords
    for idx, pin in enumerate(pin_coords):
        if not isinstance(pin, (list, tuple)) or len(pin) != 4:
            raise ValueError(f"错误：第{idx}个PIN坐标格式错误！应为[x1,y1,x2,y2]，实际为{pin}")

    if len(pin_coords) <= 1:
        return pin_coords  # 无重复，直接返回

    # 排序：优先保留更优PIN（高置信度→大框）
    if conf_scores is not None and len(conf_scores) == len(pin_coords):
        # 置信度降序（高置信度优先）
        sorted_pairs = sorted(zip(pin_coords, conf_scores), key=lambda x: x[1], reverse=True)
        sorted_pins = [pair[0] for pair in sorted_pairs]
    else:
        # 面积降序（大框优先，避免保留小噪声框）
        sorted_pins = sorted(pin_coords, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)

    # 去重核心逻辑
    deduplicated = []
    for current_pin in sorted_pins:
        is_duplicate = False
        current_area = (current_pin[2] - current_pin[0]) * (current_pin[3] - current_pin[1])

        for kept_pin in deduplicated:
            kept_area = (kept_pin[2] - kept_pin[0]) * (kept_pin[3] - kept_pin[1])

            # 计算重叠率（以小框面积为参考）和是否小框在大框内
            overlap_ratio, is_small_in_large = calculate_overlap_ratio(current_pin, kept_pin)

            # 判定规则（满足任一即视为重复）：
            # 1. 重叠率≥阈值（小框80%以上面积与大框重叠）；
            # 2. 小框完全在大框内（即使重叠率刚好略低于阈值，也视为噪声）
            if overlap_ratio >= overlap_threshold or is_small_in_large:
                # 仅去除小框（当前PIN是小框时才标记为重复）
                if current_area <= kept_area:
                    is_duplicate = True
                break

        if not is_duplicate:
            deduplicated.append(current_pin)

    # 打印去重信息
    print(
        f"PIN去重完成：原始{len(pin_coords)}个 → 去重后{len(deduplicated)}个")
    return deduplicated

def visualize_classified_pins(image_path, border_coords, classified_pins, output_path):
    """
    可视化分边结果：绘制Border框、不同颜色的分边PIN框及标签
    :param image_path: 原始图像路径
    :param border_coords: Border坐标 [left, top, right, bottom]
    :param classified_pins: 分边结果字典（classify_pins_by_border的输出）
    :param output_path: 可视化结果保存路径
    """
    # 读取图像（支持中文路径）
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"警告：无法读取图像 {image_path}")
        return

    # 定义颜色（BGR格式）和字体
    colors = {
        'border': (0, 0, 255),  # 红色：Border框
        'top': (0, 255, 0),  # 绿色：上边PIN
        'bottom': (0, 165, 255),  # 橙色：下边PIN
        'left': (255, 0, 0),  # 蓝色：左边PIN
        'right': (128, 0, 128),  # 紫色：右边PIN
        'invalid': (128, 128, 128)  # 灰色：无效PIN
    }
    labels = {
        'top': 'Top',
        'bottom': 'Bottom',
        'left': 'Left',
        'right': 'Right',
        'invalid': 'Invalid'
    }
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    box_thickness = 2

    # 1. 绘制Border框（红色粗线）
    left, top, right, bottom = border_coords
    cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)),
                  colors['border'], thickness=3)
    # 标注Border标签
    cv2.putText(img, 'Border', (int(left) + 10, int(top) + 25),
                font, font_scale, colors['border'], font_thickness)

    # 2. 绘制各边PIN框和标签
    for edge in ['top', 'bottom', 'left', 'right', 'invalid']:
        pins = classified_pins[edge]['pins']
        centers = classified_pins[edge]['centers']

        for i, (pin, center) in enumerate(zip(pins, centers)):
            x1, y1, x2, y2 = pin
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            center_x, center_y = int(center[0]), int(center[1])

            # 绘制PIN框
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[edge], thickness=box_thickness)

            # 标注边名称（避免遮挡，放在PIN框右上角）
            label_text = f"{labels[edge]}_{i + 1}"
            label_pos = (x2 + 5, y1 + 15)
            # 绘制标签背景（提高可读性）
            text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
            cv2.rectangle(img, (label_pos[0], label_pos[1] - text_size[1] - 3),
                          (label_pos[0] + text_size[0], label_pos[1] + 3),
                          colors[edge], -1)  # -1表示填充
            cv2.putText(img, label_text, label_pos, font, font_scale,
                        (255, 255, 255), font_thickness)  # 白色文字

    # 3. 绘制统计信息（图像左上角）
    stats_text = [
        f"Total PINs: {sum(len(classified_pins[edge]['pins']) for edge in classified_pins.keys())}",
        f"Top: {len(classified_pins['top']['pins'])}",
        f"Bottom: {len(classified_pins['bottom']['pins'])}",
        f"Left: {len(classified_pins['left']['pins'])}",
        f"Right: {len(classified_pins['right']['pins'])}",
        f"Invalid: {len(classified_pins['invalid']['pins'])}"
    ]
    y_offset = 30
    for text in stats_text:
        cv2.putText(img, text, (10, y_offset), font, font_scale,
                    (0, 0, 0), font_thickness + 1)  # 黑色边框
        cv2.putText(img, text, (10, y_offset), font, font_scale,
                    (255, 255, 255), font_thickness)  # 白色文字
        y_offset += 20

    # 保存图像（支持中文路径）
    cv2.imencode('.png', img)[1].tofile(output_path)
    print(f"可视化结果已保存至：{output_path}")

    # 显示图像（可选）
    cv2.imshow('PIN Classification Visualization', img)
    print("按任意键关闭图像窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_QFN_res(img_path):
    # 基础配置参数（不变）
    std_h, std_w = 640, 640
    conf_thres = 0.5
    iou_thres = 0.6
    class_config = ['Border', 'PIN', 'PIN_Number']
    img_path = img_path
    ONNX_MODEL_PATH = model_path("yolo_model","pin_detect", "QFN_pin_detect.onnx")
    # 加载模型
    try:
        sess = rt.InferenceSession(ONNX_MODEL_PATH)
        print(f"成功加载模型：{ONNX_MODEL_PATH}")
    except Exception as e:
        print(f"错误：无法加载模型 {ONNX_MODEL_PATH} - {str(e)}")
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
        show_image=False,
    )
    return res


def get_adjacent_pin_pairs(classified_pins):
    """
    获取X方向和Y方向上最佳的两个相邻PIN坐标
    :param classified_pins: classify_pins_by_border 返回的字典
    :return: 字典 {'x_pair': [pin1, pin2], 'y_pair': [pin1, pin2]}，如果无法获取则为None
    """
    result = {
        'x_pair': None,
        'y_pair': None
    }

    # --- 处理 X 方向 (从 Top 或 Bottom 中取) ---
    # 优先选择检测到PIN数量较多的一边，数据更可靠
    top_cnt = len(classified_pins['top']['pins'])
    btm_cnt = len(classified_pins['bottom']['pins'])

    target_edge_x = 'top' if top_cnt >= btm_cnt else 'bottom'
    pins_x = classified_pins[target_edge_x]['pins']

    # 必须至少有2个PIN才能找相邻
    if len(pins_x) >= 2:
        # 将 box 和 center 组合在一起以便排序
        # box: [x1, y1, x2, y2], center: (cx, cy)
        combined_x = list(zip(pins_x, classified_pins[target_edge_x]['centers']))
        # 按中心点 x 坐标排序 (从左到右)
        combined_x.sort(key=lambda k: k[1][0])

        # 为了精度，尽量取靠近中间的两个相邻PIN（避免镜头边缘畸变）
        mid_idx = len(combined_x) // 2
        # 如果是偶数长度，取 mid-1 和 mid；如果是奇数，取 mid 和 mid+1
        # 这里简化处理：只要不是最后两个即可，取中间最稳妥
        idx1 = max(0, mid_idx - 1)
        idx2 = idx1 + 1

        result['x_pair'] = [combined_x[idx1][0], combined_x[idx2][0]]
        print(f"X方向取值 ({target_edge_x}边): 索引 {idx1} 和 {idx2}")

    # --- 处理 Y 方向 (从 Left 或 Right 中取) ---
    left_cnt = len(classified_pins['left']['pins'])
    right_cnt = len(classified_pins['right']['pins'])

    target_edge_y = 'left' if left_cnt >= right_cnt else 'right'
    pins_y = classified_pins[target_edge_y]['pins']

    if len(pins_y) >= 2:
        combined_y = list(zip(pins_y, classified_pins[target_edge_y]['centers']))
        # 按中心点 y 坐标排序 (从上到下)
        combined_y.sort(key=lambda k: k[1][1])

        mid_idx = len(combined_y) // 2
        idx1 = max(0, mid_idx - 1)
        idx2 = idx1 + 1

        result['y_pair'] = [combined_y[idx1][0], combined_y[idx2][0]]
        print(f"Y方向取值 ({target_edge_y}边): 索引 {idx1} 和 {idx2}")

    return result
def visualize_selected_pairs(image_path, border_coords, classified_pins, adj_pairs, output_path):
    """
    可视化：在原图上标出Border、所有PIN，并高亮显示用于计算Pitch的相邻PIN对及其间距。
    """
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"警告：无法读取图像用于可视化 {image_path}")
        return

    # --- 定义样式 ---
    colors = {
        'border': (0, 0, 255),        # 红色: Border
        'background_pin': (100, 100, 100), # 灰色: 普通PIN背景
        'x_highlight': (255, 255, 0), # 青色: X方向选中对
        'y_highlight': (255, 0, 255), # 品红色: Y方向选中对
        'connector': (0, 255, 255)    # 黄色: 连接线
    }
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness_thin = 1
    thickness_thick = 2

    # --- 1. 绘制基础 Border ---
    left, top, right, bottom = map(int, border_coords)
    cv2.rectangle(img, (left, top), (right, bottom), colors['border'], thickness_thick)
    cv2.putText(img, 'Border', (left, top - 10), font, font_scale, colors['border'], thickness_thick)

    # --- 2. 绘制所有 PIN 作为背景 (细灰线) ---
    for edge in classified_pins:
        for pin in classified_pins[edge]['pins']:
            x1, y1, x2, y2 = map(int, pin)
            cv2.rectangle(img, (x1, y1), (x2, y2), colors['background_pin'], thickness_thin)

    # 辅助函数：获取PIN中心点
    get_center = lambda p: (int((p[0] + p[2]) / 2), int((p[1] + p[3]) / 2))

    # --- 3. 高亮 X 方向选中对 ---
    if adj_pairs['x_pair']:
        p1, p2 = adj_pairs['x_pair']
        c1 = get_center(p1)
        c2 = get_center(p2)

        # 绘制高亮框
        cv2.rectangle(img, (int(p1[0]), int(p1[1])), (int(p1[2]), int(p1[3])), colors['x_highlight'], thickness_thick)
        cv2.rectangle(img, (int(p2[0]), int(p2[1])), (int(p2[2]), int(p2[3])), colors['x_highlight'], thickness_thick)

        # 绘制连接线和中心点
        cv2.line(img, c1, c2, colors['connector'], thickness=2)
        cv2.circle(img, c1, 4, colors['x_highlight'], -1)
        cv2.circle(img, c2, 4, colors['x_highlight'], -1)

        # 计算并标注 Pitch
        x_pitch = abs(c1[0] - c2[0])
        text_pos = (int((c1[0] + c2[0]) / 2) - 40, int((c1[1] + c2[1]) / 2) - 20)
        cv2.putText(img, f"X-Pitch:{x_pitch:.1f}", text_pos, font, font_scale, colors['connector'], thickness_thick)

    # --- 4. 高亮 Y 方向选中对 ---
    if adj_pairs['y_pair']:
        p1, p2 = adj_pairs['y_pair']
        c1 = get_center(p1)
        c2 = get_center(p2)

        # 绘制高亮框
        cv2.rectangle(img, (int(p1[0]), int(p1[1])), (int(p1[2]), int(p1[3])), colors['y_highlight'], thickness_thick)
        cv2.rectangle(img, (int(p2[0]), int(p2[1])), (int(p2[2]), int(p2[3])), colors['y_highlight'], thickness_thick)

        # 绘制连接线和中心点
        cv2.line(img, c1, c2, colors['connector'], thickness=2)
        cv2.circle(img, c1, 4, colors['y_highlight'], -1)
        cv2.circle(img, c2, 4, colors['y_highlight'], -1)

        # 计算并标注 Pitch
        y_pitch = abs(c1[1] - c2[1])
        # 文字位置稍微错开一点，防止和X重叠
        text_pos = (int((c1[0] + c2[0]) / 2) + 10, int((c1[1] + c2[1]) / 2))
        cv2.putText(img, f"Y-Pitch:{y_pitch:.1f}", text_pos, font, font_scale, colors['connector'], thickness_thick)

    # 保存结果
    # cv2.imencode('.png', img)[1].tofile(output_path)
    # print(f"可视化结果已保存至：{output_path}")
    # 如果需要在运行时的电脑上查看，取消下面注释
    # cv2.imshow("Selected Pairs Visualization", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



def save_simple_txt(adj_pairs, output_path):
    """
    按指定格式保存，并自动创建不存在的目录：
    X: [x1,y1,x2,y2],[x1,y1,x2,y2]
    Y: [x1,y1,x2,y2],[x1,y1,x2,y2]
    """

    def fmt_pin(p):
        return "[" + ", ".join([f"{v:.2f}" for v in p]) + "]"

    try:
        # === 新增逻辑：检查并创建目录 ===
        dir_name = os.path.dirname(output_path)
        # 如果路径中有目录部分，且该目录不存在，则创建
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            print(f"目录不存在，已自动创建: {dir_name}")
        # ==============================

        with open(output_path, 'w', encoding='utf-8') as f:
            # --- 写入 X ---
            if adj_pairs['x_pair']:
                p1_str = fmt_pin(adj_pairs['x_pair'][0])
                p2_str = fmt_pin(adj_pairs['x_pair'][1])
                f.write(f"X: {p1_str},{p2_str}\n")
            else:
                f.write("X: None\n")

            # --- 写入 Y ---
            if adj_pairs['y_pair']:
                p1_str = fmt_pin(adj_pairs['y_pair'][0])
                p2_str = fmt_pin(adj_pairs['y_pair'][1])
                f.write(f"Y: {p1_str},{p2_str}\n")

        print(f"TXT 已保存至: {output_path}")
    except Exception as e:
        print(f"保存失败: {str(e)}")

def QFN_extract_pins(img_path):
    try:
        res = get_QFN_res(img_path)
        pin_coords = extract_pin_coords(res)
        border_coords = extract_border_coords(res)
        # 去重
        pin_coords = deduplicate_pins(pin_coords)

        # 打印原始数据
        # print("\n=== 原始数据 ===")
        # print(f"所有PIN坐标（共{len(pin_coords)}个）：")
        # print(pin_coords)
        # print(f"\nBorder坐标（left, top, right, bottom）：")
        # print(border_coords)
        # PIN分边处理
        classified_pins = classify_pins_by_border(pin_coords, border_coords)
        # print(classified_pins)
        X = max(len(classified_pins['top']['pins']), len(classified_pins['bottom']['pins']))
        Y = max(len(classified_pins['left']['pins']), len(classified_pins['right']['pins']))
        # if X==0:
        #     X=None
        # if Y==0:
        #     Y=None
        print(f"最终结果：X={X}, Y={Y}")


        # # 打印分边结果
        # print("\n=== PIN分边结果 ===")
        # for edge in ['top', 'bottom', 'left', 'right', 'invalid']:
        #     count = len(classified_pins[edge]['pins'])
        #     print(f"{edge}边PIN（共{count}个）：")
        #     print(classified_pins[edge]['pins'])

        # === 获取相邻PIN坐标 ===
        adj_pairs = get_adjacent_pin_pairs(classified_pins)
        txt_output_path = result_path("Package_view","pin","QFN_adjacent_pins.txt")
        save_simple_txt(adj_pairs, txt_output_path)

        # # 简单的打印 Pitch 信息 (可选)
        # if adj_pairs['x_pair']:
        #     p1, p2 = adj_pairs['x_pair']
        #     x_pitch = abs(((p1[0] + p1[2]) / 2) - ((p2[0] + p2[2]) / 2))
        #     print(f"  [Info] 估算 X-Pitch: {x_pitch:.2f} pixels")
        # if adj_pairs['y_pair']:
        #     p1, p2 = adj_pairs['y_pair']
        #     y_pitch = abs(((p1[1] + p1[3]) / 2) - ((p2[1] + p2[3]) / 2))
        #     print(f"  [Info] 估算 Y-Pitch: {y_pitch:.2f} pixels")

        # === 新增：调用可视化 ===
        # 定义可视化图片保存路径 (例如保存到原图片目录下，文件名加后缀)
        # vis_output_path = os.path.splitext(img_path)[0] + "_visualized_pairs.png"
        # 如果想保存到特定目录：
        # vis_output_path = os.path.join("你的输出目录", os.path.basename(img_path).replace(".png", "_vis.png"))

        # visualize_selected_pairs(img_path, border_coords, classified_pins, adj_pairs, vis_output_path)

        # 可视化分边结果
        # visualization_output_path = os.path.join("pin_classification_visualization.png")
        # visualize_classified_pins(img_path, border_coords, classified_pins, visualization_output_path)

        return X,Y
    except Exception as e:
        print(f"PIN处理过程发生错误，{str(e)}")
        return "",""





if __name__ == "__main__":
    # 获取模型推理结果和路径信息
    img_path = r"D:\HuaweiMoveData\Users\LNQ\Desktop\训练数据\QFN\bottom\JPEGImages\6.png"

    X,Y = QFN_extract_pins(img_path)


