import cv2
import os
import numpy as np
from ultralytics import YOLO

from package_core.PackageExtract.yolox_onnx_py.model_paths import result_path, model_path

# ================= 配置区域 =================
MODEL_PATH = model_path("yolo_model","pin_detect","BGA.onnx") # 模型路径
# 修改：指定单张图片的完整路径
SINGLE_IMAGE_PATH = result_path("Package_extract","data","bottom.jpg")  # 替换成你的单张图片路径
CROP_DIR = result_path("Package_extract","data_bottom_crop")  # 输出文件夹名

# 关键设置
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# 【ID配置】
BORDER_CLASS_ID = 0  # 边框类别 ID
PIN_CLASS_ID = 1  # Pin 类别 ID

# 裁剪边缘留白 (Padding)，单位像素
CROP_PADDING = 5

# ===========================================

def is_center_in_box(pin_box, border_box):
    """
    判断 pin_box 的中心点是否在 border_box 内部
    box 格式: [x1, y1, x2, y2]
    """
    px1, py1, px2, py2 = pin_box
    bx1, by1, bx2, by2 = border_box

    # 计算 Pin 中心点
    cx = (px1 + px2) / 2
    cy = (py1 + py2) / 2

    # 判断中心点是否在 Border 范围内
    return (bx1 < cx < bx2) and (by1 < cy < by2)

# ========== 关键修改1：函数名从 main 改为 get_pinmap ==========
def get_pinmap():
    print(f"🔄 加载模型: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    print(f"📋 类别配置: Border={BORDER_CLASS_ID}, Pin={PIN_CLASS_ID}")
    os.makedirs(CROP_DIR, exist_ok=True)

    # 检查单张图片是否存在
    if not os.path.exists(SINGLE_IMAGE_PATH):
        print(f"❌ 指定的图片不存在: {SINGLE_IMAGE_PATH}")
        return

    print(f"🚀 开始处理单张图片: {SINGLE_IMAGE_PATH}")

    # ========== 单张图片处理逻辑 ==========
    filename = os.path.basename(SINGLE_IMAGE_PATH)

    # 读取图片（兼容中文路径）
    frame = cv2.imdecode(np.fromfile(SINGLE_IMAGE_PATH, dtype=np.uint8), -1)
    if frame is None:
        print(f"❌ 图片读取失败: {SINGLE_IMAGE_PATH}")
        return

    img_h, img_w = frame.shape[:2]

    # 推理
    results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False, max_det=3000)
    result = results[0]

    # === 1. 收集原始数据 ===
    raw_pin_boxes = []
    border_boxes = []

    if result.boxes:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            coords = box.xyxy[0].cpu().numpy().astype(int)

            if cls_id == PIN_CLASS_ID:
                raw_pin_boxes.append(coords)
            elif cls_id == BORDER_CLASS_ID:
                border_boxes.append(coords)

    # === 2. 执行过滤逻辑 ===
    final_pin_boxes = []

    # 只有当检测到【唯一的】Border时，才启用过滤
    if len(border_boxes) == 1:
        target_border = border_boxes[0]

        for pin in raw_pin_boxes:
            if is_center_in_box(pin, target_border):
                final_pin_boxes.append(pin)

        # 打印过滤信息
        # removed_count = len(raw_pin_boxes) - len(final_pin_boxes)
        # if removed_count > 0:
        #     print(
        #         f"   🛡️ [{filename}] 检测到唯一 Border，已过滤干扰点: {len(raw_pin_boxes)} -> {len(final_pin_boxes)} (移除 {removed_count} 个)")
        # else:
        #     print(f"   🛡️ [{filename}] 检测到唯一 Border，所有点均有效。")

    else:
        # 如果没找到 Border，或者找到多个 Border，则保留所有 Pin（避免误杀）
        final_pin_boxes = raw_pin_boxes
        if len(raw_pin_boxes) > 0:
            print(
                f"   ⚠️ [{filename}] Border数量为 {len(border_boxes)}，跳过区域过滤，保留所有 {len(raw_pin_boxes)} 个 Pin。")

    # === 3. 裁剪逻辑 (使用 final_pin_boxes) ===
    if final_pin_boxes:
        np_boxes = np.array(final_pin_boxes)

        # 计算紧凑边界
        min_x = np.min(np_boxes[:, 0])
        min_y = np.min(np_boxes[:, 1])
        max_x = np.max(np_boxes[:, 2])
        max_y = np.max(np_boxes[:, 3])

        tight_x1 = max(0, min_x)
        tight_y1 = max(0, min_y)
        tight_x2 = min(img_w, max_x)
        tight_y2 = min(img_h, max_y)

        # 校验宽高
        if tight_x2 <= tight_x1 or tight_y2 <= tight_y1:
            print(f"⚠️ [{filename}] 有效区域宽高异常，跳过。")
            return

        # 抠图
        tight_crop_img = frame[tight_y1:tight_y2, tight_x1:tight_x2]

        # 加白边
        if CROP_PADDING > 0:
            white_color = [255, 255, 255]
            final_img = cv2.copyMakeBorder(
                tight_crop_img,
                top=CROP_PADDING,
                bottom=CROP_PADDING,
                left=CROP_PADDING,
                right=CROP_PADDING,
                borderType=cv2.BORDER_CONSTANT,
                value=white_color
            )
        else:
            final_img = tight_crop_img

        # 固定保存文件名为 pinmap.jpg
        save_name = "pinmap.jpg"
        save_path = os.path.join(CROP_DIR, save_name)
        cv2.imencode('.jpg', final_img)[1].tofile(save_path)

        h, w = final_img.shape[:2]
        # print(f"✅ 保存: {save_name} ({w}x{h})")

    else:
        print(f"⚠️ [{filename}] 无有效 Pin，跳过。")

    # print(f"\n🏁 单张图片处理完成，结果在: {CROP_DIR}")

# ========== 关键修改2：调用函数名同步改为 get_pinmap ==========
if __name__ == "__main__":
    get_pinmap()