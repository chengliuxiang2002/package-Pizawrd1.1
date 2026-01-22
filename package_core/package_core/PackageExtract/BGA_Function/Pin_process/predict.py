import copy
import onnxruntime as rt
import numpy as np
import cv2
import os



# 前处理
def resize_image(image, size, letterbox_image):
    """
        对输入图像进行resize
    Args:
        size:目标尺寸
        letterbox_image: bool 是否进行letterbox变换
    Returns:指定尺寸的图像
    """
    ih, iw, _ = image.shape
    h, w = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # 生成画布
        image_back = np.ones((h, w, 3), dtype=np.uint8) * 128
        # 将image放在画布中心区域-letterbox
        image_back[(h - nh) // 2: (h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw, :] = image
    else:
        image_back = image
    return image_back


def img2input(img):
    img = np.transpose(img, (2, 0, 1))
    img = img / 255
    return np.expand_dims(img, axis=0).astype(np.float32)


def std_output(pred):
    """
    将（1，84，8400）处理成（8400， 85）  85= box:4  conf:1 cls:80
    """
    pred = np.squeeze(pred)
    pred = np.transpose(pred, (1, 0))
    pred_class = pred[..., 4:]
    pred_conf = np.max(pred_class, axis=-1)
    pred = np.insert(pred, 4, pred_conf, axis=-1)
    return pred


def xywh2xyxy(*box):
    """
    将xywh转换为左上角点和右下角点
    Args:
        box:
    Returns: x1y1x2y2
    """
    ret = [box[0] - box[2] // 2, box[1] - box[3] // 2, \
           box[0] + box[2] // 2, box[1] + box[3] // 2]
    return ret


def get_inter(box1, box2):
    """
    计算相交部分面积
    Args:
        box1: 第一个框
        box2: 第二个框
    Returns: 相交部分的面积
    """
    x1, y1, x2, y2 = xywh2xyxy(*box1)
    x3, y3, x4, y4 = xywh2xyxy(*box2)
    # 验证是否存在交集
    if x1 >= x4 or x2 <= x3:
        return 0
    if y1 >= y4 or y2 <= y3:
        return 0
    # 将x1,x2,x3,x4排序，计算交集宽度
    x_list = sorted([x1, x2, x3, x4])
    x_inter = x_list[2] - x_list[1]
    # 将y1,y2,y3,y4排序，计算交集高度
    y_list = sorted([y1, y2, y3, y4])
    y_inter = y_list[2] - y_list[1]
    # 计算交集的面积
    inter = x_inter * y_inter
    return inter


def get_iou(box1, box2):
    """
    计算交并比： (A n B)/(A + B - A n B)
    Args:
        box1: 第一个框
        box2: 第二个框
    Returns: 交并比的值
    """
    box1_area = box1[2] * box1[3]  # 计算第一个框的面积
    box2_area = box2[2] * box2[3]  # 计算第二个框的面积
    inter_area = get_inter(box1, box2)
    union = box1_area + box2_area - inter_area  # (A n B)/(A + B - A n B)
    iou = inter_area / union
    return iou


def nms(pred, conf_thres, iou_thres):
    """
    非极大值抑制nms
    Args:
        pred: 模型输出特征图
        conf_thres: 置信度阈值
        iou_thres: iou阈值
    Returns: 输出后的结果
    """
    box = pred[pred[..., 4] > conf_thres]  # 置信度筛选
    if len(box) == 0:
        return []  # 没有检测到目标时返回空列表

    cls_conf = box[..., 5:]
    cls = []
    for i in range(len(cls_conf)):
        cls.append(int(np.argmax(cls_conf[i])))
    total_cls = list(set(cls))  # 记录图像内共出现几种物体
    output_box = []
    # 每个预测类别分开考虑
    for i in range(len(total_cls)):
        clss = total_cls[i]
        cls_box = []
        temp = box[:, :6]
        for j in range(len(cls)):
            # 记录[x,y,w,h,conf(最大类别概率),class]值
            if cls[j] == clss:
                temp[j][5] = clss
                cls_box.append(temp[j][:6])
        #  cls_box 里面是[x,y,w,h,conf(最大类别概率),class]
        cls_box = np.array(cls_box)
        sort_cls_box = sorted(cls_box, key=lambda x: -x[4])  # 将cls_box按置信度从大到小排序
        # 得到置信度最大的预测框
        max_conf_box = sort_cls_box[0]
        output_box.append(max_conf_box)
        sort_cls_box = np.delete(sort_cls_box, 0, 0)
        # 对除max_conf_box外其他的框进行非极大值抑制
        while len(sort_cls_box) > 0:
            # 得到当前最大的框
            max_conf_box = output_box[-1]
            del_index = []
            for j in range(len(sort_cls_box)):
                current_box = sort_cls_box[j]
                iou = get_iou(max_conf_box, current_box)
                if iou > iou_thres:
                    # 筛选出与当前最大框Iou大于阈值的框的索引
                    del_index.append(j)
            # 删除这些索引
            sort_cls_box = np.delete(sort_cls_box, del_index, 0)
            if len(sort_cls_box) > 0:
                output_box.append(sort_cls_box[0])
                sort_cls_box = np.delete(sort_cls_box, 0, 0)
    return output_box


def cod_trf(result, pre, after):
    """
    因为预测框是在经过letterbox后的图像上做预测所以需要将预测框的坐标映射回原图像上
    Args:
        result:  [x,y,w,h,conf(最大类别概率),class]
        pre:    原尺寸图像
        after:  经过letterbox处理后的图像
    Returns: 坐标变换后的结果,
    """
    if not result:
        return np.array([])  # 空结果直接返回

    res = np.array(result)
    x, y, w, h, conf, cls = res.transpose((1, 0))
    x1, y1, x2, y2 = xywh2xyxy(x, y, w, h)  # 左上角点和右下角的点
    h_pre, w_pre, _ = pre.shape
    h_after, w_after, _ = after.shape
    scale = max(w_pre / w_after, h_pre / h_after)  # 缩放比例
    h_pre, w_pre = h_pre / scale, w_pre / scale  # 计算原图在等比例缩放后的尺寸
    x_move, y_move = abs(w_pre - w_after) // 2, abs(h_pre - h_after) // 2  # 计算平移的量
    ret_x1, ret_x2 = (x1 - x_move) * scale, (x2 - x_move) * scale
    ret_y1, ret_y2 = (y1 - y_move) * scale, (y2 - y_move) * scale
    ret = np.array([ret_x1, ret_y1, ret_x2, ret_y2, conf, cls]).transpose((1, 0))
    return ret


# 新增：检测结果格式标准化函数（核心修改）
def format_detection_results(res):
    """
    标准化检测结果格式：
    - 坐标（x1,y1,x2,y2）：取整，去掉小数点
    - 置信度（conf）：保留两位小数
    - 类别ID（cls_id）：取整，转为整数格式
    Args:
        res: 原始检测结果数组（shape: [N,6]）
    Returns: 格式化后的数组（保持原结构，优化数值显示）
    """
    if len(res) == 0:
        return res  # 空结果直接返回

    # 深拷贝避免修改原数据
    formatted_res = copy.deepcopy(res)

    for i in range(len(formatted_res)):
        x1, y1, x2, y2, conf, cls_id = formatted_res[i]

        # 1. 坐标取整（去掉小数点，四舍五入避免偏差）
        formatted_res[i][0] = int(round(x1))  # 左上角X
        formatted_res[i][1] = int(round(y1))  # 左上角Y
        formatted_res[i][2] = int(round(x2))  # 右下角X
        formatted_res[i][3] = int(round(y2))  # 右下角Y

        # 2. 置信度保留两位小数
        formatted_res[i][4] = round(conf, 2)

        # 3. 类别ID取整（去掉小数，转为纯整数）
        formatted_res[i][5] = int(cls_id)

    return formatted_res


# ===================== 新增函数：提取类别ID=1的坐标 =====================
def extract_pin_coords(res):
    """
    从格式化后的检测结果中，提取所有类别ID为1的目标坐标
    Args:
        res: 格式化后的检测结果（format_detection_results处理后的数组）
    Returns: 类别ID=1的坐标列表，格式：[[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
             坐标为整数，无多余小数
    """
    if len(res) == 0:
        return []  # 无检测结果时返回空列表

    class1_coords = []
    for det in res:
        cls_id = int(det[5])  # 获取类别ID（已格式化，直接转int）
        if cls_id == 1:  # 筛选类别ID=1的目标（BGA_PIN）
            # 提取前4列坐标（x1, y1, x2, y2），转为整数后添加到列表
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            class1_coords.append([x1, y1, x2, y2])

    return class1_coords


# 提取类别ID=0的最大坐标（Border，多框时取面积最大）
def extract_border_coords(res):
    """
    从格式化后的检测结果中，提取类别ID=0的目标坐标（BGA_Border）
    若存在多个类别0的目标，返回面积最大的那个（面积=（x2-x1）*(y2-y1)）
    Args:
        res: 格式化后的检测结果（format_detection_results处理后的数组）
    Returns: 最大面积的类别0坐标，格式：[x1, y1, x2, y2]（坐标为整数）
             无检测结果时返回空列表
    """
    if len(res) == 0:
        return []  # 无检测结果时返回空列表

    # 第一步：筛选所有类别ID=0的目标
    class0_dets = []
    for det in res:
        cls_id = int(det[5])
        if cls_id == 0:  # 筛选类别ID=0
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            # 确保边界框有效（x2>x1，y2>y1）
            if x2 > x1 and y2 > y1:
                class0_dets.append([x1, y1, x2, y2])

    # 第二步：处理筛选结果
    if len(class0_dets) == 0:
        return []  # 无类别0的目标
    elif len(class0_dets) == 1:
        return class0_dets[0]  # 只有1个，直接返回
    else:
        # 多个目标时，计算每个框的面积，取面积最大的
        max_area = 0
        max_coords = []
        for coords in class0_dets:
            x1, y1, x2, y2 = coords
            area = (x2 - x1) * (y2 - y1)  # 计算面积
            if area > max_area:
                max_area = area
                max_coords = coords
        return max_coords

def draw(res, image, cls):
    """
    将预测框绘制在image上（仅画框，4类对应4种颜色）
    Args:
        res: 预测框数据
        image: 原图
        cls: 类别列表（仅用于确认类别数量，实际不显示）
    Returns: 绘制后的图像
    """
    # 定义4种鲜明颜色（对应4个类别，顺序与class_config一致）
    # 颜色格式：(B, G, R)，OpenCV默认BGR通道
    class_colors = [
        (0, 0, 255),  # 类别0：BGA_Border → 红色
        (0, 255, 0),  # 类别1：BGA_PIN → 绿色
        (255, 0, 0),  # 类别2：BGA_serial_letter → 蓝色
        (0, 255, 255)  # 类别3：BGA_serial_number → 黄色
    ]

    for r in res:
        class_id = int(r[5])  # 获取当前框的类别ID（0-3）
        # 选择对应类别的颜色（超出4类时默认红色）
        color = class_colors[class_id] if class_id < len(class_colors) else (0, 0, 255)
        # 画框（线宽2，更清晰）
        image = cv2.rectangle(
            image,
            (int(r[0]), int(r[1])),  # 左上角坐标（已格式化，直接转int不影响）
            (int(r[2]), int(r[3])),  # 右下角坐标
            color,
            thickness=2  # 线宽调整为2，比原1更易看清
        )
    return image



# 支持的图片格式
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')


def process_single_image(img_path, output_dir, sess, std_h, std_w, class_config, conf_thres=0.5,
                         iou_thres=0.4,
                         show_image=False):
    """
    处理单张图片的函数（优化：直接接收完整img_path，去掉input_dir参数）
    :param img_path: 完整的图片路径（如"imgs/QFP/001.png"），支持中文路径
    :param output_dir: 输出目录
    :param sess: onnx模型会话
    :param std_h: 标准输入高度
    :param std_w: 标准输入宽度
    :param class_config: 类别配置列表
    :param conf_thres: 置信度阈值
    :param iou_thres: IOU阈值
    :param show_image: 是否显示结果图
    :return: 格式化后的检测结果res + 绘制后的图像
    """
    # 直接使用传入的完整img_path，无需拼接（支持中文路径）
    try:
        # 读取图片（支持中文路径，替换cv2.imread）
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"警告：无法读取图片 {img_path}（路径可能错误或文件损坏）- {str(e)}，跳过该文件")
        return []

    if img is None or img.size == 0:
        print(f"警告：图片 {img_path} 为空，跳过该文件")
        return []

    try:
        # 前处理
        img_after = resize_image(img, (std_w, std_h), True)
        # 将图像处理成输入的格式
        data = img2input(img_after)
        # 模型推理
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        pred = sess.run([label_name], {input_name: data})[0]
        pred = std_output(pred)
        # 置信度过滤+nms
        result = nms(pred, conf_thres, iou_thres)
        # 坐标变换
        res = cod_trf(result, img, img_after)
        # 核心：格式化检测结果（坐标取整、置信度两位小数、类别ID整数）
        res = format_detection_results(res)
        # 绘制预测框
        image = draw(res, img, class_config)

        # 显示图片（可选）
        if show_image:
            # 窗口标题用图片文件名（避免中文路径乱码）
            win_name = os.path.basename(img_path)
            cv2.imshow(win_name, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # 保存输出图像（提取文件名，避免保存时创建多余目录）
        if output_dir:
            output_filename = os.path.basename(img_path)  # 从完整路径中提取文件名（如"001.png"）
            output_path = os.path.join(output_dir, output_filename)
            # 保存时支持中文路径（替换cv2.imwrite）
            cv2.imencode(os.path.splitext(output_filename)[1], image)[1].tofile(output_path)
            print(f"\n成功处理：{img_path} -> 保存至：{output_path}")

        return res # 返回格式化后的res

    except Exception as e:
        print(f"\n错误：处理图片 {img_path} 时发生异常 - {str(e)}，跳过该文件")
        return []


if __name__ == '__main__':
    # 基础配置参数（不变）
    std_h, std_w = 640, 640  # 标准输入尺寸
    conf_thres = 0.2  # 置信度阈值
    iou_thres = 0.2  # IOU阈值
    class_config = ['BGA_Border', 'BGA_PIN', 'BGA_serial_letter', 'BGA_serial_number']  # 类别配置
    input_dir = "imgs/QFN"  # 输入图片目录
    output_dir = r"output"  # 输出目录
    model_path = r'D:\workspace\PackageWizard1.1\model\yolo_model\pin_detect\BGA_pin_detect.onnx'  # 模型路径
    img_path = r"D:\workspace\PackageWizard1.1\Result\Package_extract\data\bottom.jpg"
    # 创建输出目录（不变）
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录：{output_dir}（已确保存在）")

    # 加载ONNX模型（不变）
    try:
        sess = rt.InferenceSession(model_path)
        print(f"成功加载模型：{model_path}")
    except Exception as e:
        print(f"错误：无法加载模型 {model_path} - {str(e)}")
        exit(1)

    # 处理单张图片（不变）
    res = process_single_image(
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

    # ===================== 调用新增函数：提取类别ID=1的坐标 =====================
    pin_coords = extract_pin_coords(res)
    border_coords = extract_border_coords(res)
    print(pin_coords)
    print(border_coords)