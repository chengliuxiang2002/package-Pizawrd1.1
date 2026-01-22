import cv2
import numpy as np


def calculate_iou(boxA, boxB):
    # 确定矩形的 (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 计算交集面积
    interArea = max(xB - xA + 1, 0) * max(yB - yA + 1, 0)

    # 计算并集面积
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    unionArea = boxAArea + boxBArea - interArea

    # 避免除数为0的情况
    if unionArea == 0:
        return 0
        # 计算IoU
    iou = interArea / unionArea
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def calculate_intersection(rect1, rect2):
    # rect1 和 rect2 都是形如 (x1, y1, x2, y2) 的矩形
    x_left = max(rect1[0], rect2[0])
    y_top = max(rect1[1], rect2[1])
    x_right = min(rect1[2], rect2[2])
    y_bottom = min(rect1[3], rect2[3])

    # 检查交集矩形是否有效
    if x_right >= x_left and y_bottom >= y_top:
        # 计算交集矩形的面积
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        return intersection_area
    else:
        # 没有交集
        return 0


def merge_rectangles(rects):
    # rects 是一个形如 [(x1, y1, x2, y2), ...] 的矩形列表
    if not rects:
        return None  # 如果没有矩形，则返回None

    # 初始化边界为第一个矩形的边界
    x_left = rects[0][0]
    y_top = rects[0][1]
    x_right = rects[0][2]
    y_bottom = rects[0][3]

    # 遍历所有矩形，更新边界
    for rect in rects:
        x_left = min(x_left, rect[0])
        y_top = min(y_top, rect[1])
        x_right = max(x_right, rect[2])
        y_bottom = max(y_bottom, rect[3])

        # 返回包含所有矩形的最小矩形
    return [x_left, y_top, x_right, y_bottom]


def abcdefg(img, box):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    valid_rects = []
    pred_box = box
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        screen_rect = (x, y, x + w, y + h)
        intersection = calculate_intersection(pred_box, screen_rect)
        if intersection:
            screen_area = (screen_rect[2] - screen_rect[0]) * (screen_rect[3] - screen_rect[1])
            overlap_ratio = intersection / screen_area
            if overlap_ratio > 0.95:
                valid_rects.append(screen_rect)
    return valid_rects


def expand_and_crop_image(box, image, expansion_factor_x, expansion_factor_y):
    height, width = image.shape[0:2]

    # 框的坐标
    x1, y1, x2, y2 = box

    # 计算框的中心点和宽度/高度
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width_box = x2 - x1
    height_box = y2 - y1

    # 计算扩大后的框的尺寸
    expanded_width = int(width_box * expansion_factor_x)
    expanded_height = int(height_box * expansion_factor_y)

    # 计算扩大后的框的左上角和右下角坐标
    expanded_x1 = max(0, center_x - expanded_width / 2)
    expanded_y1 = max(0, center_y - expanded_height / 2)
    expanded_x2 = min(width, center_x + expanded_width / 2)
    expanded_y2 = min(height, center_y + expanded_height / 2)

    # 确保框的坐标是整数
    expanded_x1, expanded_y1, expanded_x2, expanded_y2 = map(int, (expanded_x1, expanded_y1, expanded_x2, expanded_y2))

    # 裁剪图片
    cropped_img = image[expanded_y1:expanded_y2, expanded_x1:expanded_x2]

    rectangle = expanded_x1, expanded_y1, expanded_x2, expanded_y2

    return rectangle, cropped_img


def straight_line(img, horizontalSize, verticalSize):
    src_img0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    src_img2 = cv2.bitwise_not(src_img0)

    thresh, AdaptiveThreshold = cv2.threshold(src_img2, 5, 255, 0)

    horizontal = AdaptiveThreshold.copy()
    vertical = AdaptiveThreshold.copy()

    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalSize))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    mask = horizontal + vertical
    return mask


def find_best_match(bbox, image):
    typ = [None, None, None, None, None]
    rectangle, img = expand_and_crop_image(bbox, image, 2, 2)
    (rx1, ry1, rx2, ry2) = rectangle
    (bx1, by1, bx2, by2) = bbox
    new_bx1 = bx1 - rx1
    new_by1 = by1 - ry1
    new_bx2 = bx2 - rx1
    new_by2 = by2 - ry1
    box = (new_bx1, new_by1, new_bx2, new_by2)

    src_img0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    src_img1 = cv2.GaussianBlur(src_img0, (3, 3), 0)
    src_img2 = cv2.bitwise_not(src_img1)
    thresh, AdaptiveThreshold = cv2.threshold(src_img2, 5, 255, 0)

    x1, y1, x2, y2 = box
    box_area = AdaptiveThreshold[y1:y2, x1:x2]
    white_area = np.count_nonzero(box_area)
    box_area = (y2 - y1) * (x2 - x1)
    if white_area / box_area > 0.75:
        typ[0] = "Background"
        typ[1] = "no frame"
        typ[2] = "no outer frame"
        typ[3] = "no Horizontal lines"
        typ[4] = "no vertical lines"
        mask = straight_line(img, int(image.shape[1] / 20), int(image.shape[0] / 30))
        kernel = np.ones((6, 6), np.uint8)
        binary = cv2.dilate(mask, kernel)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        screened_rectangles = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            screened_rectangles.append([x, y, x + w, y + h])

        best_match = None
        max_iou = 0
        for screen_rect in screened_rectangles:
            iou = calculate_iou(box, screen_rect)
            if iou > max_iou:
                max_iou = iou
                best_match = screen_rect
    else:
        typ[0] = "no Background"
        mask = straight_line(img, int(image.shape[1] / 20), int(image.shape[0] / 30))
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        screened_rectangles = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            cnt_area = cv2.contourArea(cnt)
            ratio = cnt_area / area
            screened_rectangles.append([x, y, x + w, y + h, ratio])

        match = None
        max_iou = 0
        rect_ratio = 0
        for screen_rect_with_ratio in screened_rectangles:
            screen_rect = screen_rect_with_ratio[:4]
            iou = calculate_iou(box, screen_rect)
            if iou > max_iou:
                max_iou = iou
                match = screen_rect
                rect_ratio = screen_rect_with_ratio[-1]

        # match_area = (match[2] - match[0]) * (match[3] - match[1])
        # box_area = (box[2] - box[0]) * (box[3] - box[1])

        if max_iou > 0.5:
            typ[1] = "frame"
            best_match = match
            if rect_ratio >= 0.9:
                typ[2] = "outer frame"
            else:
                typ[2] = "no outer frame"

            need1, expanded_rectangle1 = expand_and_crop_image(best_match, img, 1.05, 1.05)
            # 设定长度阈值，即行中白色像素的最小数量
            length_threshold = expanded_rectangle1.shape[1] / 2

            src_img1 = cv2.cvtColor(expanded_rectangle1, cv2.COLOR_BGR2GRAY)
            src_img2 = cv2.bitwise_not(src_img1)
            thresh, AdaptiveThreshold = cv2.threshold(src_img2, 5, 255, 0)
            horizontal = AdaptiveThreshold.copy()
            horizontalSize = int(horizontal.shape[1] / 10)
            horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
            horizontal = cv2.erode(horizontal, horizontalStructure)
            horizontal = cv2.dilate(horizontal, horizontalStructure)

            sss = []
            for y in range(horizontal.shape[0]):
                row = horizontal[y, :]
                non_zero_count = np.count_nonzero(row)  # 计算非零像素的数量
                # 如果当前行的长度超过阈值
                if non_zero_count > length_threshold:
                    sss.append(y)
            if sss:
                count = 1
            else:
                count = 0
            for i in range(1, len(sss)):
                if sss[i] - sss[i - 1] > 1:  # 如果当前元素与前一个元素的差值大于1
                    count += 1  # 增加组的数量
            if count <= 3:
                typ[3] = "no Horizontal lines"
            else:
                typ[3] = "Horizontal lines"

            typ[4] = "vertical lines"

        else:
            typ[1] = "no frame"
            typ[2] = "no outer frame"
            typ[3] = "Horizontal lines"
            typ[4] = "no vertical lines"
            need, expanded_rectangle = expand_and_crop_image(box, mask, 1.2, 1.2)

            # 设定长度阈值，即行中白色像素的最小数量
            length_threshold = expanded_rectangle.shape[1] / 2

            # 初始化变量来存储结果
            long_lines_y_max = 0
            long_lines_y_min = expanded_rectangle.shape[0]

            # 遍历每一行
            for y in range(expanded_rectangle.shape[0]):
                row = expanded_rectangle[y, :]
                if row[0] == 0 and row[-1] == 0:  # 6/19加入了一个判断防止合并到过长的线
                    non_zero_count = np.count_nonzero(row)  # 计算非零像素的数量

                    # 如果当前行的长度超过阈值
                    if non_zero_count > length_threshold:
                        # 更新纵坐标的最大值和最小值
                        long_lines_y_max = max(long_lines_y_max, y)
                        long_lines_y_min = min(long_lines_y_min, y)

            cv2.line(expanded_rectangle, (expanded_rectangle.shape[1] // 2, long_lines_y_max), (expanded_rectangle.shape[1] // 2, long_lines_y_min), (255), 3)
            contours2, hierarchy2 = cv2.findContours(expanded_rectangle, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            fff = []
            for cnt in contours2:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                fff.append([x, y, x+w, y+h, area])
            sorted_lst = sorted(fff, key=lambda x: x[4], reverse=True)
            match = sorted_lst[0][:4]

            best_match = [match[0] + need[0], match[1] + need[1], match[2] + need[0], match[3] + need[1]]

            match22 = abcdefg(AdaptiveThreshold, box)  # 6/19增加了下面这三行
            match22.append(best_match)
            best_match = merge_rectangles(match22)

    (cx1, cy1, cx2, cy2) = best_match
    new_cx1 = cx1 + rx1
    new_cy1 = cy1 + ry1
    new_cx2 = cx2 + rx1
    new_cy2 = cy2 + ry1
    best_match_box = [new_cx1, new_cy1, new_cx2, new_cy2]
    def Reconfig_Tabletype(tableType):
        Background = False if 'no' in tableType[0] else True
        Frame = False if 'no' in tableType[1] else True
        OuterFrame = False if 'no' in tableType[2] else True
        HorizontalLines = False if 'no' in tableType[3] else True
        VerticalLines = False if 'no' in tableType[4] else True
        return {'Background':Background,
                'Frame':Frame,
                'OuterFrame':OuterFrame,
                'HorizontalLines':HorizontalLines,
                'VerticalLines':VerticalLines}
    list_type = Reconfig_Tabletype(typ)
    return best_match_box, list_type
