import cv2
import glob
import fitz
from PIL import Image
import shutil
import os
import math
import numpy as np
import copy
def find_list(dic_list, listname):
    """
    在字典列表中寻找名字为listname的列表/矩阵，返回列表/矩阵
    :param dic_list:
    :param listname:
    :return:
    """
    target_dict = [d for d in dic_list if d.get('list_name') == listname]
    if target_dict:
        target_dict = target_dict[0]['list']
    else:
        print("未找到匹配的字典", listname)
        target_dict = []
    return target_dict


def recite_data(dict_list, listname, list):
    """
    更新字典列表中名字为listname的列表/矩阵，如果字典中没有则新加入
    :param dict_list:
    :param list_name:
    :param list:
    :return:
    """
    target_dict = [d for d in dict_list if d.get('list_name') == listname]
    if target_dict:
        target_dict[0]['list'] = list
    else:
        print("未找到匹配的字典,将新建一个字典加入字典列表中", listname)
        dict_list.append({'list_name': listname, 'list': list})
    return dict_list

def empty_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
    except FileNotFoundError:
        print(f"文件夹 {folder_path} 不存在！无须删除文件夹")


def pdf2img(pdfPath, pageNumber, imgfilePath, save_name):
    scale = 3  # 放大倍率
    images_np = []
    with fitz.open(pdfPath) as pdfDoc:
        # for pageNumber in pageNumbers:
        page = pdfDoc.load_page(pageNumber - 1)
        mat = fitz.Matrix(scale, scale).prerotate(0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(imgfilePath + '/' + save_name)  # 将图片写入指定的文件夹内
    return scale, images_np


def ResizeImage(filein, fileout, scale=1):
    """
    改变图片大小
    :param filein: 输入图片
    :param fileout: 输出图片
    :param width: 输出图片宽度
    :param height: 输出图片宽度
    :param type: 输出图片类型（png, gif, jpeg...）
    :return:
    """
    img = Image.open(filein)
    if img.size[0] < 1080 and img.size[1] < 1080:  # 限制图片大小避免过大

        width = int(img.size[0] * scale)
        height = int(img.size[1] * scale)
        type = img.format
        out = img.resize((width, height),
                         Image.LANCZOS)
        # 第二个参数：
        # Image.NEAREST ：低质量
        # Image.BILINEAR：双线性
        # Image.BICUBIC ：三次样条插值
        # Image.ANTIALIAS：高质量
        out.save(fileout, type)


def manual_get_boxes(folder_path, save_path, save_name):
    # 定义文件夹路径和保存路径
    global rect_list
    # 获取文件夹内所有图片文件
    file_list = glob.glob(os.path.join(folder_path, '*.jpg'))
    # 循环处理每张图片
    for i, file_path in enumerate(file_list):
        # 读取图片
        img = cv2.imread(file_path)
        # 获取图片尺寸
        height, width, _ = img.shape
        # 计算缩放比例
        # scale = min(1.0, 1024 / max(height, width))
        scale = 1
        # 缩放图片
        img_resized = cv2.resize(img, None, fx=scale, fy=scale)
        # 创建窗口并显示图片
        cv2.namedWindow('image', 0)
        # cv2.namedWindow('picture', 0)
        cv2.imshow('image', img_resized)
        # 初始化框选区域列表
        rect_list = []
        # 循环框选区域
        while True:
            # 等待用户框选区域
            rect = cv2.selectROI('image', img_resized, False)
            # 计算缩放后的框选区域
            rect_resized = [int(x / scale) for x in rect]
            # 如果没有框选区域，则退出循环
            if rect == (0, 0, 0, 0):
                break
            # 截取选中区域并保存
            crop_img = img[rect_resized[1]:rect_resized[1] + rect_resized[3],
                       rect_resized[0]:rect_resized[0] + rect_resized[2]]
            rect_list.append(rect)
            # print(rect_list)
            cv2.imwrite(os.path.join(save_path, save_name), crop_img)

    selectRec = (rect_list[0][0], rect_list[0][1], rect_list[0][2] + rect_list[0][0], rect_list[0][3] + rect_list[0][1])
    # crop_img_save()
    cv2.destroyAllWindows()  # 关闭弹框

    return selectRec

def set_Image_size(filein, fileout):
    """
    改变图片大小
    :param filein: 输入图片
    :param fileout: 输出图片
    :param width: 输出图片宽度
    :param height: 输出图片宽度
    :param type: 输出图片类型（png, gif, jpeg...）
    :return:
    """
    max_length = 1000
    img = Image.open(filein)
    if img.size[0] > img.size[1]:  # 限制图片大小避免过大
        width = max_length
        height = int(img.size[1] * max_length / img.size[0])
    else:
        height = max_length
        width = int(img.size[0] * max_length / img.size[1])
    type = img.format
    out = img.resize((width, height), Image.LANCZOS)
    out = hist(out, show_img_key=0)
    # 第二个参数：
    # Image.NEAREST ：低质量
    # Image.BILINEAR：双线性
    # Image.BICUBIC ：三次样条插值
    # Image.ANTIALIAS：高质量
    out = Image.fromarray(np.uint8(out))
    out.save(fileout, type)


def hist(img, show_img_key):
    # 求出img 的最大最小值
    Maximg = np.max(img)
    Minimg = np.min(img)
    # 输出最小灰度级和最大灰度级
    Omin, Omax = 0, 255
    # 求 a, b
    a = float(Omax - Omin) / (Maximg - Minimg)
    b = Omin - a * Minimg
    # 线性变换
    O = a * img + b
    O = O.astype(np.uint8)
    if show_img_key == 1:
        cv2.imshow('enhance-0', O)
        # cv2.imwrite('hist.png', O, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return O


def hist_auto(img):
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值均衡化
    dst = clahe.apply(img)
    # 使用全局直方图均衡化
    equa = cv2.equalizeHist(img)
    # 分别显示原图，CLAHE，HE
    # cv.imshow("img", img)
    # cv2.imshow("dst", dst)
    cv2.imwrite('hist_auto.png', dst, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


def calcGrayHist(I):
    # 计算灰度直方图
    h, w = I.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(h):
        for j in range(w):
            grayHist[I[i][j]] += 1
    return grayHist


def equalHist(img):
    # import math
    # 灰度图像矩阵的高、宽
    h, w = img.shape
    # 第一步：计算灰度直方图
    grayHist = calcGrayHist(img)
    # 第二步：计算累加灰度直方图
    zeroCumuMoment = np.zeros([256], np.uint32)
    for p in range(256):
        if p == 0:
            zeroCumuMoment[p] = grayHist[0]
        else:
            zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
    # 第三步：根据累加灰度直方图得到输入灰度级和输出灰度级之间的映射关系
    outPut_q = np.zeros([256], np.uint8)
    cofficient = 256.0 / (h * w)
    for p in range(256):
        q = cofficient * float(zeroCumuMoment[p]) - 1
        if q >= 0:
            outPut_q[p] = math.floor(q)
        else:
            outPut_q[p] = 0
    # 第四步：得到直方图均衡化后的图像
    equalHistImage = np.zeros(img.shape, np.uint8)
    for i in range(h):
        for j in range(w):
            equalHistImage[i][j] = outPut_q[img[i][j]]
    return equalHistImage


def linear(img):
    # img = cv2.imread(source, 0)
    # 使用自己写的函数实现
    equa = equalHist(img)
    cv2.imshow("equa", equa)
    cv2.imwrite('temp.png', equa, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    cv2.waitKey()

def get_BGA_parameter_data(QFP_parameter_list, nx, ny):
    """

        :param BGA_parameter_list:
        :param nx: PIN行数
        :param ny: PIN列数
        :return parameter_list:
        """
    parameter_list = []
    for i in range(19):
        parameter_list.append(['', '', '', ''])
    # PIN行/列间距
    if len(QFP_parameter_list[6]['maybe_data']) > 0:
        parameter_list[0][1] = round(QFP_parameter_list[6]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[0][2] = round(QFP_parameter_list[6]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[0][3] = round(QFP_parameter_list[6]['maybe_data'][0]['max_medium_min'][0], 4)
    if len(QFP_parameter_list[7]['maybe_data']) > 0:
        parameter_list[1][1] = round(QFP_parameter_list[7]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[1][2] = round(QFP_parameter_list[7]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[1][3] = round(QFP_parameter_list[7]['maybe_data'][0]['max_medium_min'][0], 4)
    # PIN行数
    # if nx > 0:
    # parameter_list[9][1] = nx
    # parameter_list[9][2] = nx
    # parameter_list[9][3] = nx
    # # PIN列数
    # if ny > 0:
    # parameter_list[10][1] = ny
    # parameter_list[10][2] = ny
    # parameter_list[10][3] = ny
    # 实体高
    if len(QFP_parameter_list[4]['maybe_data']) > 0:
        parameter_list[4][1] = round(QFP_parameter_list[4]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[4][2] = round(QFP_parameter_list[4]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[4][3] = round(QFP_parameter_list[4]['maybe_data'][0]['max_medium_min'][0], 4)
    # 支撑高
    if len(QFP_parameter_list[5]['maybe_data']) > 0:
        parameter_list[5][1] = round(QFP_parameter_list[5]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[5][2] = round(QFP_parameter_list[5]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[5][3] = round(QFP_parameter_list[5]['maybe_data'][0]['max_medium_min'][0], 4)
    # 实体长
    if len(QFP_parameter_list[2]['maybe_data']) > 0:
        parameter_list[6][1] = round(QFP_parameter_list[2]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[6][2] = round(QFP_parameter_list[2]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[6][3] = round(QFP_parameter_list[2]['maybe_data'][0]['max_medium_min'][0], 4)
    elif len(QFP_parameter_list[3]['maybe_data']) > 0:
        parameter_list[6][1] = round(QFP_parameter_list[3]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[6][2] = round(QFP_parameter_list[3]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[6][3] = round(QFP_parameter_list[3]['maybe_data'][0]['max_medium_min'][0], 4)
    # 实体宽
    if len(QFP_parameter_list[3]['maybe_data']) > 0:
        parameter_list[7][1] = round(QFP_parameter_list[3]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[7][2] = round(QFP_parameter_list[3]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[7][3] = round(QFP_parameter_list[3]['maybe_data'][0]['max_medium_min'][0], 4)
    elif len(QFP_parameter_list[2]['maybe_data']) > 0:
        parameter_list[7][1] = round(QFP_parameter_list[2]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[7][2] = round(QFP_parameter_list[2]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[7][3] = round(QFP_parameter_list[2]['maybe_data'][0]['max_medium_min'][0], 4)

    return parameter_list

def get_QFP_parameter_data(QFP_parameter_list, nx, ny):
    """

    :param QFP_parameter_list:
    :param nx: PIN行数
    :param ny: PIN列数
    :return parameter_list:
    """
    parameter_list = []
    for i in range(19):
        parameter_list.append(['', '', '', ''])
    # 实体长
    if len(QFP_parameter_list[2]['maybe_data']) > 0:
        parameter_list[0][1] = round(QFP_parameter_list[2]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[0][2] = round(QFP_parameter_list[2]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[0][3] = round(QFP_parameter_list[2]['maybe_data'][0]['max_medium_min'][0], 4)
    elif len(QFP_parameter_list[3]['maybe_data']) > 0:
        parameter_list[0][1] = round(QFP_parameter_list[3]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[0][2] = round(QFP_parameter_list[3]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[0][3] = round(QFP_parameter_list[3]['maybe_data'][0]['max_medium_min'][0], 4)
    # 实体宽
    if len(QFP_parameter_list[3]['maybe_data']) > 0:
        parameter_list[1][1] = round(QFP_parameter_list[3]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[1][2] = round(QFP_parameter_list[3]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[1][3] = round(QFP_parameter_list[3]['maybe_data'][0]['max_medium_min'][0], 4)
    elif len(QFP_parameter_list[2]['maybe_data']) > 0:
        parameter_list[1][1] = round(QFP_parameter_list[2]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[1][2] = round(QFP_parameter_list[2]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[1][3] = round(QFP_parameter_list[2]['maybe_data'][0]['max_medium_min'][0], 4)
    # 实体高
    if len(QFP_parameter_list[4]['maybe_data']) > 0:
        parameter_list[2][1] = round(QFP_parameter_list[4]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[2][2] = round(QFP_parameter_list[4]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[2][3] = round(QFP_parameter_list[4]['maybe_data'][0]['max_medium_min'][0], 4)
    # 支撑高
    if len(QFP_parameter_list[5]['maybe_data']) > 0:
        parameter_list[3][1] = round(QFP_parameter_list[5]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[3][2] = round(QFP_parameter_list[5]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[3][3] = round(QFP_parameter_list[5]['maybe_data'][0]['max_medium_min'][0], 4)
    # 端子高
    # 外围长
    if len(QFP_parameter_list[0]['maybe_data']) > 0:
        parameter_list[5][1] = round(QFP_parameter_list[0]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[5][2] = round(QFP_parameter_list[0]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[5][3] = round(QFP_parameter_list[0]['maybe_data'][0]['max_medium_min'][0], 4)
    elif len(QFP_parameter_list[1]['maybe_data']) > 0:
        parameter_list[5][1] = round(QFP_parameter_list[1]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[5][2] = round(QFP_parameter_list[1]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[5][3] = round(QFP_parameter_list[1]['maybe_data'][0]['max_medium_min'][0], 4)
    # 外围宽
    if len(QFP_parameter_list[1]['maybe_data']) > 0:
        parameter_list[6][1] = round(QFP_parameter_list[1]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[6][2] = round(QFP_parameter_list[1]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[6][3] = round(QFP_parameter_list[1]['maybe_data'][0]['max_medium_min'][0], 4)
    elif len(QFP_parameter_list[0]['maybe_data']) > 0:
        parameter_list[6][1] = round(QFP_parameter_list[0]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[6][2] = round(QFP_parameter_list[0]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[6][3] = round(QFP_parameter_list[0]['maybe_data'][0]['max_medium_min'][0], 4)
    # PIN长
    # PIN宽
    if len(QFP_parameter_list[7]['maybe_data']) > 0:
        parameter_list[8][1] = round(QFP_parameter_list[7]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[8][2] = round(QFP_parameter_list[7]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[8][3] = round(QFP_parameter_list[7]['maybe_data'][0]['max_medium_min'][0], 4)
    # PIN行数
    if nx > 0:
        parameter_list[9][1] = nx
        parameter_list[9][2] = nx
        parameter_list[9][3] = nx
    # PIN列数
    if ny > 0:
        parameter_list[10][1] = ny
        parameter_list[10][2] = ny
        parameter_list[10][3] = ny
    # PIN行/列间距
    if len(QFP_parameter_list[6]['maybe_data']) > 0:
        parameter_list[11][1] = round(QFP_parameter_list[6]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[11][2] = round(QFP_parameter_list[6]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[11][3] = round(QFP_parameter_list[6]['maybe_data'][0]['max_medium_min'][0], 4)
    elif len(QFP_parameter_list[7]['maybe_data']) > 0:
        parameter_list[11][1] = round(QFP_parameter_list[7]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[11][2] = round(QFP_parameter_list[7]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[11][3] = round(QFP_parameter_list[7]['maybe_data'][0]['max_medium_min'][0], 4)
    # 散热盘长
    if len(QFP_parameter_list[8]['maybe_data']) > 0:
        parameter_list[12][1] = round(QFP_parameter_list[8]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[12][2] = round(QFP_parameter_list[8]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[12][3] = round(QFP_parameter_list[8]['maybe_data'][0]['max_medium_min'][0], 4)
    elif len(QFP_parameter_list[9]['maybe_data']) > 0:
        parameter_list[12][1] = round(QFP_parameter_list[9]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[12][2] = round(QFP_parameter_list[9]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[12][3] = round(QFP_parameter_list[9]['maybe_data'][0]['max_medium_min'][0], 4)
    # 散热盘宽
    if len(QFP_parameter_list[9]['maybe_data']) > 0:
        parameter_list[13][1] = round(QFP_parameter_list[9]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[13][2] = round(QFP_parameter_list[9]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[13][3] = round(QFP_parameter_list[9]['maybe_data'][0]['max_medium_min'][0], 4)
    elif len(QFP_parameter_list[8]['maybe_data']) > 0:
        parameter_list[13][1] = round(QFP_parameter_list[8]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[13][2] = round(QFP_parameter_list[8]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[13][3] = round(QFP_parameter_list[8]['maybe_data'][0]['max_medium_min'][0], 4)
    # 削角长度
    # 端子厚度（terminal thickness=C）
    if len(QFP_parameter_list[12]['maybe_data']) > 0:
        parameter_list[15][1] = round(QFP_parameter_list[12]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[15][2] = round(QFP_parameter_list[12]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[15][3] = round(QFP_parameter_list[12]['maybe_data'][0]['max_medium_min'][0], 4)
    # 接触角度
    if len(QFP_parameter_list[13]['maybe_data']) > 0:
        parameter_list[16][1] = round(QFP_parameter_list[13]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[16][2] = round(QFP_parameter_list[13]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[16][3] = round(QFP_parameter_list[13]['maybe_data'][0]['max_medium_min'][0], 4)
    # 端腿角度
    if len(QFP_parameter_list[14]['maybe_data']) > 0:
        parameter_list[17][1] = round(QFP_parameter_list[14]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[17][2] = round(QFP_parameter_list[14]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[17][3] = round(QFP_parameter_list[14]['maybe_data'][0]['max_medium_min'][0], 4)
    # 主体顶部绘制角度
    if len(QFP_parameter_list[15]['maybe_data']) > 0:
        parameter_list[18][1] = round(QFP_parameter_list[15]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[18][2] = round(QFP_parameter_list[15]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[18][3] = round(QFP_parameter_list[15]['maybe_data'][0]['max_medium_min'][0], 4)
    # if len(QFP_parameter_list[12]['maybe_data']) > 0:
    #     parameter_list[12][1] = round(QFP_parameter_list[12]['maybe_data'][0]['max_medium_min'][2], 4)
    #     parameter_list[12][2] = round(QFP_parameter_list[12]['maybe_data'][0]['max_medium_min'][1], 4)
    #     parameter_list[12][3] = round(QFP_parameter_list[12]['maybe_data'][0]['max_medium_min'][0], 4)
    # # 接触角度（contact angle=θ）
    # if len(QFP_parameter_list[13]['maybe_data']) > 0:
    #     parameter_list[13][1] = round(QFP_parameter_list[13]['maybe_data'][0]['max_medium_min'][2], 4)
    #     parameter_list[13][2] = round(QFP_parameter_list[13]['maybe_data'][0]['max_medium_min'][1], 4)
    #     parameter_list[13][3] = round(QFP_parameter_list[13]['maybe_data'][0]['max_medium_min'][0], 4)
    # # 端腿角度（terminal leg angle=θ1）
    # if len(QFP_parameter_list[14]['maybe_data']) > 0:
    #     parameter_list[14][1] = round(QFP_parameter_list[14]['maybe_data'][0]['max_medium_min'][2], 4)
    #     parameter_list[14][2] = round(QFP_parameter_list[14]['maybe_data'][0]['max_medium_min'][1], 4)
    #     parameter_list[14][3] = round(QFP_parameter_list[14]['maybe_data'][0]['max_medium_min'][0], 4)
    # # 主体顶部绘制角度（body top draft=θ2）
    # if len(QFP_parameter_list[15]['maybe_data']) > 0:
    #     parameter_list[15][1] = round(QFP_parameter_list[15]['maybe_data'][0]['max_medium_min'][2], 4)
    #     parameter_list[15][2] = round(QFP_parameter_list[15]['maybe_data'][0]['max_medium_min'][1], 4)
    #     parameter_list[15][3] = round(QFP_parameter_list[15]['maybe_data'][0]['max_medium_min'][0], 4)
    # # 主体底部绘制角度（body top draft=θ3）
    # if len(QFP_parameter_list[16]['maybe_data']) > 0:
    #     parameter_list[16][1] = round(QFP_parameter_list[16]['maybe_data'][0]['max_medium_min'][2], 4)
    #     parameter_list[16][2] = round(QFP_parameter_list[16]['maybe_data'][0]['max_medium_min'][1], 4)
    #     parameter_list[16][3] = round(QFP_parameter_list[16]['maybe_data'][0]['max_medium_min'][0], 4)
    return parameter_list


def alter_QFP_parameter_data(QFP_parameter_list):
    for i in range(len(QFP_parameter_list)):
        for j in range(len(QFP_parameter_list[i])):
            if j > 0:
                if QFP_parameter_list[i][j] == '':
                    QFP_parameter_list[i][j] = 0
    print(QFP_parameter_list)
    new_parameter_list = copy.deepcopy(QFP_parameter_list)

    # 检测长和宽与外围长和宽的大小是否符合要求，不符合则互换
    if QFP_parameter_list[0][2] > QFP_parameter_list[5][2]:
        new_parameter_list[0][1] = QFP_parameter_list[5][1]
        new_parameter_list[0][2] = QFP_parameter_list[5][2]
        new_parameter_list[0][3] = QFP_parameter_list[5][3]
        new_parameter_list[5][1] = QFP_parameter_list[1][1]
        new_parameter_list[5][2] = QFP_parameter_list[1][2]
        new_parameter_list[5][3] = QFP_parameter_list[1][3]
    if QFP_parameter_list[1][2] > QFP_parameter_list[6][2]:
        new_parameter_list[1][1] = QFP_parameter_list[6][1]
        new_parameter_list[1][2] = QFP_parameter_list[6][2]
        new_parameter_list[1][3] = QFP_parameter_list[6][3]
        new_parameter_list[6][1] = QFP_parameter_list[1][1]
        new_parameter_list[6][2] = QFP_parameter_list[1][2]
        new_parameter_list[6][3] = QFP_parameter_list[1][3]
    print(QFP_parameter_list)
    if new_parameter_list[0][2] == new_parameter_list[5][2] or new_parameter_list[1][2] == new_parameter_list[6][2]:
        # 取QFP_parameter_list的[0][1][5][6]的后三位中最大的一组为new_parameter_list的[5][6]的后三位值
        new_parameter_list[5][1:] = max(QFP_parameter_list[0][1:], QFP_parameter_list[1][1:], QFP_parameter_list[5][1:], QFP_parameter_list[6][1:])
        new_parameter_list[6][1:] = max(QFP_parameter_list[0][1:], QFP_parameter_list[1][1:], QFP_parameter_list[5][1:], QFP_parameter_list[6][1:])
        new_parameter_list[0][1:] = min(QFP_parameter_list[0][1:], QFP_parameter_list[1][1:], QFP_parameter_list[5][1:], QFP_parameter_list[6][1:])
        new_parameter_list[1][1:] = min(QFP_parameter_list[0][1:], QFP_parameter_list[1][1:], QFP_parameter_list[5][1:], QFP_parameter_list[6][1:])
    print(QFP_parameter_list)
    print(new_parameter_list)
    # 添加PIN长
    # for j in range(1, len(new_parameter_list[6])):
    #     new_parameter_list[7][j] = max(
    #         float(new_parameter_list[6][j]) - float(new_parameter_list[1][j]),
    #         float(new_parameter_list[5][j]) - float(new_parameter_list[0][j])
    #     )
    #     # 精确到小数点后几位
    #     try:
    #         new_parameter_list[7][j][1] = round(new_parameter_list[7][j][1], 4)
    #         new_parameter_list[7][j][2] = round(new_parameter_list[7][j][2], 4)
    #         new_parameter_list[7][j][3] = round(new_parameter_list[7][j][3], 4)
    #     except:
    #         print("精确失败")
    for i in range(len(new_parameter_list)):
        for j in range(len(new_parameter_list[i])):
            if j > 0:
                if new_parameter_list[i][j] == 0:
                    new_parameter_list[i][j] = ''
    return new_parameter_list



def get_SOP_parameter_data(SOP_parameter_list, nx, ny):
    """
    实体长（body x=D）
    实体宽（body y=E）
    实体高(package height=A)
    支撑高(standoff=A1)
    PIN长(contact length=L)
    PIN宽(contact width=b)
    行PIN数(pins D)
    列PIN数(pins E)
    行/列PIN间距(pitch=e)
    PIN端距
    散热盘长(thermal_x=D2)
    散热盘宽(thermal_y=E2)
    端子厚度（terminal thickness=C）
    接触角度（contact angle=θ）
    端腿角度（terminal leg angle=θ1）
    主体顶部绘制角度（body top draft=θ2）
    主体底部绘制角度（body top draft=θ3）
    :param SOP_parameter_list:
    :param nx: PIN行数
    :param ny: PIN列数
    :return parameter_list:
    """
    parameter_list = []
    for i in range(17):
        parameter_list.append(['', '', '', ''])
    # 实体长
    if len(SOP_parameter_list[2]['maybe_data']) > 0:
        parameter_list[0][1] = round(SOP_parameter_list[2]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[0][2] = round(SOP_parameter_list[2]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[0][3] = round(SOP_parameter_list[2]['maybe_data'][0]['max_medium_min'][0], 4)
    elif len(SOP_parameter_list[3]['maybe_data']) > 0:
        parameter_list[0][1] = round(SOP_parameter_list[3]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[0][2] = round(SOP_parameter_list[3]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[0][3] = round(SOP_parameter_list[3]['maybe_data'][0]['max_medium_min'][0], 4)
    # 实体宽
    if len(SOP_parameter_list[3]['maybe_data']) > 0:
        parameter_list[1][1] = round(SOP_parameter_list[3]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[1][2] = round(SOP_parameter_list[3]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[1][3] = round(SOP_parameter_list[3]['maybe_data'][0]['max_medium_min'][0], 4)
    elif len(SOP_parameter_list[2]['maybe_data']) > 0:
        parameter_list[1][1] = round(SOP_parameter_list[2]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[1][2] = round(SOP_parameter_list[2]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[1][3] = round(SOP_parameter_list[2]['maybe_data'][0]['max_medium_min'][0], 4)
    # 实体高
    if len(SOP_parameter_list[4]['maybe_data']) > 0:
        parameter_list[2][1] = round(SOP_parameter_list[4]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[2][2] = round(SOP_parameter_list[4]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[2][3] = round(SOP_parameter_list[4]['maybe_data'][0]['max_medium_min'][0], 4)
    # 支撑高
    if len(SOP_parameter_list[5]['maybe_data']) > 0:
        parameter_list[3][1] = round(SOP_parameter_list[5]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[3][2] = round(SOP_parameter_list[5]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[3][3] = round(SOP_parameter_list[5]['maybe_data'][0]['max_medium_min'][0], 4)
    # PIN长
    # PIN宽
    if len(SOP_parameter_list[7]['maybe_data']) > 0:
        parameter_list[5][1] = round(SOP_parameter_list[7]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[5][2] = round(SOP_parameter_list[7]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[5][3] = round(SOP_parameter_list[7]['maybe_data'][0]['max_medium_min'][0], 4)
    # PIN行数
    if nx > 0:
        parameter_list[6][1] = nx
        parameter_list[6][2] = nx
        parameter_list[6][3] = nx
    # PIN列数
    if ny > 0:
        parameter_list[7][1] = ny
        parameter_list[7][2] = ny
        parameter_list[7][3] = ny
    # 行/列PIN间距(pitch=e)
    if len(SOP_parameter_list[6]['maybe_data']) > 0:
        parameter_list[8][1] = round(SOP_parameter_list[6]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[8][2] = round(SOP_parameter_list[6]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[8][3] = round(SOP_parameter_list[6]['maybe_data'][0]['max_medium_min'][0], 4)
    elif len(SOP_parameter_list[7]['maybe_data']) > 0:
        parameter_list[8][1] = round(SOP_parameter_list[7]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[8][2] = round(SOP_parameter_list[7]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[8][3] = round(SOP_parameter_list[7]['maybe_data'][0]['max_medium_min'][0], 4)
    # PIN端距
    # 散热盘长(thermal_x=D2)
    if len(SOP_parameter_list[8]['maybe_data']) > 0:
        parameter_list[10][1] = round(SOP_parameter_list[8]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[10][2] = round(SOP_parameter_list[8]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[10][3] = round(SOP_parameter_list[8]['maybe_data'][0]['max_medium_min'][0], 4)
    elif len(SOP_parameter_list[9]['maybe_data']) > 0:
        parameter_list[10][1] = round(SOP_parameter_list[9]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[10][2] = round(SOP_parameter_list[9]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[10][3] = round(SOP_parameter_list[9]['maybe_data'][0]['max_medium_min'][0], 4)
    # 散热盘宽(thermal_y=E2)
    if len(SOP_parameter_list[9]['maybe_data']) > 0:
        parameter_list[11][1] = round(SOP_parameter_list[9]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[11][2] = round(SOP_parameter_list[9]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[11][3] = round(SOP_parameter_list[9]['maybe_data'][0]['max_medium_min'][0], 4)
    elif len(SOP_parameter_list[8]['maybe_data']) > 0:
        parameter_list[11][1] = round(SOP_parameter_list[8]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[11][2] = round(SOP_parameter_list[8]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[11][3] = round(SOP_parameter_list[8]['maybe_data'][0]['max_medium_min'][0], 4)

    # 端子厚度（terminal thickness=C）
    if len(SOP_parameter_list[12]['maybe_data']) > 0:
        parameter_list[12][1] = round(SOP_parameter_list[12]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[12][2] = round(SOP_parameter_list[12]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[12][3] = round(SOP_parameter_list[12]['maybe_data'][0]['max_medium_min'][0], 4)
    # 接触角度（contact angle=θ）
    if len(SOP_parameter_list[13]['maybe_data']) > 0:
        parameter_list[13][1] = round(SOP_parameter_list[13]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[13][2] = round(SOP_parameter_list[13]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[13][3] = round(SOP_parameter_list[13]['maybe_data'][0]['max_medium_min'][0], 4)
    # 端腿角度（terminal leg angle=θ1）
    if len(SOP_parameter_list[14]['maybe_data']) > 0:
        parameter_list[14][1] = round(SOP_parameter_list[14]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[14][2] = round(SOP_parameter_list[14]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[14][3] = round(SOP_parameter_list[14]['maybe_data'][0]['max_medium_min'][0], 4)
    # 主体顶部绘制角度（body top draft=θ2）
    if len(SOP_parameter_list[15]['maybe_data']) > 0:
        parameter_list[15][1] = round(SOP_parameter_list[15]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[15][2] = round(SOP_parameter_list[15]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[15][3] = round(SOP_parameter_list[15]['maybe_data'][0]['max_medium_min'][0], 4)
    # 主体底部绘制角度（body top draft=θ3）
    if len(SOP_parameter_list[16]['maybe_data']) > 0:
        parameter_list[16][1] = round(SOP_parameter_list[16]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[16][2] = round(SOP_parameter_list[16]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[16][3] = round(SOP_parameter_list[16]['maybe_data'][0]['max_medium_min'][0], 4)


    return parameter_list

def get_SON_parameter_data(SON_parameter_list, nx, ny):
    """
    实体长（body x=D）
    实体宽（body y=E）
    实体高(package height=A)
    PIN长(contact length=L)
    PIN宽(contact width=b)
    行PIN数(pins D)
    列PIN数(pins E)
    行/列PIN间距(pitch=e)
    PIN端距
    散热盘长(thermal_x=D2)
    散热盘宽(thermal_y=E2)
    :param SON_parameter_list:
    :param nx: PIN行数
    :param ny: PIN列数
    :return parameter_list:
    """
    parameter_list = []
    for i in range(12):
        parameter_list.append(['', '', '', ''])
    # 实体长
    if len(SON_parameter_list[2]['maybe_data']) > 0:
        parameter_list[0][1] = round(SON_parameter_list[2]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[0][2] = round(SON_parameter_list[2]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[0][3] = round(SON_parameter_list[2]['maybe_data'][0]['max_medium_min'][0], 4)
    elif len(SON_parameter_list[3]['maybe_data']) > 0:
        parameter_list[0][1] = round(SON_parameter_list[3]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[0][2] = round(SON_parameter_list[3]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[0][3] = round(SON_parameter_list[3]['maybe_data'][0]['max_medium_min'][0], 4)
    # 实体宽
    if len(SON_parameter_list[3]['maybe_data']) > 0:
        parameter_list[1][1] = round(SON_parameter_list[3]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[1][2] = round(SON_parameter_list[3]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[1][3] = round(SON_parameter_list[3]['maybe_data'][0]['max_medium_min'][0], 4)
    elif len(SON_parameter_list[2]['maybe_data']) > 0:
        parameter_list[1][1] = round(SON_parameter_list[2]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[1][2] = round(SON_parameter_list[2]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[1][3] = round(SON_parameter_list[2]['maybe_data'][0]['max_medium_min'][0], 4)
    # 实体高
    if len(SON_parameter_list[4]['maybe_data']) > 0:
        parameter_list[2][1] = round(SON_parameter_list[4]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[2][2] = round(SON_parameter_list[4]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[2][3] = round(SON_parameter_list[4]['maybe_data'][0]['max_medium_min'][0], 4)
    # 支撑高
    if len(SON_parameter_list[5]['maybe_data']) > 0:
        parameter_list[3][1] = round(SON_parameter_list[5]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[3][2] = round(SON_parameter_list[5]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[3][3] = round(SON_parameter_list[5]['maybe_data'][0]['max_medium_min'][0], 4)
    # PIN长
    if len(SON_parameter_list[6]['maybe_data']) > 0:
        parameter_list[4][1] = round(SON_parameter_list[6]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[4][2] = round(SON_parameter_list[6]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[4][3] = round(SON_parameter_list[6]['maybe_data'][0]['max_medium_min'][0], 4)
    # PIN宽
    if len(SON_parameter_list[7]['maybe_data']) > 0:
        parameter_list[5][1] = round(SON_parameter_list[7]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[5][2] = round(SON_parameter_list[7]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[5][3] = round(SON_parameter_list[7]['maybe_data'][0]['max_medium_min'][0], 4)
    # PIN行数
    if nx > 0:
        parameter_list[6][1] = nx
        parameter_list[6][2] = nx
        parameter_list[6][3] = nx
    # PIN列数
    if ny > 0:
        parameter_list[7][1] = ny
        parameter_list[7][2] = ny
        parameter_list[7][3] = ny
    # 行/列PIN间距(pitch=e)
    if len(SON_parameter_list[6]['maybe_data']) > 0:
        parameter_list[8][1] = round(SON_parameter_list[6]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[8][2] = round(SON_parameter_list[6]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[8][3] = round(SON_parameter_list[6]['maybe_data'][0]['max_medium_min'][0], 4)
    elif len(SON_parameter_list[7]['maybe_data']) > 0:
        parameter_list[8][1] = round(SON_parameter_list[7]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[8][2] = round(SON_parameter_list[7]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[8][3] = round(SON_parameter_list[7]['maybe_data'][0]['max_medium_min'][0], 4)
    # PIN端距
    # 散热盘长(thermal_x=D2)
    if len(SON_parameter_list[8]['maybe_data']) > 0:
        parameter_list[10][1] = round(SON_parameter_list[8]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[10][2] = round(SON_parameter_list[8]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[10][3] = round(SON_parameter_list[8]['maybe_data'][0]['max_medium_min'][0], 4)
    elif len(SON_parameter_list[9]['maybe_data']) > 0:
        parameter_list[10][1] = round(SON_parameter_list[9]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[10][2] = round(SON_parameter_list[9]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[10][3] = round(SON_parameter_list[9]['maybe_data'][0]['max_medium_min'][0], 4)
    # 散热盘宽(thermal_y=E2)
    if len(SON_parameter_list[9]['maybe_data']) > 0:
        parameter_list[11][1] = round(SON_parameter_list[9]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[11][2] = round(SON_parameter_list[9]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[11][3] = round(SON_parameter_list[9]['maybe_data'][0]['max_medium_min'][0], 4)
    elif len(SON_parameter_list[8]['maybe_data']) > 0:
        parameter_list[11][1] = round(SON_parameter_list[8]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[11][2] = round(SON_parameter_list[8]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[11][3] = round(SON_parameter_list[8]['maybe_data'][0]['max_medium_min'][0], 4)

    # # 端子厚度（terminal thickness=C）
    # if len(SON_parameter_list[12]['maybe_data']) > 0:
    #     parameter_list[12][1] = round(SON_parameter_list[12]['maybe_data'][0]['max_medium_min'][2], 4)
    #     parameter_list[12][2] = round(SON_parameter_list[12]['maybe_data'][0]['max_medium_min'][1], 4)
    #     parameter_list[12][3] = round(SON_parameter_list[12]['maybe_data'][0]['max_medium_min'][0], 4)
    # # 接触角度（contact angle=θ）
    # if len(SON_parameter_list[13]['maybe_data']) > 0:
    #     parameter_list[13][1] = round(SON_parameter_list[13]['maybe_data'][0]['max_medium_min'][2], 4)
    #     parameter_list[13][2] = round(SON_parameter_list[13]['maybe_data'][0]['max_medium_min'][1], 4)
    #     parameter_list[13][3] = round(SON_parameter_list[13]['maybe_data'][0]['max_medium_min'][0], 4)
    # # 端腿角度（terminal leg angle=θ1）
    # if len(SON_parameter_list[14]['maybe_data']) > 0:
    #     parameter_list[14][1] = round(SON_parameter_list[14]['maybe_data'][0]['max_medium_min'][2], 4)
    #     parameter_list[14][2] = round(SON_parameter_list[14]['maybe_data'][0]['max_medium_min'][1], 4)
    #     parameter_list[14][3] = round(SON_parameter_list[14]['maybe_data'][0]['max_medium_min'][0], 4)
    # # 主体顶部绘制角度（body top draft=θ2）
    # if len(SON_parameter_list[15]['maybe_data']) > 0:
    #     parameter_list[15][1] = round(SON_parameter_list[15]['maybe_data'][0]['max_medium_min'][2], 4)
    #     parameter_list[15][2] = round(SON_parameter_list[15]['maybe_data'][0]['max_medium_min'][1], 4)
    #     parameter_list[15][3] = round(SON_parameter_list[15]['maybe_data'][0]['max_medium_min'][0], 4)
    # # 主体底部绘制角度（body top draft=θ3）
    # if len(SON_parameter_list[16]['maybe_data']) > 0:
    #     parameter_list[16][1] = round(SON_parameter_list[16]['maybe_data'][0]['max_medium_min'][2], 4)
    #     parameter_list[16][2] = round(SON_parameter_list[16]['maybe_data'][0]['max_medium_min'][1], 4)
    #     parameter_list[16][3] = round(SON_parameter_list[16]['maybe_data'][0]['max_medium_min'][0], 4)


    return parameter_list

def get_QFN_parameter_data(QFN_parameter_list, nx, ny):
    """
    实体长（body x=D）
    实体宽（body y=E）
    实体高(package height=A)
    支撑高(standoff=A1)
    端子高(terminal height=A3)
    PIN长(contact length=L)
    PIN宽(contact width=b)
    行PIN数(pins D)
    列PIN数(pins E)
    行/列PIN间距(pitch=e)
    散热盘长(thermal_x=D2)
    散热盘宽(thermal_y=E2)
    削角否(chamfer or not)
    削角长度（chamfer_length）
    端子圆角否(teminal round or not)
    :param QFP_parameter_list:
    :param nx: PIN行数
    :param ny: PIN列数
    :return parameter_list:
    """
    parameter_list = []
    for i in range(15):
        parameter_list.append(['', '', '', ''])
    # 实体长
    if len(QFN_parameter_list[2]['maybe_data']) > 0:
        parameter_list[0][1] = round(QFN_parameter_list[2]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[0][2] = round(QFN_parameter_list[2]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[0][3] = round(QFN_parameter_list[2]['maybe_data'][0]['max_medium_min'][0], 4)
    elif len(QFN_parameter_list[3]['maybe_data']) > 0:
        parameter_list[0][1] = round(QFN_parameter_list[3]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[0][2] = round(QFN_parameter_list[3]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[0][3] = round(QFN_parameter_list[3]['maybe_data'][0]['max_medium_min'][0], 4)
    # 实体宽
    if len(QFN_parameter_list[3]['maybe_data']) > 0:
        parameter_list[1][1] = round(QFN_parameter_list[3]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[1][2] = round(QFN_parameter_list[3]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[1][3] = round(QFN_parameter_list[3]['maybe_data'][0]['max_medium_min'][0], 4)
    elif len(QFN_parameter_list[2]['maybe_data']) > 0:
        parameter_list[1][1] = round(QFN_parameter_list[2]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[1][2] = round(QFN_parameter_list[2]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[1][3] = round(QFN_parameter_list[2]['maybe_data'][0]['max_medium_min'][0], 4)
    # 实体高
    if len(QFN_parameter_list[4]['maybe_data']) > 0:
        parameter_list[2][1] = round(QFN_parameter_list[4]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[2][2] = round(QFN_parameter_list[4]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[2][3] = round(QFN_parameter_list[4]['maybe_data'][0]['max_medium_min'][0], 4)
    # 支撑高
    if len(QFN_parameter_list[5]['maybe_data']) > 0:
        parameter_list[3][1] = round(QFN_parameter_list[5]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[3][2] = round(QFN_parameter_list[5]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[3][3] = round(QFN_parameter_list[5]['maybe_data'][0]['max_medium_min'][0], 4)
    # 端子高
    # PIN长
    # PIN宽
    if len(QFN_parameter_list[7]['maybe_data']) > 0:
        parameter_list[6][1] = round(QFN_parameter_list[7]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[6][2] = round(QFN_parameter_list[7]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[6][3] = round(QFN_parameter_list[7]['maybe_data'][0]['max_medium_min'][0], 4)
    # PIN行数
    if nx > 0:
        parameter_list[7][1] = nx
        parameter_list[7][2] = nx
        parameter_list[7][3] = nx
    # PIN列数
    if ny > 0:
        parameter_list[8][1] = ny
        parameter_list[8][2] = ny
        parameter_list[8][3] = ny
    # 行 / 列PIN间距(pitch=e)
    if len(QFN_parameter_list[6]['maybe_data']) > 0:
        parameter_list[9][1] = round(QFN_parameter_list[6]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[9][2] = round(QFN_parameter_list[6]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[9][3] = round(QFN_parameter_list[6]['maybe_data'][0]['max_medium_min'][0], 4)
    elif len(QFN_parameter_list[7]['maybe_data']) > 0:
        parameter_list[9][1] = round(QFN_parameter_list[7]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[9][2] = round(QFN_parameter_list[7]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[9][3] = round(QFN_parameter_list[7]['maybe_data'][0]['max_medium_min'][0], 4)
    # 散热盘长
    if len(QFN_parameter_list[8]['maybe_data']) > 0:
        parameter_list[10][1] = round(QFN_parameter_list[8]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[10][2] = round(QFN_parameter_list[8]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[10][3] = round(QFN_parameter_list[8]['maybe_data'][0]['max_medium_min'][0], 4)
    elif len(QFN_parameter_list[9]['maybe_data']) > 0:
        parameter_list[10][1] = round(QFN_parameter_list[9]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[10][2] = round(QFN_parameter_list[9]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[10][3] = round(QFN_parameter_list[9]['maybe_data'][0]['max_medium_min'][0], 4)
    # 散热盘宽
    if len(QFN_parameter_list[9]['maybe_data']) > 0:
        parameter_list[11][1] = round(QFN_parameter_list[9]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[11][2] = round(QFN_parameter_list[9]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[11][3] = round(QFN_parameter_list[9]['maybe_data'][0]['max_medium_min'][0], 4)
    elif len(QFN_parameter_list[8]['maybe_data']) > 0:
        parameter_list[11][1] = round(QFN_parameter_list[8]['maybe_data'][0]['max_medium_min'][2], 4)
        parameter_list[11][2] = round(QFN_parameter_list[8]['maybe_data'][0]['max_medium_min'][1], 4)
        parameter_list[11][3] = round(QFN_parameter_list[8]['maybe_data'][0]['max_medium_min'][0], 4)
    # 削角否(chamfer or not)
    # 削角长度（chamfer_length）
    # 端子圆角否

    return parameter_list


def extract_package(img_path):
    """
    提取封装图
    :param img_path:data文件夹路径，包含分割好的图片
    :return:
    """

    # 多线程同时进行：

    #1、使用Opencv提取器件外框线：

    #2、DBNet提取所有文本框坐标范围L1集

    #3、Yolo分类，并提取图像元素坐标范围L2集合，包括尺寸标注、尺寸线、PIN序号、other、特有信息

    #---------------结束多线程---------------
    # F5.1 在L1集中寻找与Other坐标范围重合的个体，标志Other框则在L1集中删除相关的文本框

    # F5.2 在L1集中寻找与PIN序号坐标范围重合的个体，标记为PIN序号外框，并在L1集中删除相关的文本框

    # F5.3 在L1集中寻找与角度标注坐标范围重合的个体，进行合并，标记为角度标注外框，则在L1集中删除相关的文本框

    # F5.4 在L1集中寻找与尺寸标注坐标范围重合的个体，进行合并，标记为尺寸信息外框，则在L1集中删除相关的文本框

    # F5.5 使用SVTR识别L3集合内容、PIN序号框内容、角度标注框、Other框内容；根据框的类型分别对识别结果做信息整理

    # F5.6 将尺寸信息坐标范围与标尺线坐标范围进行中心距离匹配

    # F5.7 使用OPENCV获取标尺线两端的尺寸界线

    # F5.8 将尺寸信息坐标范围与标尺线坐标范围进行中心距离匹配