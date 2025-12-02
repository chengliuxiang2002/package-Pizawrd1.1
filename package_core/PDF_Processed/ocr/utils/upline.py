import cv2
from package_core.PDF_Processed.ocr.utils import imgProject

def isExistUpline(img):
    # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray', gray)
    ret1, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # cv2.imshow('th',th)
    h_h =  imgProject.hProject(th)
    flag = 0
    Pixel_start = []
    Pixel_end = []
    index = 0
    for i in h_h:
        if i != 0 and flag == 0:
            flag = 1
            Pixel_start.append(index)
        elif i == 0 and flag == 1:
            flag = 0
            Pixel_end.append(index-1)
        index = index + 1
    if h_h[-1] != 0:
        Pixel_end.append(index-1)

    # print(Pixel_start)
    # print(Pixel_end)

    #根据水平直方图统计像素高度
    Pixel_len = []
    for i in range(len(Pixel_start)):
        Pixel_len.append(Pixel_end[i] - Pixel_start[i] + 1)
    # print(Pixel_len)

    #计算像素高度占整张图片高度的比例
    height = img.shape[0]
    # print(height)
    Pixel_len_propotion = []
    for i in Pixel_len:
        Pixel_len_propotion.append(i/height)


    if len(Pixel_len) == 2:
        return True
    else:
        return False

#查找上划线坐标
def uplineCoordinate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
    ret1, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_h = imgProject.hProject(th)
    flag = 0
    height_start = []
    height_end = []
    index = 0
    for i in h_h:
        if i != 0 and flag == 0:
            flag = 1
            height_start.append(index)
        elif i == 0 and flag == 1:
            flag = 0
            height_end.append(index-1)
        index = index + 1
    if h_h[-1] != 0:
        height_end.append(index-1)

    # print(height_start)
    # print(height_end)

    # 根据水平直方图统计像素高度
    Pixel_len = []
    for i in range(len(height_start)):
        Pixel_len.append(height_end[i] - height_start[i] + 1)
    # print(Pixel_len)
    # print(height_start)
    cropheight = height_start[1] - 1
    # print(cropheight)
    cropped = th[:int(cropheight),:]
    # cv2.imshow('cropimg',cropped)
    # cv2.waitKey()

    w_w = imgProject.vProject(cropped)
    width_start = []
    width_end = []
    index2 = 0
    flag2 = 0
    for i in w_w:
        if i != 0 and flag2 == 0:
            width_start.append(index2)
            flag2 = 1
        elif i == 0 and flag2 == 1:
            width_end.append(index2-1)
            flag2 = 0
        index2 = index2 + 1
    if w_w[-1] != 0:
        height_end.append(index-1)

    # print(w_w)
    # print('-------------------')

    #******************* attention ***************************
    if len(width_end) == 0:
        width_end.append(len(w_w))
    # print(width_start)
    # print(width_end)
    upline_coordinate = []
    if len(width_start)!= len(width_end) or len(height_start) != len(height_end):
        return
    for i in range(len(width_start)):
        upline_coordinate.append((width_start[i], height_start[0], width_end[i], height_end[0]))

    # print(upline_coordinate)
    # cropped1 = th[:,width_start[0]:width_end[0]]
    # cv2.imshow('cropped1',cropped1)
    # cv2.waitKey()
    return upline_coordinate,cropheight

