import cv2
import numpy as np
import fitz
import tkinter.messagebox
from PIL import Image

def getCoordinate(pdfPath, pageNumber):
    root = tkinter.Tk()
    root.withdraw()  # 退出默认 tk 窗口
    scale = 1
    with fitz.open(pdfPath) as pdfDoc:

        page = pdfDoc.load_page(pageNumber-1)
        mat = fitz.Matrix(scale, scale).prerotate(0)
        pix = page.get_pixmap(matrix=mat, alpha=False,)
        # pix.save('tmp.png')  # 将图片写入指定的文件夹内

        # 将Pixmap对象转换成np对象并存储在images列表中
        pix_np = np.array(Image.frombytes("RGB", (pix.width, pix.height), pix.samples), dtype=np.uint8)
        image_np = pix_np
    # 对image队对象进行放缩，得到与pdf页面1:1大小的图片
    imgNomnalize = cv2.resize(image_np, dsize=(int(image_np.shape[1]/scale), int(image_np.shape[0]/scale)),interpolation=cv2.INTER_AREA)
    # cv_show_img(imgNomnalize)
    # 框选得到表格坐标
    get_image_roi(imgNomnalize)
    

    return tableCoordinate
# 得到圈出的图像
def get_image_roi(imgNomnalize):
    '''
    获得用户ROI区域的rect=[x,y,w,h]
    :param imgNomnalize:
    :return:
    '''
    global img
    img = cv2.cvtColor(imgNomnalize, cv2.COLOR_RGB2BGR)

    cv2.namedWindow('image')
    while True:
        cv2.setMouseCallback('image', on_mouse)
        # cv2.startWindowThread()  # 加在这个位置 
        cv2.imshow('image', img)
        key = cv2.waitKey(0)
        if key == 13 or key == 32:  # 按空格和回车键退出
            result = tkinter.messagebox.askyesnocancel(title='确定', message='是否截取这个部分？')
            if result == None:
                print('画需要剪切的框')
                get_image_roi(imgNomnalize)
                break
            if result:
                break
            else:
                cv2.destroyAllWindows()
                return False
    cv2.destroyAllWindows()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return True

# 鼠标移动指令
def on_mouse(event, x, y, flags, param):
    global img, tableCoordinate
    global point1, point2
    img2 = img.copy()
    # cv2.imwrite('img2.png', img2)
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击,则在原图打点
        # print("1-EVENT_LBUTTONDOWN")
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 255, 0), 5)
        cv2.imshow('image', img2)

    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳，画框
        # print("2-EVENT_FLAG_LBUTTON")
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), thickness=2)
        cv2.imshow('image', img2)

    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放，显示
        # print("3-EVENT_LBUTTONUP")
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), thickness=2)
        cv2.imshow('image', img2)
        if point1 != point2:
            min_x = min(point1[0], point2[0])
            min_y = min(point1[1], point2[1])
            width = abs(point1[0] - point2[0])
            height = abs(point1[1] - point2[1])

            tableCoordinate = [min_x, min_y, min_x + width, min_y + height]

            cut_image = img[min_y:min_y + height, min_x:min_x + width]
            cv2.imshow('ROI', cut_image)
            