import fitz
from PIL import Image, ImageEnhance
import numpy as np
import cv2

scale = 2
# 指定pdf页面转成图片，不带文本
def pdf2img_WithoutText(pdfPath, pageNumber, scale):

    with fitz.open(pdfPath) as pdfDoc:

        page = pdfDoc.load_page(pageNumber-1)
        page.add_redact_annot([-500, -500, 3000, 3000])  # 删除该区域的所有文字
        page.apply_redactions()
        mat = fitz.Matrix(scale, scale).prerotate(0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        # pix.save('tmp.png')  # 将图片写入指定的文件夹内
        img = Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
        enhancer = ImageEnhance.Contrast(img)
        enhance_image = enhancer.enhance(factor=6) # factor 是增强因子，小于1时为减弱因子
        image_np = np.array(enhance_image)
        # image_np = np.array(Image.frombytes("RGB", (pix.width, pix.height), pix.samples), dtype=np.uint8)
        
    return image_np

def pdf2img_WithText2(pdfPath, pageNumber, scale):
    with fitz.open(pdfPath) as pdfDoc:
        page = pdfDoc.load_page(pageNumber - 1)
        # page.add_redact_annot([-500, -500, 3000, 3000])  # 删除该区域的所有文字
        # page.apply_redactions()
        mat = fitz.Matrix(scale, scale).prerotate(0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        # pix.save('tmp.png')  # 将图片写入指定的文件夹内
        img = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        enhancer = ImageEnhance.Contrast(img)
        enhance_image = enhancer.enhance(factor=6)  # factor 是增强因子，小于1时为减弱因子
        image_np = np.array(enhance_image)
        # image_np = np.array(Image.frombytes("RGB", (pix.width, pix.height), pix.samples), dtype=np.uint8)

    return image_np

# 指定pdf页面转成图片,带文本
def pdf2img_WithText(pdfPath, pageNumber, scale):

    with fitz.open(pdfPath) as pdfDoc:

        page = pdfDoc.load_page(pageNumber-1)
        mat = fitz.Matrix(scale, scale).prerotate(0)
        pix = page.get_pixmap(matrix=mat, alpha=False,)
        # pix.save('tmp.png')  # 将图片写入指定的文件夹内
        image_np = np.array(Image.frombytes("RGB", (pix.width, pix.height), pix.samples), dtype=np.uint8)

    return image_np

# 判断图像是否全为白色
def is_all_white(image, Coordinate):
    tableImage = image[Coordinate[1]:Coordinate[3], Coordinate[0]:Coordinate[2]]
    return np.all(tableImage == [255, 255, 255])

# 添加外框线
def draw_rectangle(image):
        
    # 初始化边界坐标  
    min_x, min_y = float('inf'), float('inf')  
    max_x, max_y = -float('inf'), -float('inf')  
    
    # 遍历图像寻找非白色像素的边界  
    for y in range(0,image.shape[0],2):  
        for x in range(0,image.shape[1],2):  
            # 检查像素是否为非白色（这里我们假设白色是[255, 255, 255]）  
            if not np.array_equal(image[y, x], [255, 255, 255]):  
                min_x = min(min_x, x)  
                min_y = min(min_y, y)  
                max_x = max(max_x, x)  
                max_y = max(max_y, y)  
    
    # 确保找到了非白色像素  
    if min_x == float('inf') or min_y == float('inf') or max_x == -float('inf') or max_y == -float('inf'):  
        x1,x2 = 0,0
        print("No non-white pixels found in the image.")  
    else:  
        # 计算外接矩形的左上角和右下角坐标  
        x1,y1 = (int(min_x), int(min_y))  
        x2,y2 = (int(max_x), int(max_y))  
        
        # 在图像上绘制矩形
        cv2.rectangle(image, (x1,y1), (x2,y2), (0, 0, 0), 1)
    # cv_show_img(image)
    return image

'''
    图像处理相关
'''
def cv_show_img(img):
    img = cv2.resize(img,(round(img.shape[1]/scale),round(img.shape[0]/scale)))
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
# 查看得到的单元格坐标是否与原图匹配
def show_each_retangle(image_np, cellsCoordinate):
    imagecopy = image_np.copy()
    for cellCoordinates in cellsCoordinate:
        for cellCoordinate in cellCoordinates:
            imagecopy = cv2.rectangle(imagecopy, (cellCoordinate[0],cellCoordinate[1]), (cellCoordinate[2],cellCoordinate[3]), (255, 0, 0), thickness=2)
    cv_show_img(imagecopy)

# 将图像二值化
def get_threshold(image_np):
    # 转化成灰度图
    image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)  
    # 图像像素取反
    imgBitwise = cv2.bitwise_not(image)
    # 根据阈值二值化灰度图
    Binari_image = cv2.adaptiveThreshold(imgBitwise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, -2)
    return Binari_image
