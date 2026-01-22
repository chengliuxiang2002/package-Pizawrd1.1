import cv2
import numpy as np
from package_core.Table_Processed.Table_function.Tool import *

# F4.2 有底色表格处理
def No_Background_Process(img):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thre = [5, 10, 20, 50, 80, 150, 200, 225, 240, 250]
    contours = []
    for i in thre:
        _, binary = cv2.threshold(imgray, i, 255, 0)
        contour, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours.extend(contour)
    height, width = img.shape[:2]
    binary_image = np.zeros((height, width, 1), dtype=np.uint8)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        area_cnt = cv2.contourArea(contour)
        if area_cnt / area >= 0.9 or w / h < 0.01 or w / h > 100:
            con = cv2.rectangle(binary_image, (x, y), (x + w, y + h), 255, 3)

    kernel = np.ones((3, 3), np.uint8)
    con = cv2.erode(con, kernel, iterations=1)
    return con

# 仅保留表格部分的Mat
def GetTableImage(image, area):
    tableImage = np.zeros((area[3]-area[1], area[2]-area[0], 3), np.uint8) 
    tableImage = image[area[1]:area[3], area[0]:area[2]]
    return tableImage

def Get_Ocr_TableImage(pdfPath, pageNumber,Coordinate):
    scale = 4
    with fitz.open(pdfPath) as pdfDoc:
        x1,y1,x2,y2 = Coordinate
        page = pdfDoc.load_page(pageNumber-1)
        clip_rect = fitz.Rect(x1,y1,x2,y2)
        mat = fitz.Matrix(scale, scale).prerotate(0)
        pix = page.get_pixmap(matrix=mat, alpha=False,clip = clip_rect)
        # pix.save('tmp.png')  # 将图片写入指定的文件夹内
        tableImage = np.array(Image.frombytes("RGB", (pix.width, pix.height), pix.samples), dtype=np.uint8)
        # img = Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
        # enhancer = ImageEnhance.Contrast(img)
        # enhance_image = enhancer.enhance(factor=3) # factor 是增强因子，小于1时为减弱因子

        # tableImage = np.array(enhance_image)
    return tableImage

# F4.3 找到需要添加的内框线
def search_add_line(image_np, tableCoordinate, scale, direction):
    # 二值化图像在y轴或者x轴方向的投影统计
    def binImgHist(img, bin_width=1, direction=1):
        '''
            二值化图像在y轴或者x轴方向的投影统计
        '''
        height, width = img.shape
        bins = None
        if direction == 0:
            # 在y轴方向上统计
            bins = int(height / bin_width)
        else:
            bins = int(width / bin_width)
        # 获取非零元素坐标
        nonzero_points = np.nonzero(img)
        # 获取非零元素坐标中的x坐标集合或者y坐标集合
        nonzero_idx_x = nonzero_points[direction]
        #返回的统计直方图
        hist = np.histogram(np.int64(nonzero_idx_x), bins=bins)[0]

        return hist
    
    # 根据阈值筛选出的坐标进行匹配
    def find_split_lines(possible_lines,direction):
        splitLines = []
        count = 0
        threshold = 2 if direction == 0 else 30
        for i in range(1,len(possible_lines)):
            if possible_lines[i] == (possible_lines[i-1] + 1):
                count += 1
                continue
            elif count < threshold:
                count = 0
                continue
            else:
                line = round((possible_lines[i-1]-count+1)) if direction == 0 else round((possible_lines[i-1]-count/2))
                splitLines.append(line)
                count = 0
      
        return splitLines
    

    
    # 从图片中切割出表格
    tableImage= GetTableImage(image_np, tableCoordinate)
    Area = tableImage.shape[0]*tableImage.shape[1]
    # 二值化
    Binari_image = get_threshold(tableImage)
    # 找到所有轮廓
    contours,_ = cv2.findContours(Binari_image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    # 将二值化图像中所有轮廓涂白色
    for Box in boundingBoxes:
        x,y,w,h = [int(i) for i in Box]
        if w*h > Area*0.01:
            continue
        Binari_image[y:y+h,x:x+w] = 255
    # 每个单位横向有多少个像素点是白色
    hist = binImgHist(Binari_image, 1, direction)
    possible_lines = np.where(hist < 5)[0] if direction == 0 else np.where(hist < 10)[0]
    # 找到分割线
    splitLines = find_split_lines(possible_lines,direction)
    # 划线
    if direction == 1:
        addLine = [round(line+tableCoordinate[0]) for line in splitLines]
    else:
        addLine = [round(line+tableCoordinate[1]) for line in splitLines]

    tableImageCopy = tableImage.copy()
    for line in splitLines:
        if direction == 1 :
            tableImageCopy = cv2.line(tableImageCopy, (line, 0), (line, tableCoordinate[3]- tableCoordinate[1]), (0,0,0), 1) 
        else:
            tableImageCopy = cv2.line(tableImageCopy, (0, line), (tableCoordinate[2]- tableCoordinate[0], line), (0,0,0), 1) 

    # cv_show_img(tableImageCopy)
    return addLine, direction

# 标准化框线
def delete_redundanceLine_reset_net(xList, yList, HorizontalLine, VerticalLine):
    # 找到最接近的网格点作为线段的起始点
    def find_closest_number_Index(arr, target):  
        left, right = 0, len(arr) - 1  
        closest = arr[0]  # 初始化为数组的第一个元素  

        while left <= right:  
            mid = (left + right) // 2  
            if arr[mid] == target:
                return mid  # 如果找到目标值，直接返回  
            elif arr[mid] < target:  
                closest = arr[mid] if abs(target - arr[mid]) < abs(target - closest) else closest  
                left = mid + 1
            else:
                closest = arr[mid] if abs(target - arr[mid]) < abs(target - closest) else closest  
                right = mid - 1

        return arr.index(closest)  # 返回与目标值最接近的元素的位置
    # 遍历每一条横线
    redundance_HorizontalLine = []
    for index in range(len(HorizontalLine)):
        # 记录线段的初始位置
        Line = HorizontalLine[index]
        LineStart, LineEnd = Line[0], Line[2]
        # 找到线段在初始网格中的起止位置
        IndexStart = find_closest_number_Index(xList, LineStart) 
        IndexEnd = find_closest_number_Index(xList, LineEnd)
        if IndexEnd - IndexStart == 0:
            redundance_HorizontalLine.append(Line)
        elif IndexEnd - IndexStart == 1:
            # 设置偏移量，适当扩大约束条件
            offset = (LineEnd - LineStart)*0.05
            # 线段太短则判断为是冗余线段，需要删除
            if (LineStart - offset) > xList[IndexStart] and (LineEnd + offset) < xList[IndexEnd]:
                redundance_HorizontalLine.append(Line)
        else:
            HorizontalLine[index][0], HorizontalLine[index][2] = LineStart, LineEnd
            # HorizontalLine[index][1] = yList[find_closest_number_Index(yList, Line[1])]
            # HorizontalLine[index][3] = HorizontalLine[index][1]
    # 剔除冗余线段
    HorizontalLine = [Line for Line in HorizontalLine if Line not in redundance_HorizontalLine]
    yList = [Line[1] for Line in HorizontalLine]
    yList = list(set(yList))
    yList = sorted(yList, key = lambda x:x)
    i = 0
    while i < len(yList) - 1:
        if yList[i + 1] - yList[i] < 10:
            yList.pop(i + 1)
        else:
            i += 1
            
    # 遍历每一条竖线
    redundance_VerticalLine = []
    for index in range(len(VerticalLine)):
        Line = VerticalLine[index]
        # 记录线段的初始位置
        LineStart, LineEnd = Line[1], Line[3]
        # 找到线段在初始网格中的起止位置
        IndexStart = find_closest_number_Index(yList, LineStart) 
        IndexEnd = find_closest_number_Index(yList, LineEnd)
        if IndexEnd - IndexStart == 0:
            redundance_VerticalLine.append(Line)
        elif IndexEnd - IndexStart == 1: 
            # 设置偏移量，适当扩大约束条件
            offset = round((LineEnd - LineStart)*0.08)
            # 线段太短则判断为是冗余线段，需要删除
            if (LineStart - offset) > yList[IndexStart] or (LineEnd + offset*2) < yList[IndexEnd]:
                redundance_VerticalLine.append(Line)
        else:
            VerticalLine[index][1], VerticalLine[index][3] = LineStart, LineEnd
            # VerticalLine[index][0] = xList[find_closest_number_Index(xList, Line[0])]
            # VerticalLine[index][2] = VerticalLine[index][0]
    VerticalLine = [Line for Line in VerticalLine if Line not in redundance_VerticalLine]
    xList = [Line[0] for Line in VerticalLine]
    xList = list(set(xList))
    xList = sorted(xList, key = lambda x:x)
    # 对网格线进行后处理，合并过近的竖线，保留谁
    i = 0
    while i < len(xList) - 1:
        if xList[i + 1] - xList[i] < 15:
            xList.pop(i + 1)
        else:
            i += 1
    return xList, yList, HorizontalLine, VerticalLine

# 找到所有的线
def findLines(BinaryThreshold, tableCoordinate = []):  # img_path
    # 标准化框线，找到最接近的网格点作为线段的起始点
    def find_closest_number(arr, target):  
        left, right = 0, len(arr) - 1  
        closest = arr[0]  # 初始化为数组的第一个元素  
    
        while left <= right:  
            mid = (left + right) // 2  
            if arr[mid] == target:  
                return arr[mid]  # 如果找到目标值，直接返回  
            elif arr[mid] < target:  
                closest = arr[mid] if abs(target - arr[mid]) < abs(target - closest) else closest  
                left = mid + 1  
            else:
                closest = arr[mid] if abs(target - arr[mid]) < abs(target - closest) else closest  
                right = mid - 1
    
        return closest  # 返回与目标值最接近的元素 
    # 得到每每条框线的起止点，并返回横向或纵向有哪些坐标
    def get_HorizontalLine_Coordinate(image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yList = []
        Line = []
        # 遍历轮廓并获取每个轮廓的起止点
        for contour in contours:
            # 轮廓的点是按顺序排列的，所以可以取第一个点和最后一个点作为起止点
            combined_array = np.vstack(contour)  
            max_x = np.max(combined_array[:, 0])
            min_x = np.min(combined_array[:, 0])
            max_y = np.max(combined_array[:, 1])
            yList.append(max_y)
            Line.append([min_x,max_y,max_x,max_y])
        # 对得到的List和Line进行后处理
        yList = list(set(sorted(yList)))
        yList = sorted(yList, key = lambda x:x)
        sum_of_diffs = 0
        for i in range(1, len(yList)):  
            # 计算当前元素和前一个元素的差值，并累加到sum_of_diffs  
            sum_of_diffs += yList[i] - yList[i-1]
        i = 0
        if yList[i+1] - yList[i] < 7:
            yList.pop(i)
        while i < len(yList) - 1:
            if yList[i + 1] - yList[i] < 8:
                yList.pop(i + 1)
            else:
                i += 1
        return yList, Line
    def get_VerticalLine_Coordinate(image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        List = []
        Line = []
        # 遍历轮廓并获取每个轮廓的起止点
        for contour in contours:
            # 轮廓的点是按顺序排列的，所以可以取第一个点和最后一个点作为起止点
            combined_array = np.vstack(contour)  
            max_x = np.max(combined_array[:, 0])
            min_y = np.min(combined_array[:, 1])
            max_y = np.max(combined_array[:, 1])
            List.append(max_x)
            Line.append([max_x,min_y,max_x,max_y])
        List = list(set(sorted(List)))
        List = sorted(List, key = lambda x:x)
        sum_of_diffs = 0
        for i in range(1, len(List)):  
            # 计算当前元素和前一个元素的差值，并累加到sum_of_diffs  
            sum_of_diffs += List[i] - List[i-1]
        i = 0
        if List[i+1] - List[i] < 7:
            List.pop(i)
        while i < len(List) - 1:
            if List[i + 1] - List[i] < 15:
                List.pop(i + 1)
            else:
                i += 1
        return List, Line
    horizontal = BinaryThreshold.copy()
    vertical = BinaryThreshold.copy()

    horizontalSize = 30
    # 构造横向卷积核
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
    # 图像腐蚀
    horizontal = cv2.erode(horizontal, horizontalStructure, iterations=1)
    # 图像膨胀
    horizontal = cv2.dilate(horizontal, horizontalStructure, iterations=1)
    yList, HorizontalLine = get_HorizontalLine_Coordinate(horizontal)

    verticalsize = 20
    # 构造纵向卷积核
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    # 图像腐蚀
    vertical = cv2.erode(vertical, verticalStructure, iterations = 1)
    # 图像膨胀
    vertical = cv2.dilate(vertical, verticalStructure, iterations= 1)
    xList, VerticalLine= get_VerticalLine_Coordinate(vertical)
    HorizontalLine = sorted(HorizontalLine, key = lambda x:x[1])
    VerticalLine = sorted(VerticalLine, key = lambda x:x[0])

    # 剔除冗余框线
    xList, yList, HorizontalLine, VerticalLine = delete_redundanceLine_reset_net(xList, yList, HorizontalLine, VerticalLine)
    # 将线的端点与网格点相匹配
    for index in range(len(HorizontalLine)):
        HorizontalLine[index][0] = find_closest_number(xList,HorizontalLine[index][0]) 
        HorizontalLine[index][2] = find_closest_number(xList,HorizontalLine[index][2])
        HorizontalLine[index][1] = find_closest_number(yList,HorizontalLine[index][1])
        HorizontalLine[index][3] = find_closest_number(yList,HorizontalLine[index][3])
    for index in range(len(VerticalLine)):
        VerticalLine[index][1] = find_closest_number(yList,VerticalLine[index][1])
        VerticalLine[index][3] = find_closest_number(yList,VerticalLine[index][3])
        VerticalLine[index][0] = find_closest_number(xList,VerticalLine[index][0])
        VerticalLine[index][2] = find_closest_number(xList,VerticalLine[index][2])

    return xList, yList, HorizontalLine, VerticalLine

def get_Border(image_np, tableCoordinate):
    # 表格无底色时，进行二值化
    if image_np.shape.__len__() == 3:
        # 1. 仅保留框线中的内容
        image_np_crop = np.full_like(image_np, 255)
        image_np_crop[tableCoordinate[1]:tableCoordinate[3],tableCoordinate[0]:tableCoordinate[2]] = image_np[tableCoordinate[1]:tableCoordinate[3],tableCoordinate[0]:tableCoordinate[2]]
        # 2. 画出外框
        image_draw_line = draw_rectangle(image_np_crop)
        # cv_show_img(image_draw_line)
        # 3. 对图像进行二值化
        BinaryThreshold = get_threshold(image_draw_line)
    # 表格有底色时，输入的是处理后的二值化图片
    else:
        BinaryThreshold = np.full_like(image_np, 0)
        BinaryThreshold[tableCoordinate[1]:tableCoordinate[3],tableCoordinate[0]:tableCoordinate[2]] = image_np[tableCoordinate[1]:tableCoordinate[3],tableCoordinate[0]:tableCoordinate[2]]
    # 4. opencv得到表格框选中拐角处的坐标
    xList, yList, HorizontalLine, VerticalLine = findLines(BinaryThreshold)

    return xList, yList, HorizontalLine, VerticalLine

def add_Line_Boder(xList, yList, HorizontalLine, VerticalLine, addLine, direction):
    if direction == 0:
        yList.extend(addLine)
        yList = sorted(yList, key = lambda x:x)
        for line in addLine:
            HorizontalLine.append([xList[0],line,xList[-1],line])
    elif direction == 1:
        xList.extend(addLine)
        xList = sorted(xList, key = lambda x:x)
        for line in addLine:
            VerticalLine.append([line,yList[0],line,yList[-1]])

    return xList, yList, HorizontalLine, VerticalLine
