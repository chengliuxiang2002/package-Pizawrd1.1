import math

import fitz
import pdfplumber
import pandas as pd
import re
from operator import itemgetter
from sklearn.cluster import KMeans
import numpy as np


def extract_table_structure(TableImage):
    """
    输入表格图片，输出二维表格结构（无完全边框表格）
    Args:
        TableImage (np.ndarray): 原始图像
    Returns:
        list: 二维列表，表示识别出的表格内容
    """
    # 获取文本块信息
    blocks = get_text_blocks(TableImage)
    # 构建表格结构
    table = build_table_from_blocks(blocks)
    return table

def get_text_blocks(image):

    from package_core.Table_Processed.ocr_onnx.OCR_use import ONNX_Use
    words, coordinates = ONNX_Use(image, 'test')

    # 构造 blocks
    blocks = []
    for word, coord in zip(words, coordinates):
        text = word.strip()
        if text:
            x1, y1 = coord
            blocks.append({
                'text': text,
                'bbox': [x1, y1]
            })
    return blocks
def build_table_from_blocks(blocks, row_threshold=15):
    """
    表格构建算法，使用列中心聚类
    """
    if not blocks:
        return []

    # 按Y坐标排序
    blocks.sort(key=lambda b: b['bbox'][1])

    # 自适应行阈值计算
    y_coords = [b['bbox'][1] for b in blocks]
    y_diffs = [y_coords[i + 1] - y_coords[i] for i in range(len(y_coords) - 1)]
    if y_diffs:
        avg_diff = sum(y_diffs) / len(y_diffs)
        row_threshold = max(row_threshold, avg_diff * 0.5)

    # 行分组
    rows = []
    current_row = [blocks[0]]
    for block in blocks[1:]:
        last_block = current_row[-1]
        if abs(block['bbox'][1] - last_block['bbox'][1]) < row_threshold:
            current_row.append(block)
        else:
            rows.append(current_row)
            current_row = [block]
    rows.append(current_row)

    # 提取所有X坐标用于聚类
    all_x = [b['bbox'][0] for b in blocks]
    X = np.array(all_x).reshape(-1, 1)

    # 聚类数 = 最大列数
    max_cols = max(len(row) for row in rows)
    kmeans = KMeans(n_clusters=min(max_cols, len(X)), random_state=0).fit(X)
    col_centers = sorted(kmeans.cluster_centers_.flatten())

    # 每行按列中心分配文本块，并填充 "_"
    aligned_table = []
    for row in rows:
        row.sort(key=lambda b: b['bbox'][0])
        row_dict = {}
        for block in row:
            x = block['bbox'][0]
            closest_col = min(col_centers, key=lambda c: abs(c - x))
            col_idx = col_centers.index(closest_col)
            row_dict[col_idx] = block['text']

        # 插入 "_" 表示空白单元格
        aligned_row = [row_dict.get(i, '_') for i in range(len(col_centers))]
        aligned_table.append(aligned_row)

    return aligned_table

# 获取以xList, yList构成的坐标集合围成的单元格的坐标、合并单元格
def get_cells_coordinate(xList, yList, HorizontalLine, VerticalLine):
    # 将需要合并的单元格存在mergeCellsList的对应位置
    def add_coordinate_in_mergeCellsLsit(cell_1, cell_2, mergeCellsList):
        if len(mergeCellsList) == 0:
            mergeCellsList.append([cell_1, cell_2])
            return mergeCellsList
        tag = 0
        for mergeCellList in mergeCellsList:
            if cell_1 in mergeCellList:
                mergeCellList.append(cell_2)
                tag = 1
                break
        if tag == 0:
            mergeCellsList.append([cell_1, cell_2])

        return mergeCellsList
    
    # 判断是否需要合并单元格，返回True表示需要合并
    def merge_cells(cell, VerticalLine, direction):
        if direction == 0:
            # 找到与cell右框线处于同于x轴的所有线的纵坐标，横坐标比较时无用，直接舍弃
            Lines = [(line[1],line[3]) for line in VerticalLine if line[0] == cell[2]]
            cellLine = (cell[1],cell[3])
            for line in Lines:
                if cellLine[0] >= line[0] and cellLine[1] <= line[1]:
                    return False
            return True
        else:
            # 找到与cell下框线处于同一y轴的所有线的横坐标，纵坐标比较时无用，直接舍弃
            Lines = [(line[0],line[2]) for line in VerticalLine if line[1] == cell[3]]
            cellLine = (cell[0],cell[2])
            for line in Lines:
                if cellLine[0] >= line[0] and cellLine[1] <= line[1]:
                    return False
            return True
    
    Tablerows = len(yList) - 1  # 行数
    Tablecols = len(xList) - 1  # 列数
    cells_coordinate = [[[xList[j],yList[i],xList[j+1],yList[i+1]] for j in range(Tablecols)]
                        for i in range(Tablerows)]
    mergeCellsList = []
    # 行合并
    for i in range(Tablerows):
        for j in range(Tablecols):
            cell = cells_coordinate[i][j]
            if j != Tablecols - 1 and merge_cells(cell, VerticalLine, 0):
                # print(f"第{i+1}行{j+1}列需要右合并")
                mergeCellsList = add_coordinate_in_mergeCellsLsit((i,j), (i,j+1), mergeCellsList)
            if i != Tablerows - 1 and merge_cells(cell, HorizontalLine, 1):
                # print(f"第{i+1}行{j+1}列需要下合并")
                mergeCellsList = add_coordinate_in_mergeCellsLsit((i,j), (i+1,j), mergeCellsList)

    # 根据索引合并单元格坐标
    for mergeCellList in mergeCellsList:
        # 不管是先行合并还是先列合并，最后一个进入列表的总是右下角的单元格
        cellStart = cells_coordinate[mergeCellList[0][0]][mergeCellList[0][1]]
        cellEnd = cells_coordinate[mergeCellList[-1][0]][mergeCellList[-1][1]]
        truePoint = [cellStart[0],cellStart[1],cellEnd[2],cellEnd[3]]
        # 将所有被合并的单元格的坐标全设置为整个合并后的单元格的坐标
        for (i,j) in mergeCellList:
            cells_coordinate[i][j] = truePoint

    return cells_coordinate

# 根据文字坐标填充表格
def get_texts_coordinate(pdfPath, pageNumber,tableCoordinate, cellsCoordinate):
    with fitz.open(pdfPath) as pdfDoc:
        words = []
        coordinates = []
        page = pdfDoc[pageNumber -1]  # 获取第一页
        blocks = page.get_text("dict", clip=tableCoordinate)['blocks']
        blocks_filt = [x for x in blocks if x['type'] == 0]
        for block in blocks_filt:  # iterate through the text blocks
            for line in block["lines"]:  # iterate through the text lines
                string = ''
                for span in line["spans"]:  # iterate through the text spans
                    string += span['text']
                    coordinate = [round(x) for x in span['bbox']]
                    core = [(coordinate[0]+coordinate[2])/2,(coordinate[1]+coordinate[3])/2]
                    core = [round(x) for x in core]
                if string == ' ':
                    continue
                words.append(string)
                coordinates.append(core)

    zipped_list = list(zip(words, coordinates))
    zipped_list = sorted(zipped_list, key=lambda x: x[1][0])
    zipped_list = sorted(zipped_list, key=lambda x: x[1][1])
    words, coordinates = zip(*zipped_list)
    words, coordinates = list(words), list(coordinates)

    table = [['' for _ in  range(len(cellsCoordinate[0]))] for _ in range(len(cellsCoordinate))]
    for i in range(len(cellsCoordinate)):
        for j in range(len(cellsCoordinate[0])):
            for index in range(len(coordinates)):
                text = words[index].replace(' ','')
                x,y = coordinates[index]
                x1,y1,x2,y2 = cellsCoordinate[i][j]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    if table[i][j] == '':
                        table[i][j]+=text
                    else:
                        table[i][j]+='_'+ text
    return table

def get_texts_from_coordinate(pdfPath, pageNumber,tableCoordinate, cellsCoordinate):
    table = []
    try:
        with pdfplumber.open(pdfPath) as pdf :
            pdfPage = pdf.pages[pageNumber - 1]
            for i in range(len(cellsCoordinate)):
                rowText = []
                for j in range(len(cellsCoordinate[0])):
                    x1,y1,x2,y2 = cellsCoordinate[i][j]
                    text = pdfPage.within_bbox((x1,y1-4,x2,y2+4)).extract_text(y_tolerance=5).replace("\n",",").replace(" ",",")
                    rowText.append(text)
                table.append(rowText)
    except ValueError:
        table = []
    return table

# 读取不可编辑表格
def get_texts_UsingOcr(TableImage, tableCoordinate, cellsCoordinate):
    
    from package_core.Table_Processed.ocr_onnx.OCR_use import ONNX_Use
    words, coordinates = ONNX_Use(TableImage, 'test')

    for index in range(coordinates.__len__()):
        coordinates[index][0] = tableCoordinate[0]+round(coordinates[index][0]*2/4)
        coordinates[index][1] = tableCoordinate[1]+round(coordinates[index][1]*2/4)

    zipped_list = list(zip(words, coordinates))
    zipped_list = sorted(zipped_list, key=lambda x: x[1][0])
    zipped_list = sorted(zipped_list, key=lambda x: x[1][1])
    words, coordinates = zip(*zipped_list)
    words, coordinates = list(words), list(coordinates)

    table = [['' for _ in  range(len(cellsCoordinate[0]))] for _ in range(len(cellsCoordinate))]
    for i in range(len(cellsCoordinate)):
        for j in range(len(cellsCoordinate[0])):
            for index in range(len(coordinates)):
                text = words[index]
                x,y = coordinates[index]
                x1,y1,x2,y2 = cellsCoordinate[i][j]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    if table[i][j] == '':
                        table[i][j]+=text
                    else:
                        table[i][j]+='_'+ text

    return table

##1031
# 矢量方法构建表格
def _find_vector_lines(page: fitz.Page, table_rect: fitz.Rect):
    """
        在指定页面和区域内查找所有矢量水平线和垂直线。
        (V2 - 使用 get_drawings() 来捕捉 "填充矩形" 形式的线条)
    """
    h_lines, v_lines = [], []

    # 阈值：我们认为“粗细”小于 5 个单位的矩形是“线”
    MAX_LINE_THICKNESS = 8

    drawings = page.get_drawings()
    for path in drawings:
        # 确保这个矢量路径在我们的目标表格区域内
        # path["rect"] 是这个路径的边界框
        if not table_rect.contains(path["rect"]):
            continue

        # 遍历路径中的所有操作 (items)
        # 我们主要关心 "re" (rectangle) 操作，因为线条常被绘制为细矩形
        for item in path["items"]:
            if item[0] == "re":
                rect = item[1]
                # 检查这个矩形是否在我们的表格区域内(双重检查，因为 path["rect"] 只是总边界)
                if not table_rect.contains(rect):
                    continue
                # 检查这个矩形是否像一条“线”
                if rect.width > rect.height:
                    if rect.height < MAX_LINE_THICKNESS:
                        h_lines.append(rect)
                elif rect.height > rect.width:
                    if rect.width < MAX_LINE_THICKNESS:
                        v_lines.append(rect)

    # 可视化水平和竖直线段
    # 绘制表格区域和检测到的线条，用于调试和可视化
    # try:
    #     import cv2
    #     import numpy as np
    #
    #     # 获取表格区域的尺寸
    #     x0, y0, x1, y1 = table_rect
    #     width = int(x1 - x0)
    #     height = int(y1 - y0)
    #
    #     # 创建一个空白图像用于绘制
    #     img = np.zeros((height, width, 3), dtype=np.uint8)
    #
    #     # 绘制表格区域边界
    #     cv2.rectangle(img, (0, 0), (width-1, height-1), (255, 255, 255), 1)
    #
    #     # 绘制水平线段（蓝色）
    #     for line in h_lines:
    #         x0_line = int(line.x0 - x0)
    #         x1_line = int(line.x1 - x0)
    #         y_line = int(line.y0 - y0)
    #         cv2.line(img, (x0_line, y_line), (x1_line, y_line), (255, 0, 0), 2)
    #
    #     # 绘制垂直线段（绿色）
    #     for line in v_lines:
    #         x_line = int(line.x0 - x0)
    #         y0_line = int(line.y0 - y0)
    #         y1_line = int(line.y1 - y0)
    #         cv2.line(img, (x_line, y0_line), (x_line, y1_line), (0, 255, 0), 2)
    #
    #     # 保存可视化结果到文件
    #     import os
    #     if not os.path.exists('test_Result'):
    #         os.makedirs('test_Result')
    #     cv2.imwrite('test_Result/table_lines_visualization.png', img)
    #     print("INFO: 表格线段可视化图像已保存到 'test_Result/table_lines_visualization.png'")
    # except Exception as e:
    #     print(f"WARNING: 无法生成线段可视化图像: {e}")

    print(f"INFO: 在区域内找到 {len(h_lines)} 条原始水平线段, {len(v_lines)} 条原始垂直线段。")
    return h_lines, v_lines

def _find_vector_lines2(page: fitz.Page, table_rect: fitz.Rect):
    """
    在指定页面和区域内查找所有矢量水平线和垂直线。
    Args:
        page (fitz.Page): PyMuPDF 页面对象。
        table_rect (fitz.Rect): 要搜索线条的矩形区域。
    Returns:
        Tuple[List[fitz.Rect], List[fitz.Rect]]: 一个元组，包含两个列表：水平线列表和垂直线列表。
    """
    h_lines, v_lines = [], []
    # 设置线条粗细的阈值，避免将粗的矩形框识别为线
    line_thickness_threshold = 2
    # get_cdrawings() 用于提取矢量图形
    for d in page.get_cdrawings():
        rect = fitz.Rect(d["rect"])
        # 确保线条在指定的表格区域内
        if table_rect.contains(rect):
            if rect.width > rect.height and rect.height < line_thickness_threshold:
                h_lines.append(rect)
            elif rect.height > rect.width and rect.width < line_thickness_threshold:
                v_lines.append(rect)
    print(f"INFO: 在区域内找到 {len(h_lines)} 条原始水平线段, {len(v_lines)} 条原始垂直线段。")
    return h_lines, v_lines

def _is_grid_complete_by_borders(h_lines, v_lines, table_rect: fitz.Rect):
    """
    判断标准:
    - 必须存在一条紧贴区域【左侧】【右侧】的垂直线。
    只有当这个条件都满足时，才认为表格是“完整闭合”的。
    Args:
        h_lines: 水平线列表。
        v_lines: 垂直线列表。
        table_rect: 表格区域。
    Returns:
        bool: 如果表格是闭合的，返回 True，否则返回 False。
    """
    if not h_lines or not v_lines:
        return False
    tolerance = 15  # 容忍的像素误差范围
    # 检查是否存在四侧垂直边框线
    left_border_found = any(abs(line.x0 - table_rect.x0) < tolerance for line in v_lines)
    right_border_found = any(abs(line.x1 - table_rect.x1) < tolerance for line in v_lines)
    is_enclosed = left_border_found and right_border_found
    return is_enclosed


def _build_grid_boundaries(page, table_rect, h_lines, v_lines):
    """
    根据表格的完整性，构建最终的网格行、列边界坐标。
    优化点：在表格线缺失时，增加基于文本水平间隙的列推断逻辑。
    """

    def merge_close_coords(coords, tolerance: float = 2.0):
        """合并距离非常近的坐标点，以去重和修正误差。"""
        if not coords: return []
        sorted_coords = sorted(list(coords))
        merged = [sorted_coords[0]]
        for i in range(1, len(sorted_coords)):
            if sorted_coords[i] - merged[-1] > tolerance:
                merged.append(sorted_coords[i])
        return merged

    def _infer_vertical_cols_from_text(page, table_rect, h_lines):
        """
        核心优化：当缺乏竖线时，通过分析文本的水平投影空隙来推断列边界。
        """
        # 获取表格区域内的所有单词
        words = page.get_text("words", clip=table_rect)
        if not words:
            return set()

        # 1. 预处理：为了避免跨列标题（如 "Dimensions in mm"）破坏列检测，
        # 我们可以简单过滤掉宽度过大的文本块 (例如宽度超过表格宽度 60% 的文本)
        # 或者，更精细的做法是只分析表头横线以下的区域（通常是数据区）
        valid_intervals = []
        table_width = table_rect.width

        # 尝试找到表头下方的第一条横线，只利用数据区来推断列
        # 如果找不到，就用整个表格高度
        sorted_h = sorted([l.y0 for l in h_lines])
        data_top = sorted_h[1] if len(sorted_h) > 1 else table_rect.y0

        for w in words:
            # w: (x0, y0, x1, y1, "text", ...)
            w_x0, w_y0, w_x1, w_y1 = w[:4]
            w_width = w_x1 - w_x0

            # 过滤策略：
            # 1. 忽略位于顶部的跨列大标题（根据 y 坐标过滤）
            # 2. 忽略宽度过宽的文本（根据 width 过滤）
            if w_y0 < data_top and w_width > table_width * 0.5:
                continue

            valid_intervals.append((w_x0, w_x1))

        if not valid_intervals:
            return set()

        # 2. 合并区间算法 (Interval Merging)
        # 将所有文本的 X 轴投影合并，剩下的空隙就是列分割线的位置
        valid_intervals.sort(key=lambda x: x[0])
        merged = []
        if valid_intervals:
            curr_start, curr_end = valid_intervals[0]
            for next_start, next_end in valid_intervals[1:]:
                if next_start < curr_end:  # 有重叠或相连
                    curr_end = max(curr_end, next_end)
                else:
                    merged.append((curr_start, curr_end))
                    curr_start, curr_end = next_start, next_end
            merged.append((curr_start, curr_end))

        # 3. 提取空隙的中点作为列边界
        inferred_cols = set()
        for i in range(len(merged) - 1):
            gap_start = merged[i][1]
            gap_end = merged[i + 1][0]
            gap_len = gap_end - gap_start

            # 只有当空隙足够大时（例如大于3个像素），才认为是列分割
            # 这里的阈值可以根据具体文档调整
            if gap_len > 2.0:
                mid_point = (gap_start + gap_end) / 2
                inferred_cols.add(mid_point)

        return inferred_cols

    # --- 主逻辑开始 ---

    # 判断表格是否通过矢量线完整
    grid_is_complete = _is_grid_complete_by_borders(h_lines, v_lines, table_rect)
    # 也可以保留你原本的比例判断逻辑
    ratio_check = (0.9 < len(h_lines) / len(v_lines) <= 1.4) if v_lines else False

    if grid_is_complete or ratio_check:
        print("INFO: 表格完整，将使用矢量线进行构建")
        row_coords = {line.y0 for line in h_lines}
        col_coords = {line.x0 for line in v_lines}
    else:
        print("INFO: 表格不完整，正在进行文本布局修正...")
        row_coords = {line.y0 for line in h_lines}
        col_coords = {line.x0 for line in v_lines}

        # 1. 基于 Text Blocks（你原本的逻辑）
        blocks = page.get_text("blocks", clip=table_rect)
        text_based_row_coords = {block[1] for block in blocks} | {block[3] for block in blocks}
        row_coords.update(text_based_row_coords)

        # 2. 基于 Text Words 的水平投影空隙
        # 这里调用上面定义的函数
        text_based_col_coords = _infer_vertical_cols_from_text(page, table_rect, h_lines)
        col_coords.update(text_based_col_coords)

    # 确保表格区域的四个边界都被包含在内
    row_coords.update([table_rect.y0, table_rect.y1])
    col_coords.update([table_rect.x0, table_rect.x1])

    # 合并、排序并返回最终的边界
    row_boundaries = merge_close_coords(row_coords, tolerance=2.0)
    col_boundaries = merge_close_coords(col_coords, tolerance=2.0)  # 列有时候很密，Tolerance可以适当调小

    return row_boundaries, col_boundaries
def _extract_text_from_grid(page, table_rect, row_boundaries, col_boundaries):
    """
    在构建好的网格中，提取每个单元格的文本内容。
    """
    words = page.get_text("words", clip=table_rect)
    if not words:
        return [[]]
    num_rows, num_cols = len(row_boundaries) - 1, len(col_boundaries) - 1
    if num_rows <= 0 or num_cols <= 0: return [[]]

    # 初始化一个临时网格，用于存放单词对象
    temp_grid = [[[] for _ in range(num_cols)] for _ in range(num_rows)]

    for word in words:
        # 使用单词中心点来判断其归属的单元格，避免跨单元格单词被重复添加
        word_center = fitz.Point((word[0] + word[2]) / 2, (word[1] + word[3]) / 2)
        # 二分查找可以优化，但对于一般表格，线性查找足够快
        r = next((i for i, y in enumerate(row_boundaries) if y > word_center.y), num_rows) - 1
        c = next((i for i, x in enumerate(col_boundaries) if x > word_center.x), num_cols) - 1
        if 0 <= r < num_rows and 0 <= c < num_cols:
            temp_grid[r][c].append(word)

    # 排序并拼接单元格内的单词
    table_data = [["" for _ in range(num_cols)] for _ in range(num_rows)]
    for r in range(num_rows):
        for c in range(num_cols):
            # 按单词的垂直位置(y0)和水平位置(x0)排序
            sorted_words = sorted(temp_grid[r][c], key=itemgetter(1, 0))
            table_data[r][c] = " ".join([w[4] for w in sorted_words])

    return table_data
def _clean_table_data(table):
    """
    对提取出的原始表格数据进行清洗，去除空白和重复的行列。
    """
    if not table or not table[0]: return []
    # 1. 替换全角逗号为半角逗号，替换空格（换行符）为逗号
    for i in range(len(table)):
        for j in range(len(table[i])):
            if isinstance(table[i][j], str):
                table[i][j] = table[i][j].replace("，", ",")
    # 2. 删除完全空白的行
    table = [row for row in table if not all(cell.strip() == '' for cell in row)]
    if not table: return []
    # 3. 删除完全空白的列 (通过转置实现)
    table_transposed = list(zip(*table))
    table_transposed = [col for col in table_transposed if not all(cell.strip() == '' for cell in col)]
    if not table_transposed: return []
    table = [list(row) for row in zip(*table_transposed)]
    # 4. 删除内容完全重复的行
    unique_rows = []
    seen = set()
    for row in table:
        row_tuple = tuple(row)
        if row_tuple not in seen:
            unique_rows.append(row)
            seen.add(row_tuple)
    table = unique_rows

    return table

# 得到单元格合并信息
def return_Merge_info(cellsCoordinate):
    index = 0
    Is_merge_Cell = [[-1 for _ in  range(len(cellsCoordinate[0]))] for _ in range(len(cellsCoordinate))]
    i = 0
    j = 0
    merge_unit = []
    while i < len(cellsCoordinate):
        j = 0
        while j < len(cellsCoordinate[0]):
            if Is_merge_Cell[i][j] >= 0 :
                j += 1
                continue 
            # 横着找到合并边界
            HoriLen = 1
            while j + HoriLen <=  len(cellsCoordinate[0]) - 1:
                if cellsCoordinate[i][j] == cellsCoordinate[i][j+HoriLen]:
                    HoriLen += 1
                else:
                    break
            # 纵向找到合并边界
            VertLen = 1
            while i + VertLen <=  len(cellsCoordinate) -1:
                if cellsCoordinate[i][j] == cellsCoordinate[i+VertLen][j]:
                    VertLen += 1
                else:
                    break
            # 没有需要合并的单元格就开始检测下一个单元格
            if HoriLen == 1 and VertLen == 1:
                j+=1
            else:
                for tmp in range(VertLen):
                    Is_merge_Cell[i+tmp][j:j+HoriLen] = [index for _ in range(HoriLen)]
                merge_unit.append([[x,y] for y in range(j,j+HoriLen) for x in range(i,i+VertLen)])
                index += 1
                j+=HoriLen
        i += 1
    return Is_merge_Cell, merge_unit

# 删除整行为空的行和整列为空的列
def pre_check(table,cellsCoordinate):
    rows_to_delete = []
    for i in range(len(table)):
        if all(cell == '' or cell is None for cell in table[i]):
            rows_to_delete.append(i)
    for i in reversed(rows_to_delete):
        del table[i]
        if i < len(cellsCoordinate):
            del cellsCoordinate[i]
    if len(table) > 0:
        cols_to_delete = []
        for j in range(len(table[0])):
            if all(table[i][j] == '' or table[i][j] is None for i in range(len(table))):
                cols_to_delete.append(j)
        for j in reversed(cols_to_delete):
            for i in range(len(table)):
                del table[i][j]
            for i in range(len(cellsCoordinate)):
                del cellsCoordinate[i][j]

# 特定情况拆分单元格
def split_cell(table, cellsCoordinate):
    pre_check(table,cellsCoordinate)
    Is_merge_Cell, merge_unit = return_Merge_info(cellsCoordinate)
    for unit in merge_unit:
        cell_0 = unit[0]
        Merge_cell_Num = len(unit)
        string = table[cell_0[0]][cell_0[1]]
        cell_split = string.split('_')
        if cell_split.__len__() != Merge_cell_Num:
            for cell in unit:
                table[cell[0]][cell[1]] = string.replace('_','')
        else:
            for index in range(unit.__len__()):
                cell = unit[index]
                table[cell[0]][cell[1]] = cell_split[index]
    return table

# 根据表头信息判断是否是封装表格
def judge_from_title(table, title_keyword_list):
    count = 0
    # 先判断前面几行
    limit = 4 if len(table) > 3 else 2
    for i in range(limit):
        for j in range(table[0].__len__()):
            if table[i][j] == None or table[i][j].__len__()>20:
                continue
            data = table[i][j].upper().replace(" ","").replace("\n","").replace(",","")
            if any(item in data for item in title_keyword_list) :
                count+=1
    # 在判断前面几列
    if len(table[0]) == 1:
        return False
    limit = 4 if len(table[0]) > 3 else 2
    for n in range(limit):
        for m in range(table.__len__()):
            if table[m][n] == None or table[m][n].__len__()>20:
                continue
            data = table[m][n].upper().replace(" ","").replace("\n","").replace(",","").replace('D','O')
            if any(item in data for item in title_keyword_list) :
                count+=1
    # 根据是否出现这几个关键字以及出现的数量进行判断是否是封装表格
    if count >= 2:
        return True
    else:
        return False

# 根据内容信息判断是否是封装表格
def judge_from_context(table, parameter_list):
    count = 0
    for i in range(len(table)):
        for j in range(len(table[0])):
            data = table[i][j].upper().replace(" ","").replace("\n","").replace(",","").replace('D','O')
            # 出现关键字
            if any(item in data for item in parameter_list):
                flag_1 = i+1<len(table) and len(re.findall(r'\d+\.\d+',table[i+1][j])) == 1
                flag_2 = j+2<len(table[0]) and (len(re.findall(r'\d+\.\d+',table[i][j+1])) == 1 or len(re.findall(r'\d+\.\d+',table[i][j+2])) == 1)
                # 并且后面跟着数字
                if flag_1 or flag_2:
                    count += 1
    if (count >= 2 and count < 10) or (count > 25 and count < 27):
        return True
    else:
        return False
# 判断需要旋转，以及旋转方向
def Is_Common_package(table):
    orientation_tags = ['Min', 'Nom', 'Max']  # 我们要找的标签
    match_count = 0
    if not table:
        return 0  # 如果是空表，不旋转
    for row in table:
        if not row: continue  # 跳过空行
        cell_data = str(row[0]).strip()  # 获取第一列的数据
        for tag in orientation_tags:
            # 使用 startswith 确保 'Min' 匹配 'Min.' 而不是 '1.5 Min'
            if cell_data.startswith(tag):
                match_count += 1
                break  # 换下一行
    # 如果我们在第一列找到了 'Min', 'Nom', 'Max' 中的至少2个
    if match_count >= 2:
        return 0  # 表格是正的，不旋转

    JudgeList = ['A','A1','A2','D','E','e','D1','E1']
    count = 0
    # 判断第一列是否是符合标准的，符合就不旋转
    for row in table:
        for tag in JudgeList:
            if tag in row[0]:
                count += 1
                break
    if count > 3:
        return 0
    # 判断是否能找到符合条件的行
    else:
        for index in range(len(table)):
            count = 0
            row = table[index]
            matched_elements = set()  # 用于跟踪已经匹配的元素
            for cell in row:
                for element in JudgeList:
                    if element in cell and element not in matched_elements:
                        matched_elements.add(element)
                        count += 1
            if count > 3:
                if index == 0 or index == math.floor(len(table)/2) - 1:
                    return 0
                if index >= len(table)/2:
                    return 90
                # 转表不转图
                elif index == len(table)/2 - 1:
                    return -1
        # 没找到就直接顺时针旋转270，即逆时针旋转90
        return 270
# 转表不转图
def rotate_table(table):
    tmp = [[table[i][j] for i in range(len(table))] for j in range(len(table[0]))]
    return tmp
# 表格坐标变换
def Table_coordinate_transformation(tableCoordinate, cellsCoordinate, image , direction):
    if direction == 90:
        height =  image.shape[0]
        for i in range(cellsCoordinate.__len__()):
            for j in range(cellsCoordinate[0].__len__()):
                x1,y1,x2,y2 = cellsCoordinate[i][j]
                cellsCoordinate[i][j] = [height - y2, x1, height - y1, x2]
        tmp = cellsCoordinate
        cellsCoordinate = [[tmp[tmp.__len__() - 1-j][i] for j in  range(tmp.__len__())] for i in range(tmp[0].__len__())]
        x1,y1,x2,y2 = tableCoordinate
        tableCoordinate = [height - y2, x1, height - y1, x2]
    elif direction == 270:
        width = image.shape[1]
        for i in range(cellsCoordinate.__len__()):
            for j in range(cellsCoordinate[0].__len__()):
                x1,y1,x2,y2 = cellsCoordinate[i][j]
                cellsCoordinate[i][j] = [y1, width - x2, y2, width - x1]
        tmp = cellsCoordinate
        cellsCoordinate = [[tmp[j][tmp[0].__len__()-1 - i] for j in  range(tmp.__len__())] for i in range(tmp[0].__len__())]
        x1,y1,x2,y2 = tableCoordinate
        tableCoordinate = [y1, width - x2, y2, width - x1]
    return tableCoordinate, cellsCoordinate

# 含unit的表格处理
def get_data_from_unit_table(table, unit):
    data = []
    tableCols = len(table[0])
    table = [list(i) for i in zip(*table)]
    for index in range(len(unit)):
        col,row = unit[index]
        for i in range(row+1,tableCols):
            tmp = table[i][col+1].split('_')
            tmp = [x.replace(' ','') for x in tmp]
            if len(tmp) == 1:
                tmp.append(tmp[0])
                tmp.append(tmp[0])
            elif len(tmp) == 2:
                tmp.append('-')
            data.append(
                [
                    table[i][col].replace(' ',''),
                    tmp[0],
                    tmp[2],
                    tmp[1]
                ]
                )
    return data

# 找到min nom max所在列
def find_MIN_NOM_MAX(table):
    global tmp
    cols = []
    title = 0
    limit = 4 if len(table) > 2 else 2
    for i in range(limit):
        for j in range(table[0].__len__()):
            if table[i][j] == None or table[i][j].__len__()>20:
                continue
            data = table[i][j].upper().replace(" ","").replace("\n","").replace(",","")
            if any(item in data for item in ['NOM','MOM','MAX','MIN','TYP','MN','最小','最大','公称','典型','推荐值']) and j not in cols:
                cols.append(j)
                title = i+1
    if cols.__len__() == 1:
        table = [list(row) for row in zip(*table)]
        cols = []
        for i in range(limit):
            for j in range(table[0].__len__()):
                if table[i][j] == None or table[i][j].__len__()>20:
                    continue
                data = table[i][j].upper().replace(" ","").replace("\n","").replace(",","")
                if any(item in data for item in ['NOM','MOM','MAX','MIN','MN','TYP','最小','最大','公称','典型','推荐值']) and j not in cols:
                    cols.append(j)
                    title = i+1
                if data == 'A':
                    tmp = j
        if cols.__len__() == 3:
            # 无论怎么排布保证字母在数字前一列
            if tmp != cols[0] - 1:
                for row in table:
                    # 交换指定列的元素  
                    row[cols[0] - 1], row[tmp] = row[tmp], row[cols[0] - 1]
            return title, [cols], table
    if cols.__len__() == 3:
        return title, [cols], table
    elif cols.__len__() == 6 and cols[3]== cols[2] + 1:
        return title, [cols[:3]], table
    elif cols.__len__() == 6 and cols[3]!= cols[2] + 1:
        return title, [cols[:3],cols[3:]], table
    elif cols.__len__() == 7:
        return title, [cols[:3],cols[3:]], table
    elif cols.__len__() == 4 and cols[2]== cols[1] + 1:
        return title, [cols], table
    elif cols.__len__() == 2:
        return title, [cols], table
    else:
        return 0,[], table

def delete_space_row(data):
    """
    删除整行为空的行，整行为相同数据的行，
    以及整列为空或整列数据完全相同的列。
    """

    # --- 步骤 1: 删除整行为空的行 ---
    non_empty_rows = [row for row in data if any(str(cell).strip() != '' for cell in row)]
    if not non_empty_rows:
        return []
        # --- 步骤 2: (新增) 删除整行数据相同的行 ---
    # 这将删除像 ['Header', 'Header', 'Header'] 这样的行
    valid_rows = []
    for row in non_empty_rows:
        if not row:  # 以防万一的空行
            continue
        # 将所有单元格转为字符串并去除首尾空格
        stripped_row = [str(cell).strip() for cell in row]
        # 检查是否所有值都相同
        first_val = stripped_row[0]
        if all(cell == first_val for cell in stripped_row):
            # 如果所有值都相同，则丢弃这一行
            continue
        # 如果行既不全空，也不全相同，则保留
        valid_rows.append(row)
    if not valid_rows:
        return []  # 所有行都被删除了
    # --- 步骤 3: 转置 (Transpose) ---
    # 注意：我们现在操作的是 valid_rows, 而不是 non_empty_rows
    try:
        transposed = list(zip(*valid_rows))
    except IndexError:
        # 如果行长度不一致，返回当前结果
        return valid_rows
    if not transposed:
        return []  # 转置后为空
    # --- 步骤 4: 删除“无效”的列 (全空 or 全相同) ---
    valid_cols = []
    for col in transposed:
        if not col:  # 如果列本身是空的
            continue  # 不保留
        # 将所有单元格转为字符串并去除首尾空格
        stripped_col = [str(cell).strip() for cell in col]
        # 检查是否全为空
        if all(cell == '' for cell in stripped_col):
            continue  # 不保留 (删除空列)
        # 检查是否所有值都相同 (e.g., ['Junk', 'Junk', 'Junk'])
        first_val_col = stripped_col[0]
        if all(cell == first_val_col for cell in stripped_col):
            continue  # 不保留 (删除重复列)
        # 如果列既不全空，也不全相同，则是有效数据，保留
        valid_cols.append(col)
    if not valid_cols:
        return []  # 没有有效列
    # --- 步骤 5: 再次转置，恢复原状 ---
    result = list(zip(*valid_cols))
    result = [list(row) for row in result]
    return result

def delete_thesame_cols(data):
    """
        删除整行为空的行，以及整列为空或整列数据完全相同的列。
        """
    # 1. 删除全为空字符串的行
    #    (使用 str() 和 strip() 增加健壮性)
    non_empty_rows = [row for row in data if any(str(cell).strip() != '' for cell in row)]
    if not non_empty_rows:
        return []  # 表格为空
    # 2. 转置 (Transpose)
    try:
        # 使用 zip(*...) 来转置
        transposed = list(zip(*non_empty_rows))
    except IndexError:
        # 如果行长度不一致，返回当前结果
        return non_empty_rows
    if not transposed:
        return []  # 转置后为空
    # 3. 删除“无效”的列
    valid_cols = []
    for col in transposed:
        if not col:  # 如果列本身是空的
            continue  # 不保留
        # 将所有单元格转为字符串并去除首尾空格
        stripped_col = [str(cell).strip() for cell in col]
        # 检查是否全为空
        if all(cell == '' for cell in stripped_col):
            continue  # 不保留 (这是你原来的逻辑)
        # 检查是否所有值都相同 (e.g., ['Junk', 'Junk', 'Junk'])
        # 这会删除那些填充了 'PACKAGE TYPE...' 的列
        first_val = stripped_col[0]
        if all(cell == first_val for cell in stripped_col):
            continue  # 不保留 (这是你建议的新逻辑)
        # 如果列既不全空，也不全相同，则是有效数据，保留
        valid_cols.append(col)
    if not valid_cols:
        return []  # 没有有效列
    # 4. 再次转置，恢复原状
    result = list(zip(*valid_cols))
    result = [list(row) for row in result]
    return result

# 根据数值找到数据所在列
def find_number_col(table):
    cols = []
    title = []
    # 遍历每一列
    for i in range(table[0].__len__()):
        tmp = 0
        count = 0
        for j in range(table.__len__()):
            if bool(re.match(r'^[a-zA-Z][0-9]$', table[j][i])):
                continue
            if len(re.findall("\d+",table[j][i])) > 0 or table[j][i]=='' or table[j][i]=='-'or table[j][i]=='--':
                count += 1
                tmp = j if tmp == 0 else tmp
        if count/table.__len__() > 0.5:
             cols.append(i)
             title.append(tmp)
    title = min(title)
    if cols.__len__() == 3:
        return title, [cols]
    elif cols.__len__() == 6 and cols[3]== cols[2] + 1:
        return title, [cols[:3]]
    elif cols.__len__() == 6 and cols[3]!= cols[2] + 1:
        return title, [cols[:3],cols[3:]]
    elif cols.__len__() == 4 and cols[2]== cols[1] + 1:
        return title, [cols]
    elif cols.__len__() == 1:
        return title, [cols]

# 常规表格处理
def get_data_from_common_table(table):
    data = []
    table = delete_space_row(table)
    # 找到目标值所在行列，一般会有MIN NOM MAX等标识
    title, Paircols, table = find_MIN_NOM_MAX(table)
    if Paircols == []:
        title, Paircols = find_number_col(table)
    tableRows = len(table)
    # print(table)
    # 可能会有分成6列是并列关系的存在
    for cols in Paircols:
        for i in range(title,tableRows):
            if cols.__len__() == 3:
                if 'D1,E1' in table[i][cols[0]-1]:
                    table[i][cols[0]-1] = table[i][cols[0]-1].replace('D1,E1','D1E1')
                data.append(
                    [
                        table[i][cols[0]-1], 
                        table[i][cols[0]], 
                        table[i][cols[1]], 
                        table[i][cols[2]]
                    ]
                    )
            elif cols.__len__() == 1:
                
                data.append(
                    [
                        table[i][cols[0]-1], 
                        table[i][cols[0]]
                    ]
                    )
            elif cols.__len__() == 2:
                data.append(
                    [
                        table[i][cols[0]-1], 
                        table[i][cols[0]], 
                        '',
                        table[i][cols[1]]
                    ]
                    )
            else:
                    data.append(
                    [
                        table[i][cols[0]-1], 
                        table[i][cols[0]], 
                        table[i][cols[1]], 
                        table[i][cols[2]], 
                        table[i][cols[3]], 
                    ]
                    )
    count_same = 1
    for index in range(data.__len__()-1):
        if data[index][0] == data[index+1][0]:
            count_same += 1
    if count_same > 7 and count_same == len(data[0][0].split('_')):
        tmp = data[0][0].split('_')
        for rowIndex in range(count_same):
            data[rowIndex][0] = tmp[rowIndex]
    # if data[0][0].split(',').__len__() == data.__len__():
    #     tmp = data[0][0].split(',')
    #     for rowIndex in range(data.__len__()):
    #         data[rowIndex][0] = tmp[rowIndex]

    return data

# 从table中提取出需要的行和列 
def get_info_from_table(table):
    if table == []:
        return table
    unitTag = []
    symbolTag = []
    result = []
    tableCols = len(table[0])
    for i in range(2):
        for j in range(tableCols):
            tmp = table[i][j].upper().replace(" ","").replace("\n","").replace(",","")
            if 'UNIT' in tmp or 'UNT' in tmp:
                unitTag.append([i,j])
            if 'SYMBOL' in tmp or 'MIN' in tmp or 'NOM' in tmp or 'MAX' in tmp and '_MAX' not in tmp:
                symbolTag.append([i,j])
    if unitTag != [] and symbolTag == [] :
        result = get_data_from_unit_table(table, unitTag)
    # 多列
    else:
        result = get_data_from_common_table(table)

    return result
 
 # 对OCR提取的结果进行过滤

def filt_KeyInfo_data(lst):
    if lst.__len__() == 1:
        number = re.findall("[1-9]\d*.\d*|0\.\d*[1-9]\d*", lst[0])
        number = [float(x) for x in number]
        if len(number) == 1:
            return [number[0],number[0],number[0]]
        elif len(number) == 2:
            if number[0] - number[1] < 0:
                return [number[0],(number[0]+number[1])/2,number[1]]
            else:
                return [number[0]- number[1],number[0],number[1]+number[0]]
    data = []
    tmp = []
    count = 0
    # print(lst)
    lst = [str(num.replace(',', '.')) for num in lst]
    for index in range(lst.__len__()):
        try:
            # 表格读取代码不通用
            if len(lst[index].split('_')) >= 3 and len(re.findall("[1-9]\d*.\d*|0\.\d*[1-9]\d*", lst[index])) == 3:
                tmp = [lst[index].split('_')[0],lst[index].split('_')[1],lst[index].split('_')[2]]
                tmp = [float(x) for x in tmp]
                break
            lst[index] = lst[index].replace('B','').replace('S','').replace('C.','').replace(',','').replace('C','')
            str_list = re.findall("[1-9]\d*.\d*|0\.\d*[1-9]\d*", lst[index])
            number = ''
            for x in str_list:
                number += x
            tmp.append(float(number))
        except:
            count+=1
            tmp.append(0)
    if count < 2:
        if tmp[0] == 0:
            data = [max(0,round(2*tmp[1]-tmp[2],2)),tmp[1],tmp[2]]
        elif tmp[1] == 0:
            data = [tmp[0], round((tmp[0]+tmp[2])/2,2), tmp[2]]
        elif tmp[2] == 0:
            data = [tmp[0],tmp[1],round(2*tmp[1] - tmp[0],2)]
        else:
            data = [tmp[0],tmp[1],tmp[2]]
    elif count == 2:
        singleNumber = [x for x in tmp if x!=0][0]
        data = [singleNumber,singleNumber,singleNumber]
    else:
        data = ['','','']

    return data

def table_checked(table):
    # 遍历前两行
    for i in range(min(2, len(table))):
        row = table[i]
        # 查找包含 MIN, NOM, MAX 的列索引
        for j in range(len(row)):
            cell = row[j].strip().upper() if row[j] else ""
            if ("MIN" in cell) or ("MN" in cell):
                row[j] = "min"
                if j + 2 < len(row):
                    row[j + 1] = "nom"
                    row[j + 2] = "max"
                break
            elif "NOM" in cell:
                if j - 1 >= 0:
                    row[j - 1] = "min"
                row[j] = "nom"
                if j + 1 < len(row):
                    row[j + 1] = "max"
                break
            elif "MAX" in cell:
                if j - 2 >= 0:
                    row[j - 2] = "min"
                    row[j - 1] = "nom"
                row[j] = "max"
                break
                
    return table

def keyInfo_checked(key_list):
    """
    检查key_list中的数据，如果每个列表都是['A', '0.700', '0.800', '0.028', '0.031']这种情况，
    说明存在两种单位的数据，只留下前两列数字，这两列分别是min和max，
    在这两列中添加nom列，数据相加然后除以二
    """
    # 检查是否每个列表都符合第一个元素是字母，后面四个元素是数字的模式
    def is_valid_row(row):
        if len(row) != 5:
            return False
        # 第一个元素应该是字符串且第一个字符是字母
        if not isinstance(row[0], str) or not row[0][0].isalpha():
            return False
        # 后四个元素应该是数字字符串
        for i in range(1, 5):
            try:
                float(row[i])
            except ValueError:
                return False
        return True

    # 检查所有行是否都符合模式
    if all(is_valid_row(row) for row in key_list):
        # 如果是这种情况，处理每一行数据
        processed_list = []
        for row in key_list:
            # 只保留前两列数字（min和max），它们分别是第二个和第三个元素（索引1和2）
            if len(row) >= 3:
                param_name = row[0]  # 第一列是参数名
                min_val = row[1]     # 第二列是min值
                max_val = row[2]     # 第三列是max值
                
                # 计算nom值 (min+max)/2
                try:
                    # 如果是数值字符串，则计算平均值
                    min_num = float(min_val)
                    max_num = float(max_val)
                    nom_val = (min_num + max_num) / 2
                    
                    # 创建新行：参数名，min, nom, max
                    new_row = [param_name, min_val, str(round(nom_val, 6)), max_val]
                    processed_list.append(new_row)
                except ValueError:
                    # 如果无法转换为数值，则保持原始值
                    new_row = [param_name, min_val, '', max_val]
                    processed_list.append(new_row)
            else:
                # 如果行数据不足，则保持原样
                processed_list.append(row)
        return processed_list
    else:
        # 如果不符合特定模式，保持原样
        return key_list

# 根据标准字母语义从表中找出信息进行对应
def add_info_from_KeyInfo(data, KeyInfo,packageType):
    # BGA类型的语义匹配
    if KeyInfo == []:
        return KeyInfo
    for key in KeyInfo :
        print(key)
    # print(KeyInfo)
    def BGA_add_info(data, row):
        row = [s.strip() for s in row]
        # print(row)
        row[0] = row[0].replace('_','')
        if row[0] == '0' or row[0] == '9' or row[0] == 'Φb':
            row[0] = 'b'
        row[0] = row[0].split(',')[0].replace(' ','')
        row = [x.replace('.BSC','').replace(',BSC','').replace('8SC','') for x in row]
        # 列Pitch
        if ('e' in row[0]) and data[0][1] == '':
            data[0][1:] = filt_KeyInfo_data(row[1:])
        # 行Pitch
        if ('e' in row[0] or '0' in row[0]) and data[1][1] == '':
            data[1][1:]  = filt_KeyInfo_data(row[1:])
        if ('eD' in row[0]) and data[0][1] == '':
            data[0][1:]  = filt_KeyInfo_data(row[1:])
        if ('eE' in row[0]) and data[1][1] == '':
            data[1][1:]  = filt_KeyInfo_data(row[1:])
        # 列数
        # if row[0] =='eD' and data[8][1] == '':
        #     data[8][1:]  = filt_KeyInfo_data(row[1:])
        # 行数
        # if (row[0] =='e' or row[0]=='eE' or row[0] == 'e0')and data[7][1] == '':
        #     data[7][1:]  = filt_KeyInfo_data(row[1:])
        # # 列数
        # if ('MD' in row[0] or row[0] == 'M') and data[6][1] == '':
        #     data[6][1:] = filt_KeyInfo_data(row[1:])
        # 行数
        # if ('ME' in row[0] or row[0] == 'M') and data[5][1] == '':
        #     data[5][1:] = filt_KeyInfo_data(row[1:])
        # 球直径
        if 'b' in row[0] and data[9][1] == '':
            data[9][1:]  = filt_KeyInfo_data(row[1:])
        # 支撑高
        if ('A1' in row[0]) and data[5][1] == '':
            data[5][1:]  = filt_KeyInfo_data(row[1:])
        # 实体高
        if row[0]=='A'and data[4][1] == '':
            data[4][1:] = filt_KeyInfo_data(row[1:])
        # 实体宽
        if ('E' in row[0] or 'DE' in row[0]) and data[7][1] == '':
            # row[1:] = [f"{float(num)/10:.1f}" if int(num) > 100 else num for num in row[1:]]
            data[7][1:]  = filt_KeyInfo_data(row[1:])
        # 实体长
        if ('D' in row[0] or 'DE' in row[0]) and data[6][1] == '':
            data[6][1:]  = filt_KeyInfo_data(row[1:])
    # QFP类型的语义匹配
    def QFP_add_info(data, row):
        row = [s.strip() for s in row]
        # print(row)
        row[0] = row[0].replace('_','')
        if row[0] == '0' or row[0] == '9' or row[0] == 'Φb':
            row[0] = 'b'
        row[0] = row[0].split(',')[0].replace(' ','')
        row = [x.replace('.BSC','').replace(',BSC','').replace('8SC','') for x in row]
        # 引脚的厚度
        if 'c' in row[0] and data[12][1] == '':
            data[12][1:]  = filt_KeyInfo_data(row[1:])
        # 散热盘宽
        if row[0] =='E2' and data[11][1] == '':
            data[11][1:]  = filt_KeyInfo_data(row[1:])
        # 散热盘长
        if row[0] =='D2' and data[10][1] == '':
            data[10][1:]  = filt_KeyInfo_data(row[1:])
        # 行/列Pin数
        if (row[0] =='e' or row[0]=='eE' or row[0] == 'e0')and data[9][1] == '':
            data[9][1:]  = filt_KeyInfo_data(row[1:])
        # pin宽
        if 'b' in row[0] and data[8][1] == '':
            data[8][1:]  = filt_KeyInfo_data(row[1:])
        # pin长
        if 'L' in row[0] and data[7][1] == '':
            data[7][1:]  = filt_KeyInfo_data(row[1:])
        # 外围宽
        if 'E' in row[0] and data[6][1] == '':
            data[6][1:]  = filt_KeyInfo_data(row[1:])
        # 外围长
        if 'D' in row[0] and data[5][1] == '':
            data[5][1:]  = filt_KeyInfo_data(row[1:])   
        # 端子高
        if 'A3' in row[0] and data[4][1] == '':
            data[4][1:]  = filt_KeyInfo_data(row[1:])
        # 支撑高
        if ('A1' in row[0]) and data[3][1] == '':
            data[3][1:]  = filt_KeyInfo_data(row[1:])
        # 实体高
        if row[0]=='A'and data[2][1] == '':
            data[2][1:] = filt_KeyInfo_data(row[1:])
        # 实体宽
        if 'E1' in row[0] and data[1][1] == '':
            # row[1:] = [f"{float(num)/10:.1f}" if int(num) > 100 else num for num in row[1:]]
            data[1][1:] = filt_KeyInfo_data(row[1:])
        # 实体长
        if 'D1' in row[0] and data[0][1] == '':
            data[0][1:] = filt_KeyInfo_data(row[1:])
    if packageType == 'BGA':
        # print(KeyInfo)
        for row in KeyInfo:
            try:
                BGA_add_info(data,row)
            except:
                continue
    elif packageType == 'QFP':
        for row in KeyInfo:
            try:
                QFP_add_info(data,row)
            except:
                continue
    elif packageType == 'QFN':
        for row in KeyInfo:
            row = [s.strip() for s in row]
            # print(row)
            row[0] = row[0].replace('_','')
            if row[0] == '0' or row[0] == '9' or row[0] == 'Φb':
                row[0] = 'b'
            row[0] = row[0].split(',')[0].replace(' ','')
            row = [x.replace('.BSC','').replace(',BSC','').replace('8SC','') for x in row]
            # Pin_Pitch
            if 'e' in row[0] and data[10][1] == '':
                data[10][1:] = filt_KeyInfo_data(row[1:])
            if 'e' in row[0] and data[9][1] == '':
                data[9][1:] = filt_KeyInfo_data(row[1:])
            # 引脚的厚度
            if ('c' in row[0]) or ('C' in row[0]) and data[12][1] == '':
                data[12][1:]  = filt_KeyInfo_data(row[1:])
            # 散热盘长
            if row[0] =='D2' and data[11][1] == '':
                data[11][1:]  = filt_KeyInfo_data(row[1:])
            # # 列Pin数
            # if row[0] =='eD' and data[10][1] == '':
            #     data[10][1:]  = filt_KeyInfo_data(row[1:])
            # # 行Pin数
            # if (row[0]=='eE' or row[0] == 'e0')and data[9][1] == '':
            #     data[9][1:]  = filt_KeyInfo_data(row[1:])
            # 列数
            if ('MD' in row[0] or row[0] == 'M') and data[8][1] == '':
                data[8][1:] = filt_KeyInfo_data(row[1:])
            # 行数
            if ('ME' in row[0] or row[0] == 'M') and data[7][1] == '':
                data[7][1:] = filt_KeyInfo_data(row[1:])
            # pin宽
            if 'b' in row[0] and data[6][1] == '':
                data[6][1:]  = filt_KeyInfo_data(row[1:])
            # pin长
            if 'L' in row[0] and data[5][1] == '':
                data[5][1:]  = filt_KeyInfo_data(row[1:])
            # 端子高
            if 'A3' in row[0] and data[4][1] == '':
                data[4][1:]  = filt_KeyInfo_data(row[1:])
            # 支撑高
            if 'A1' in row[0] and data[3][1] == '':
                data[3][1:]  = filt_KeyInfo_data(row[1:])
            # 实体高
            if row[0]=='A'and data[2][1] == '':
                data[2][1:] = filt_KeyInfo_data(row[1:])
            # 实体宽
            if 'E' in row[0] and data[1][1] == '':
                # row[1:] = [f"{float(num)/10:.1f}" if int(num) > 100 else num for num in row[1:]]
                data[1][1:]  = filt_KeyInfo_data(row[1:])
            # 实体长
            if 'D' in row[0] and data[0][1] == '':
                data[0][1:]  = filt_KeyInfo_data(row[1:])
    elif packageType == 'SON' or packageType == 'SOP':
        for row in KeyInfo:
            row = [s.strip() for s in row]
            # print(row)
            row[0] = row[0].replace('_','')
            if row[0] == '0' or row[0] == '9' or row[0] == 'Φb':
                row[0] = 'b'
            try:
                row[0] = row[0].split(',')[0].replace(' ','')
                row = [x.replace('.BSC','').replace(',BSC','').replace('8SC','') for x in row]
                # pin宽
                if 'b' in row[0] and data[4][1] == '':
                    data[4][1:]  = filt_KeyInfo_data(row[1:])
                # pin长
                if ('L' in row[0]) and data[3][1] == '':
                    data[3][1:]  = filt_KeyInfo_data(row[1:])
                # 实体高
                if row[0]=='A'and data[2][1] == '':
                    data[2][1:] = filt_KeyInfo_data(row[1:])
                if row[0] == 'Amax.' and data[2][1] == '':
                    data[2][1:] = filt_KeyInfo_data(row[1:])
                if row[0] == 'A(1)max.' and data[2][1] == '':
                    data[2][1:] = filt_KeyInfo_data(row[1:])
                # 实体宽
                if 'E' in row[0] and data[1][1] == '':
                    # row[1:] = [f"{float(num)/10:.1f}" if int(num) > 100 else num for num in row[1:]]
                    data[1][1:]  = filt_KeyInfo_data(row[1:])
                # 实体长
                if 'D' in row[0] and data[0][1] == '':
                    data[0][1:]  = filt_KeyInfo_data(row[1:])
                # Pin_Pitch
                if 'e' in row[0] and data[9][1] == '':
                    data[9][1:] = filt_KeyInfo_data(row[1:])
            except:
                continue
    return data