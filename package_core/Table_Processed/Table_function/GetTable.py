import fitz

from package_core.Table_Processed.Table_function.dataProcess import (
    _find_vector_lines,
    _find_vector_lines2,
    _build_grid_boundaries,
    _extract_text_from_grid,
    _clean_table_data
)
from package_core.Table_Processed.Table_function.getBoder import *
from package_core.Table_Processed.Table_function.dataProcess import *
from package_core.Table_Processed.Table_function.Tool import *
from operator import itemgetter


def check_image_overlap(page: fitz.Page, table_rect: fitz.Rect, threshold: float = 0.4) -> bool:
    """
    检查表格区域内是否有大面积的图像覆盖。

    Args:
        page (fitz.Page): PyMuPDF 页面对象。
        table_rect (fitz.Rect): 表格的边界框。
        threshold (float): 面积重叠的阈值 (0.0 到 1.0)。

    Returns:
        bool: 如果图像重叠面积超过阈值，返回 True (应跳过矢量提取)；
              否则返回 False (可以尝试矢量提取)。
    """
    print(f"INFO: 正在检查表格区域 {table_rect} 的图像重叠情况...")

    # 1. 获取表格总面积
    table_area = table_rect.get_area()
    if table_area == 0:
        print("WARNING: 表格区域面积为 0，跳过检查。")
        return False  # 避免除以零，这是一个异常情况

    total_image_overlap_area = 0.0

    # 2. 遍历页面上的所有图像
    images = page.get_images(full=True)
    found_images_in_rect = 0

    for img_info in images:
        img_rect = page.get_image_bbox(img_info)  # 获取图像的边界框

        # 3. 检查图像是否与表格区域相交
        if table_rect.intersects(img_rect):
            found_images_in_rect += 1

            # 4. 计算相交区域 ('&' 操作符)
            intersection_rect = table_rect & img_rect
            intersection_area = intersection_rect.get_area()

            total_image_overlap_area += intersection_area
            print(f"  -> 找到相交图像 {img_rect}，相交面积: {intersection_area:.2f}")

    if found_images_in_rect == 0:
        print("INFO: 表格区域内未找到图像。将使用矢量提取。")
        return False  # 没有图像，安全

    # 5. 计算重叠比例并与阈值比较
    overlap_ratio = total_image_overlap_area / table_area
    print(f"INFO: 图像总重叠面积: {total_image_overlap_area:.2f}，表格总面积: {table_area:.2f}")
    print(f"INFO: 图像覆盖比例: {overlap_ratio * 100:.2f}%")

    if overlap_ratio > threshold:
        print(f"INFO: 图像覆盖比例 ({overlap_ratio * 100:.2f}%) 超过阈值 ({threshold * 100}%)。")
        print("INFO: 应跳过矢量提取，切换至图像处理 (CV) 方法。")
        return True
    else:
        print(f"INFO: 图像覆盖比例低于阈值。仍将尝试矢量提取。")
        return False

def judge_if_image_into(pdfPath, pageNumber, Coordinate):
    try:
        doc = fitz.open(pdfPath)
    except Exception as e:
        return []
    page = doc.load_page(pageNumber - 1)
    table_rect = fitz.Rect(Coordinate)
    print("--- 正在检查区域内是否有图像 ---")
    images_in_rect = 0
    # 注意: 这里的 'page' 和 'table_rect' 变量
    # 需要是你当前上下文中真实的对象
    for img in page.get_images(full=True):
        xref = img[0]
        img_rect = page.get_image_bbox(img)  # 获取图像的边界框
        if table_rect.intersects(img_rect):  # 检查是否与表格区域相交
            print(f"INFO: 找到一个与表格区域相交的图像: {img_rect}")
            images_in_rect += 1

    if images_in_rect > 0:
        print(f"--- 检查完毕：表格区域内有 {images_in_rect} 个图像。")
        if check_image_overlap(page, table_rect):
            return True
        return False
    else:
        print("--- 检查完毕：表格区域内未找到图像，将尝试矢量提取。 ---")
        return False


def get_texts_UsingVector(pdfPath, pageNumber, Coordinate):
    """
    从PDF页面指定区域智能提取表格。
    该函数会自动判断表格线条是否完整，并选择最佳策略进行提取。
    Args:
        pdf_path (str): PDF文件路径。
        page_num (int): 要处理的页面页码 (从1开始)。
        table_area (list): 包含四个值的列表 [x0, y0, x1, y1]，定义了表格区域。
        top_line_coordinates (List[List[float]], optional): 从外部传入的上划线中心点坐标列表 [[x, y], ...]。
    Returns:
        List[List[str]]: 清洗后的二维列表，代表表格数据。如果失败则返回空列表。
    """
    try:
        doc = fitz.open(pdfPath)
    except Exception as e:
        return []
    # PyMuPDF 页面索引从0开始，需要-1
    page = doc.load_page(pageNumber - 1)
    table_rect = fitz.Rect(Coordinate)
    # 步骤 1: 查找所有矢量线条
    h_lines, v_lines = _find_vector_lines(page, table_rect)
    if len(h_lines) == 0 or len(v_lines) == 0:
        print("INFO: 方法一矢量线条未找到，将尝试方法二获取矢量线条。")
        h_lines, v_lines = _find_vector_lines2(page, table_rect)
    # 步骤 2: 根据线条完整性，构建网格边界
    row_boundaries, col_boundaries = _build_grid_boundaries(page, table_rect, h_lines, v_lines)
    # 步骤 3: 在构建的网格中提取文本
    raw_table = _extract_text_from_grid(page, table_rect, row_boundaries, col_boundaries)
    # 步骤 4: 清洗提取出的数据
    cleaned_table = _clean_table_data(raw_table)
    doc.close()
    return cleaned_table

#提取封装表格流程
def get_table(pdfPath, pageNumber, Coordinate):

    with fitz.open(pdfPath) as doc:
        page = doc.load_page(pageNumber-1)
        # 获取页面上的文本块
        blocks = page.get_text("blocks", clip=Coordinate)

    if blocks.__len__() < 5:
        TableImage = Get_Ocr_TableImage(pdfPath, pageNumber, Coordinate)
        # 获取图片表格的框线信息
        scale = 2
        try:
            image = pdf2img_WithoutText(pdfPath, pageNumber, scale)
            tableCoordinate = [round(x * scale) for x in Coordinate]
            xList, yList, HorizontalLine, VerticalLine = get_Border(image, tableCoordinate)
        except:
            image = pdf2img_WithText(pdfPath, pageNumber, scale)
            tableCoordinate = [round(x * scale) for x in Coordinate]
            xList, yList, HorizontalLine, VerticalLine = get_Border(image, tableCoordinate)
        # 得到所有单元格的坐标
        cellsCoordinate = get_cells_coordinate(xList, yList, HorizontalLine, VerticalLine)
        # 单元格可视化
        # show_each_retangle(image, cellsCoordinate)
        table = get_texts_UsingOcr(TableImage, tableCoordinate, cellsCoordinate)
        rotate = Is_Common_package(table)
        # 转表不转图
        if rotate == -1:
            table = rotate_table(table)
        if rotate == 90:
            TableImage = cv2.rotate(TableImage, cv2.ROTATE_90_CLOCKWISE)
            tableCoordinate, cellsCoordinate = Table_coordinate_transformation(tableCoordinate, cellsCoordinate, image, direction=90)
            table = get_texts_UsingOcr(TableImage, tableCoordinate, cellsCoordinate)
        elif rotate == 270:
            TableImage = cv2.rotate(TableImage, cv2.ROTATE_90_COUNTERCLOCKWISE)
            tableCoordinate, cellsCoordinate = Table_coordinate_transformation(tableCoordinate, cellsCoordinate, image, direction=270)
            table = get_texts_UsingOcr(TableImage, tableCoordinate, cellsCoordinate)
            if Is_Common_package(table) == 270:
                return []
        # 对没有框线分割但实际上需要进行单元格拆分的情况进行处理
        table = split_cell(table, cellsCoordinate)
    # elif judge_if_image_into(pdfPath, pageNumber, Coordinate):
    #     TableImage = Get_Ocr_TableImage(pdfPath, pageNumber, Coordinate)
    #     # 获取图片表格的框线信息
    #     scale = 2
    #     try:
    #         image = pdf2img_WithoutText(pdfPath, pageNumber, scale)
    #         tableCoordinate = [round(x * scale) for x in Coordinate]
    #         xList, yList, HorizontalLine, VerticalLine = get_Border(image, tableCoordinate)
    #     except:
    #         image = pdf2img_WithText(pdfPath, pageNumber, scale)
    #         tableCoordinate = [round(x * scale) for x in Coordinate]
    #         xList, yList, HorizontalLine, VerticalLine = get_Border(image, tableCoordinate)
    #     # 得到所有单元格的坐标
    #     cellsCoordinate = get_cells_coordinate(xList, yList, HorizontalLine, VerticalLine)
    #     # 单元格可视化
    #     # show_each_retangle(image, cellsCoordinate)
    #     table = get_texts_UsingOcr(TableImage, tableCoordinate, cellsCoordinate)
    #     rotate = Is_Common_package(table)
    #     # 转表不转图
    #     if rotate == -1:
    #         table = rotate_table(table)
    #     if rotate == 90:
    #         TableImage = cv2.rotate(TableImage, cv2.ROTATE_90_CLOCKWISE)
    #         tableCoordinate, cellsCoordinate = Table_coordinate_transformation(tableCoordinate, cellsCoordinate, image, direction=90)
    #         table = get_texts_UsingOcr(TableImage, tableCoordinate, cellsCoordinate)
    #     elif rotate == 270:
    #         TableImage = cv2.rotate(TableImage, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #         tableCoordinate, cellsCoordinate = Table_coordinate_transformation(tableCoordinate, cellsCoordinate, image, direction=270)
    #         table = get_texts_UsingOcr(TableImage, tableCoordinate, cellsCoordinate)
    #         if Is_Common_package(table) == 270:
    #             return []
    #     # 对没有框线分割但实际上需要进行单元格拆分的情况进行处理
    #     table = split_cell(table, cellsCoordinate)
    else:
        table = get_texts_UsingVector(pdfPath, pageNumber, Coordinate)

    return table

# 根据字符判断是否是封装表格
def judge_if_package_table(table, packageType):
    if (table is None) or (table == []) or (len(table) == 1):
        return False
    title_keyword_list = ['NOM','MAX','MIN','TYP','最小','最大','公称']
    if packageType == 'BGA':
        data = ['D','E','A','A1','b','e','e1','SD','SE']
    if packageType == 'QFN':
        data = ['D','E','A','A1','A3','L','b','e','D2','E2']
    if packageType == 'QFP':
        data = ['D1','E1','A','A1','A3','L','b','e','D2','E2','c']
    if packageType == 'SON' or packageType == 'SOP':
        data = ['D','E','A','L','b']
    # 进行两次判断，先判断根据关键词判断，再根据数值进行判断
    if judge_from_title(table, title_keyword_list) or judge_from_context(table, data):
        return True
    else:
        return False

# 判断表格是否完整
def judge_if_complete_table(table, packageType):
    if (table is None) or (table == []) or (len(table) == 1):
        return False
    if packageType == 'BGA':
        data = ['D','E','A','A1','b','e']
    elif packageType == 'QFN':
        data = ['D','E','A','A1','L','b','e']
    elif packageType == 'QFP':
        data = ['D1','E1','A','A1','L','b','e','D2','E2']
    elif packageType == 'SOP' or packageType == 'SON':
        data = ['D','E','A','L','b']

    count = 0
    for j in range(len(table[0])):
        for i in range(len(table)):
            # 如果table[i][j]有数字编号带括号，将括号和内容删除
            if re.search(r'\(\d+\)', str(table[i][j])):
                table[i][j] = re.sub(r'\(\d+\)', '', str(table[i][j]))
            if table[i][j] == 'A_A': count += 1
            if table[i][j] == 'eD' or table[i][j] == 'eE': count += 1
            if table[i][j] == 'Φb' or table[i][j] == 'G': count += 1
            if table[i][j] == 'DE': count += 4
            if table[i][j] == 'D/E': count += 2
            if table[i][j] == '':
                continue
            if any(table[i][j] in item for item in data):
                count+=1
    if count>=len(data):
        return True
    else:
        return False

# 将表与其续表进行合并
def table_Merge(Table_1, Table_2):
    col_table_1 = len(Table_1[0])
    col_table_2 = len(Table_2[0])
    # 列数相同就直接合并
    if col_table_1 == col_table_2:
        return Table_1+ Table_2
    # 续表与前一页表格列不相同就先拓展再合并
    elif col_table_1 > col_table_2:
        extend_table = [['' for _ in range(len(Table_1[0]))]for _ in range(len(Table_2))]
        # 对表进行填充
        for i in range(len(extend_table)):
            for j in range(len(extend_table[0])):
                extend_table[i][j] = Table_2[i][j] if j < col_table_2 else Table_2[i][col_table_2 - 1]
        return Table_1 + extend_table
    else:
        # col_table_1 < col_table_2 续表列数大于前一页表格列数
        # 删除续表的多余列（空列或者数据相同的列）
        Table_2_modified = [list(row) for row in Table_2]
        # 先删除超出需要的列（从右到左）
        while len(Table_2_modified[0]) > col_table_1 + 1:
            col_index = len(Table_2_modified[0]) - 1
            # 收集该列的所有值
            column_values = []
            for row in Table_2_modified:
                if col_index < len(row):
                    cell_value = row[col_index].strip()
                    column_values.append(cell_value)
                else:
                    column_values.append('')  # 补全缺失行
            # 判断是否为空列（所有值都是空字符串）
            is_empty_column = True
            for value in column_values:
                if value != '':
                    is_empty_column = False
                    break
            # 判断是否为全列相同数据
            is_same_data_column = True
            if len(column_values) > 0:
                first_value = column_values[0]
                for value in column_values:
                    if value != first_value:
                        is_same_data_column = False
                        break
            # 如果是空白列或整列数据都一样，则删除该列
            if is_empty_column or is_same_data_column:
                for row in Table_2_modified:
                    if col_index < len(row):
                        row.pop(col_index)
            else:
                break
        # 检查前col_table_1列中是否有空列或数据相同列
        cols_to_remove = []  # 记录需要删除的列索引
        for col_index in range(col_table_1 - 1, -1, -1):  # 从右到左检查前col_table_1列
            # 收集该列的所有值
            column_values = []
            for row in Table_2_modified:
                if col_index < len(row):
                    cell_value = row[col_index].strip()
                    column_values.append(cell_value)
                else:
                    column_values.append('')  # 补全缺失行
            is_empty_column = True
            for value in column_values:
                if value != '':
                    is_empty_column = False
                    break
            is_same_data_column = True
            if len(column_values) > 0:
                first_value = column_values[0]
                for value in column_values:
                    if value != first_value:
                        is_same_data_column = False
                        break
                # 如果所有值都相同且都是空字符串，则认为是可删除的相同数据列
                # 如果所有值都相同但不是空字符串，则只在特定条件下删除
                if first_value != '' and is_same_data_column:
                    # 这里我们只删除空列或完全相同且无意义的列
                    # 对于非空但相同的数据列，我们保留它们
                    is_same_data_column = False
            # 如果是空白列，则记录该列索引以便删除
            if is_empty_column:
                cols_to_remove.append(col_index)
        # 删除记录的列（从右到左删除以保持索引正确）
        for col_index in sorted(cols_to_remove, reverse=True):
            for row in Table_2_modified:
                if col_index < len(row):
                    row.pop(col_index)
            # 更新col_table_1，因为前面的列被删除了
            col_table_1 -= 1
        # 检查处理后的续表列数是否与前表相等
        if len(Table_2_modified[0]) == col_table_1:
            return Table_1 + Table_2_modified
        else:
            # 如果列数仍然不相等，则拓展前一页表格
            extend_table = [['' for _ in range(len(Table_2_modified[0]))]for _ in range(len(Table_1))]
            # 对表进行填充
            for i in range(len(extend_table)):
                for j in range(len(extend_table[0])):
                    extend_table[i][j] = Table_1[i][j] if j < col_table_1 else Table_1[i][col_table_1 - 1]
            return extend_table + Table_2_modified
# 根据信息判断是否需要进行表格合并
def table_Select(Table, Type, Integrity):
    if len(Table) == 1:
        return Table[0]
    if len(Type) == 2 and len(Integrity) == 2:
        Type = [Type[0],False,Type[1]]
        Integrity = [Integrity[0],False,Integrity[1]]
        Table = [Table[0],[], Table[1]]
    # 当页表有效完全
    if Type[1] == True:
        # 当前页有效且完全
        if Integrity[1] == True:
            return Table[1]
        # 当前页有效但是不完全
        else:
            # 相邻页表均无效
            if Type[0] != True and Type[2] != True:
                return Table[1]
            # 相邻页的表有效
            else:
                # 下一页的表有效
                if Type[2] == True:
                    # 且完全
                    if Integrity[2] == True:
                        return Table[1]
                    # 不完全
                    else:
                        return table_Merge(Table[1], Table[2])
                # 上一页的表有效
                elif Type[0] == True:
                    # 且完全
                    if Integrity[0] == True:
                        return Table[1]
                    # 不完全
                    else:
                        return table_Merge(Table[0], Table[1])
                # 上一页、下一页均为有效表
                else:
                    return Table[1]
    # 当前页的表无效
    else:
        # 上一页有效且完全
        if Type[0] == True and Integrity[0] == True:
            return Table[0]
        # 下一页有效且完全
        elif Type[2] == True and Integrity[2] == True:
            return Table[2]
        else:
            return []
# 后处理
def get_nx_ny_from_title(page_num, nx, ny):
    import json
    if nx == '':
        nx = 0
    if ny == '':
        ny = 0
    pin_nums = ''
    # 指定要查找的 page_num
    target_page_num = page_num
    json_file = 'package_baseinfo.json'
    result = []
    # 读取 JSON 文件
    print('正在读取JSON文件...')
    # with open(json_file, 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    with open(json_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content:
            try:
                data = json.loads(content)
                print("解析成功")
                for item in data:
                    if item['page_num'] == target_page_num:
                        result.append(item['pin'])
                        result.append(item['length'])
                        result.append(item['width'])
                        result.append(item['height'])
                        result.append(item['horizontal_pin'])
                        result.append(item['vertical_pin'])
                print("json文件读取完毕")
                print("json:", result)
            except json.JSONDecodeError as e:
                print("JSON 解析失败:", e)
        else:
            print("文件为空")
    # 遍历列表，查找匹配的条目

    if result != []:
        if result[0] != None:
            if result[4] != None and result[5] != None:

                    nx = str(result[4])
                    ny = str(result[5])
            pin_nums = result[0]

    return nx, ny,pin_nums
def postProcess(table, packageType):
    data = [['' for _ in  range(4)] for _ in range(13)]
    KeyInfo = get_info_from_table(table)
    KeyInfo = keyInfo_checked(KeyInfo) #检查keyinfo的特殊情况
    data = add_info_from_KeyInfo(data,KeyInfo,packageType)

    print("postProcess:", data)
    return data