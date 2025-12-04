import fitz  # PyMuPDF
import os


# def draw_rect_on_pdf(pdf_path, page_num, rect, output_path=None):
#     """
#     在PDF页面上绘制矩形框（使用左上角坐标系）
#     :param rect: [x0, y0, x1, y1] 左上角坐标系
#     """
#     doc = fitz.open(pdf_path)
#     page = doc[page_num]
#
#     # 获取页面当前旋转角度
#     rotation = page.rotation
#
#     # 根据旋转角度调整坐标转换逻辑
#     if rotation == 0 or rotation == 180:
#         # 正常方向或上下颠倒
#         pdf_rect = [
#             rect[0],  # x0不变
#             page.rect.height - rect[3],  # y0转换
#             rect[2],  # x1不变
#             page.rect.height - rect[1]  # y1转换
#         ]
#     else:  # 90或270度旋转
#         # 宽高已互换，需要调整坐标转换方式
#         pdf_rect = [
#             rect[1],  # x0 = 原y0
#             page.rect.width - rect[2],  # y0 = 新宽度 - 原x1
#             rect[3],  # x1 = 原y1
#             page.rect.width - rect[0]  # y1 = 新宽度 - 原x0
#         ]
#
#     # 绘制绿色矩形框
#     page.draw_rect(fitz.Rect(pdf_rect), color=(0, 1, 0), width=2)
#
#     # 保存文件
#     output_path = output_path or pdf_path
#     doc.save(output_path, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
#     doc.close()

#
def draw_rect_on_pdf(pdf_path, page_num, rect, output_path=None):
    """
    在PDF页面上直接绘制矩形框（不考虑旋转角度）

    :param pdf_path: PDF文件路径
    :param page_num: 页码（从0开始）
    :param rect: [x0, y0, x1, y1] 矩形坐标（使用PDF坐标系，原点在左上角）
    :param output_path: 输出文件路径（None则覆盖原文件）
    """
    # 打开PDF文档
    doc = fitz.open(pdf_path)

    try:
        # 获取指定页
        page = doc[page_num]

        # 直接使用输入的rect坐标创建矩形（不进行任何坐标转换）
        pdf_rect = fitz.Rect(rect)

        # 绘制绿色矩形框（颜色RGB格式，宽度2pt）
        page.draw_rect(pdf_rect, color=(0, 0, 1), width=2)

        # 保存文件（增量更新保留原加密状态）
        output_path = output_path or pdf_path
        doc.save(output_path, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
        print(f"矩形框已绘制到第{page_num + 1}页，保存至: {output_path}")

    finally:
        doc.close()



def rotate_pdf_page(pdf_path, page_num, angle, output_path):
    """
    使用PyMuPDF旋转PDF页面，并保存到新的PDF文件
    :param angle: 必须是90的倍数(0, 90, 180, 270)
    :return: (新宽度, 新高度)
    """
    if angle % 90 != 0:
        angle = 0

    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # 获取旋转前的页面尺寸
    original_rect = page.rect
    original_width, original_height = original_rect.width, original_rect.height

    # 设置页面旋转
    page.set_rotation(angle)

    # 保存文件
    doc.save(output_path, garbage=4, deflate=True)
    doc.close()

    # 计算旋转后的页面尺寸
    if angle in (90, 270):
        new_width, new_height = original_height, original_width
    else:
        new_width, new_height = original_width, original_height

    return new_width, new_height


def transform_coordinates(rect, angle, original_width, original_height):
    """
    转换坐标到旋转后的左上角坐标系
    :param rect: [x0, y0, x1, y1] 旋转前的左上角坐标系
    :param angle: 旋转角度(0, 90, 180, 270)
    :return: 旋转后的左上角坐标系[x0, y0, x1, y1]
    """
    x0, y0, x1, y1 = rect

    if angle == 0:
        return rect
    elif angle == 90:
        # 顺时针90度：x' = height - y, y' = x
        return [
            original_height - y1,  # 新x0
            x0,                    # 新y0
            original_height - y0,  # 新x1
            x1                     # 新y1
        ]
    elif angle == 180:
        # 顺时针180度：x' = width - x, y' = height - y
        return [
            original_width - x1,   # 新x0
            original_height - y1,  # 新y0
            original_width - x0,   # 新x1
            original_height - y0   # 新y1
        ]
    elif angle == 270:
        # 顺时针270度：x' = y, y' = width - x
        return [
            y0,                    # 新x0
            original_width - x1,   # 新y0
            y1,                    # 新x1
            original_width - x0    # 新y1
        ]
    else:
        raise ValueError("不支持的旋转角度，必须是0、90、180或270度")


if __name__ == "__main__":
    pdf_path = 'temp/C514430_DC-DC电源芯片_LT8645SEV#PBF_规格书_WJ109290.PDF'
    output_path = r'D:\20250822\PackageWizard1.0_F1\PDF_Processed\detected_packages.pdf'
    page_num = 13

    rotated_rect = [427, 80, 427, 250]
    rects=[[537,117,719,292]]
    for rect in rects:
        draw_rect_on_pdf(output_path,page_num, rect)
    # draw_rect_on_pdf(output_path, page_num, rotated_rect)
