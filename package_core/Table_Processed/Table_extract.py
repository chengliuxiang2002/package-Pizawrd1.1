from pathlib import Path
#F3.表格内容解析与判断 F4.表格规范化流程
from package_core.Table_Processed.Table_function.GetTable import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, numpy as np, requests, io, base64

# ===== 可配环境变量 =====
MODEL_HOST = os.getenv('TABLE_MODEL_HOST', '127.0.0.1')
MODEL_PORT = int(os.getenv('TABLE_MODEL_PORT', 8080))
MODEL_PATH = os.getenv('TABLE_MODEL_PATH', '/table/predict')
MODEL_TIMEOUT = int(os.getenv('TABLE_MODEL_TIMEOUT', 5))
# ========================

# 探活缓存（同之前）
_last_probe_ts = 0
_model_alive_cache = False
def _model_alive():
    global _last_probe_ts, _model_alive_cache
    import time, threading, socket
    with threading.Lock():
        now = time.time()
        if now - _last_probe_ts < 30:
            return _model_alive_cache
        _last_probe_ts = now
        try:
            with socket.create_connection((MODEL_HOST, MODEL_PORT), timeout=1):
                _model_alive_cache = True
        except Exception:
            _model_alive_cache = False
        return _model_alive_cache

def crop_pdf_page(pdf_path: str, page_number: int, coords: list) -> Image.Image:
    """
    把 PDF 指定页的指定坐标区域裁成 PIL.Image
    :param pdf_path:  PDF 文件路径
    :param page_number:  1-based 页码（第一页传 1）
    :param coords:       [x0, y0, x1, y1]  PDF 逻辑点
    :return:             PIL.Image 对象（RGB）
    """
    # 打开文档
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number - 1]  # PyMuPDF 0-based
        x0, y0, x1, y1 = coords
        # 生成矩阵：分辨率 300 dpi（可改）
        mat = fitz.Matrix(300 / 72, 300 / 72)
        # 裁剪区域
        pix = page.get_pixmap(matrix=mat, clip=fitz.Rect(x0, y0, x1, y1))
        # 转 PIL
        img_bytes = pix.tobytes("png")
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    finally:
        doc.close()

# 选图（单页里裁坐标，但只给模型一张图）
def _crop_single_page(pdfPath, pageNumber, TableCoordinate):
    """返回 PIL Image"""
    return crop_pdf_page(pdfPath, pageNumber, TableCoordinate)  # 你已有实现

# 模型调用
def _model_predict(image, packageType):
    """image: PIL Image -> dict {"A":[...], ...}"""
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    url = f"http://{MODEL_HOST}:{MODEL_PORT}{MODEL_PATH}"
    rsp = requests.post(url,
                        json={'image': b64, 'package': packageType},
                        timeout=MODEL_TIMEOUT)
    rsp.raise_for_status()
    return rsp.json()

# JSON -> 二维表（复用老 table_Select 逻辑）
def _json_to_table(json_dict):
    """
    把模型输出的 {"A":[1.2,1.2,1.2], "B":[...]} 转成二维表：
    [表头, 均值行]  这样后续 table_Select / postProcess 一行不用改
    """
    headers = list(json_dict.keys())
    row   = [float(np.mean(v)) for v in json_dict.values()]
    return [headers, row]


def _process_single_page(args):
    """
    处理单个页面的表格提取（用于并行处理）
    Args:
        args: (pageNumber, TableCoordinate, pdfPath, packageType)

    Returns:
        (table, type_result, integrity_result)
    """
    pageNumber, TableCoordinate, pdfPath, packageType = args
    if TableCoordinate == []:
        return [], False, False

    # 1. 模型分支
    # if _model_alive():
    #     try:
    #         print("INFO:调用大模型进行预测")
    #         # img = _crop_single_page(pdfPath, pageNumber, TableCoordinate)
    #         # pred = _model_predict(img, packageType)
    #         pred ="```json\n{\"parameters\": {\"A\": [1.2, 1.2, 1.2], \"A1\": [0.05, 0.1, 0.15], \"b/B\": [0.17, 0.2, 0.27], \"D\": [16.0, 16.0, 16.0], \"E\": [16.0, 16.0, 16.0], \"L\": [0.45, 0.6, 0.75], \"e\": [0.5, 0.5, 0.5], \"c/A3\": [0.09, 0.2, 0.2], \"D2\": [12.0, 12.0, 12.0], \"E2\": [12.0, 12.0, 12.0]}}\n```"
    #         table = _json_to_table(pred)
    #         # 模型默认认为“类型正确 & 完整”
    #         return table, True, True
    #     except Exception as e:
    #         print("INFO:模型预测失败页 {pageNumber}，已回退传统逻辑: {e}")
    #         #  fall through 到传统逻辑


    # save_table_image(pdfPath, pageNumber, TableCoordinate
    table = get_table(pdfPath, pageNumber, TableCoordinate)
    if (table is None) or (table == []) or (len(table) == 1):
        print("INFO:传统方法识别的表格有误:\n")
    table = table_checked(table)

    type_result = judge_if_package_table(table, packageType)
    integrity_result = judge_if_complete_table(table, packageType)

    return table, type_result, integrity_result

def extract_table(pdfPath, page_Number_List, Table_Coordinate_List, packageType):
    """
    传入表格坐标，获得表格信息（支持并行处理多页面）
    :param pdfPath: pdf路径
    :param page_Number_List: 存在表格页
    :param Table_Coordinate_List: 表格坐标
    :param packageType: 封装类型
    :return: 当前表格信息
    """
    print(f'文件路径：{pdfPath}\n', f'存在表格页：{page_Number_List}\n',
          f'表格对应坐标：{Table_Coordinate_List}\n', f'封装类型：{packageType}')
    Table = []
    Type = []
    Integrity = []

    try:
        # 准备并行任务参数
        tasks = [
            (pageNumber, TableCoordinate, pdfPath, packageType)
            for pageNumber, TableCoordinate in zip(page_Number_List, Table_Coordinate_List)
        ]
        # 如果只有一个页面，直接处理（避免线程开销）
        if len(tasks) == 1:
            table, type_result, integrity_result = _process_single_page(tasks[0])
            Table.append(table)
            Type.append(type_result)
            Integrity.append(integrity_result)
        else:
            # 多页面并行处理
            # 使用字典保持顺序
            results = {}

            with ThreadPoolExecutor(max_workers=min(3, len(tasks))) as executor:
                future_to_idx = {
                    executor.submit(_process_single_page, task): idx
                    for idx, task in enumerate(tasks)
                }

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        print(f"处理页面 {page_Number_List[idx]} 时出错: {e}")
                        results[idx] = ([], False, False)

            # 按原始顺序整理结果
            for idx in range(len(tasks)):
                table, type_result, integrity_result = results.get(idx, ([], False, False))
                Table.append(table)
                Type.append(type_result)
                Integrity.append(integrity_result)
                print("##")

    finally:
        # 清理 PDF 缓存
        PDFDocumentCache.close_doc(pdfPath)

    # 根据封装表是否完整进行合并
    table = table_Select(Table, Type, Integrity)
    table = table_checked(table)
    # 提取表内信息
    data = postProcess(table, packageType)

    return data

def save_table_image(pdfPath, pageNumber, tableCoordinate):
    """
    保存表格图片到结果目录
    :param pdfPath: PDF文件路径
    :param pageNumber: 页码
    :param tableCoordinate: 表格坐标 [x0, y0, x1, y1]
    """
    # 创建保存图片的目录
    result_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "Result", "Table_images")
    os.makedirs(result_dir, exist_ok=True)
    
    # 打开PDF并获取页面
    doc = fitz.open(pdfPath)
    page = doc.load_page(pageNumber - 1)  # 页码从0开始
    
    # 截取表格区域
    rect = fitz.Rect(tableCoordinate)
    mat = fitz.Matrix(2.0, 2.0)  # 放大2倍以提高清晰度
    pix = page.get_pixmap(matrix=mat, clip=rect)
    
    # 生成文件名
    filename = f"{Path(pdfPath).stem}_{pageNumber}.png"
    filepath = os.path.join(result_dir, filename)
    
    # 保存图片
    pix.save(filepath)
    pix = None
    doc.close()
    
    print(f"表格图片已保存: {filepath}")