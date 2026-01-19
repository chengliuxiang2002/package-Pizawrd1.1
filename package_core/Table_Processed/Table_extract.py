import base64
import io
import json
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# F3.表格内容解析与判断 F4.表格规范化流程
from package_core.Table_Processed.Table_function.GetTable import *

# ================= 配置区 (适配本地 Qwen2.5-VL) =================
# 默认配置适配测试代码中的环境
MODEL_HOST = os.getenv('TABLE_MODEL_HOST', '10.86.163.88')
MODEL_PORT = int(os.getenv('TABLE_MODEL_PORT', 6006))
MODEL_PATH = os.getenv('TABLE_MODEL_PATH', '/v1/chat/completions')  # OpenAI 兼容接口
MODEL_NAME = os.getenv('TABLE_MODEL_NAME', 'qwen2_5_vl')
MODEL_TIMEOUT = int(os.getenv('TABLE_MODEL_TIMEOUT', 180))  # 大模型推理较慢，超时时间设长

# 核心 Prompt
PROMPT = (
    "按以下规则识别表格中的全部参数并输出 JSON：\n"
    "1. 参数过滤：仅提取封装尺寸（A, A1, A2, A3, b, c, D, D1, E, E2, e, L），其他字符均忽略\n"
    "2. 数据格式：每个参数永远输出 [最小值, 标称值, 最大值] 三元列表；"
    "若某参数仅提供两个数值，默认将其识别为最小值和最大值，标称值用 '-' 占位；缺失项用 '-' 占位\n"
    "3.读取逻辑：直接按表格自然排版（从上到下、从左到右）识别横向或纵向表\n"
    "4. 单位转换：若出现双单位，只保留毫米（mm）数值\n"
    "5. 自动计算：若标称值（中间项）为 '-' 但最小值和最大值均为数字，"
    "则需计算：标称值 = (最小值 + 最大值) / 2，保留原小数位数\n"
    "6. 直接给 JSON，不要解释。"
)

IMAGE_MAX_SIDE = 1000  # 图片最大边长限制

# =============================================================
def _model_alive():
    """
    强制返回 True，跳过复杂的 Socket 探测
    (因为 HTTP 接口可能不响应简单的 TCP 握手或鉴权导致探测失败)
    """
    return True

def preprocess_image(pil_image):
    """
    压缩并缩放图片，解决 10054 和显存溢出问题
    :param pil_image: PIL.Image 对象
    :return: base64 string
    """
    try:
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # 等比例缩放
        w, h = pil_image.size
        if max(w, h) > IMAGE_MAX_SIDE:
            scale = IMAGE_MAX_SIDE / max(w, h)
            pil_image = pil_image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=80)  # 转 JPEG 减小体积
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"图片预处理失败: {e}")
        return None

def crop_pdf_page(pdf_path: str, page_number: int, coords: list) -> Image.Image:
    """把 PDF 指定页的指定坐标区域裁成 PIL.Image"""
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number - 1]
        x0, y0, x1, y1 = coords
        # 保持较高 DPI 抓取，后续由 preprocess_image 统一压缩
        mat = fitz.Matrix(3.0, 3.0)
        pix = page.get_pixmap(matrix=mat, clip=fitz.Rect(x0, y0, x1, y1))
        img_bytes = pix.tobytes("png")
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    finally:
        doc.close()

def _model_predict_local(image):
    """
    调用本地大模型接口 (OpenAI Chat Format)
    """
    # 1. 预处理图片 (压缩/缩放/转Base64)
    b64_data = preprocess_image(image)
    if not b64_data:
        raise ValueError("Image preprocessing failed")

    url = f"http://{MODEL_HOST}:{MODEL_PORT}{MODEL_PATH}"

    # 2. 构造 OpenAI 格式 Payload
    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}}
            ]
        }],
        "temperature": 0  # 降低随机性
    }

    # 设置 Connection: close 防止旧连接干扰
    headers = {"Connection": "close", "Content-Type": "application/json"}

    # 3. 发送请求
    rsp = requests.post(url, json=payload, headers=headers, timeout=MODEL_TIMEOUT)
    rsp.raise_for_status()
    return rsp.json()

def _parse_model_output_to_table(model_response):
    """
    解析本地大模型返回的 JSON 格式
    针对格式: {"image": "...", "status": "success", "output": "```json...```"}
    """
    try:
        if not model_response:
            return None

        # 1. 提取 raw_output 字符串
        # 优先直接获取 'output' 字段 (对应你提供的本地模型格式)
        raw_output = model_response.get("output")

        # 如果没有 output，尝试兼容 OpenAI 格式 (以防未来切换)
        if not raw_output and "choices" in model_response:
            try:
                raw_output = model_response["choices"][0]["message"]["content"]
            except (IndexError, KeyError, TypeError):
                pass

        if not raw_output:
            # print("警告: 未找到有效的 output 字段")
            return None

        # 2. 提取 Markdown 代码块中的 JSON
        # 匹配 ```json ... ``` 或 ``` ... ``` 或 纯文本
        match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw_output, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            json_str = raw_output

        # 3. 解析 JSON 字符串
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            print(f"JSON 解析失败: {json_str[:50]}...")
            return None

        # 4. 提取 parameters (数据清洗)
        parameters = {}

        # 情况 A (最常见): {"parameters": {"A": [...]}}
        if isinstance(data, dict) and "parameters" in data:
            parameters = data["parameters"]
        # 情况 B: 直接是字典 {"A": [...]}
        elif isinstance(data, dict):
            parameters = data
        # 情况 C: 偶尔模型发疯返回 list [{"A":...}]
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    parameters.update(item)

        if not parameters:
            return None

        # 5. 转换为 List of Lists (统一输出结构)
        # 目标: [['A', '0.8', '0.9', '1.0'], ['A1', '0.025', '-', '0.075']]
        table_list = []

        # 确保 parameters 是字典，可以安全遍历
        if isinstance(parameters, dict):
            for key, values in parameters.items():
                # 容错：如果 values 不是列表 (例如单值)，转为列表
                if not isinstance(values, list):
                    values = [values]

                # 构造一行: [Key, Val1, Val2, Val3]
                # 必须全部转为字符串，因为传统算法处理的全是字符串
                row = [str(key)]
                for v in values:
                    # 处理 None 或 空值
                    if v is None:
                        row.append("-")
                    else:
                        row.append(str(v))

                table_list.append(row)

        return table_list

    except Exception as e:
        print(f"解析过程发生未知错误: {e}")
        return None

def _process_single_page(args):
    """处理单个页面 (AI 优先 -> 传统兜底)"""
    pageNumber, TableCoordinate, pdfPath, packageType = args
    if not TableCoordinate:
        return [], False, False

    table = None

    # --- 1. 大模型分支 ---
    if _model_alive():
        try:
            print(f"INFO: [Page {pageNumber}] 调用本地 Qwen2.5-VL...")
            img = crop_pdf_page(pdfPath, pageNumber, TableCoordinate)
            resp_json = _model_predict_local(img)
            # resp_json = {"image": "C141373_数字信号处理器(DSP-DSC)_DSPIC33EP32MC204-I-PT_规格书_MICROCHIP(美国微芯)单片机(MCUMPUSOC)规格书_505.png", "status": "success", "output": "```json\n{\"parameters\": {\"A\": [0.8, 0.9, 1.0], \"A1\": [0.025, \"-\", 0.075], \"b/B\": [0.2, 0.25, 0.3], \"D\": [6.0, 6.0, 6.0], \"E\": [6.0, 6.0, 6.0], \"e\": [0.5, 0.5, 0.5], \"L\": [0.2, 0.25, 0.3], \"e\": [0.5, 0.5, 0.5], \"D2\": [4.4, 4.55, 4.7], \"E2\": [4.4, 4.55, 4.7]}}\n```"}
            table = _parse_model_output_to_table(resp_json)
            print(f"INFO: [Page {pageNumber}] AI 识别结果: {table}")

            if table and len(table) > 0:
                print(f"INFO: [Page {pageNumber}] AI 识别成功")
                type_result = judge_if_package_table(table, packageType)
                integrity_result = judge_if_complete_table(table, packageType)
                return table, type_result, integrity_result
            else:
                print(f"WARN: [Page {pageNumber}] AI 识别结果为空，回退传统算法")
                table = None
        except Exception as e:
            print(f"WARN: [Page {pageNumber}] AI 调用异常: {e}，回退传统算法")
            table = None

    # --- 2. 传统算法兜底 ---
    if table is None:
        # print(f"INFO: [Page {pageNumber}] 使用传统算法")
        table = get_table(pdfPath, pageNumber, TableCoordinate)
        print(table)

    # --- 3. 后处理 ---
    if not table:
        return [], False, False

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