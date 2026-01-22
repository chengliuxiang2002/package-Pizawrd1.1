import fitz  # PyMuPDF
import dashscope
from dashscope.api_entities.dashscope_response import MultiModalConversationResponse
from http import HTTPStatus
import os
import json
import re
from typing import List, Dict, Any, Union

# 【安全警告】
# 不要在生产环境中硬编码 API Key。
# 这里硬编码是为了让调试脚本能直接运行。
DASHSCOPE_API_KEY = "sk-7439b0ec552f4cedbbb7528b5b373bf6"


def get_table_img(pdfPath: str, pageNumber: int, TableCoordinate):
    """
    使用 PyMuPDF (fitz) 从 PDF 截取表格图片并保存为临时文件。

    :param pdfPath: PDF文件路径
    :param pageNumber: PDF的页码 (从1开始)
    :param TableCoordinate: 表格坐标 (x0, y0, x1, y1) - 接受列表或元组
    :return: 临时图片的文件路径 (例如 "file:///path/to/temp_table.png")，如果失败则返回 None
    """
    try:
        doc = fitz.open(pdfPath)
        # PyMuPDF 页码从 0 开始，所以需要-1
        page = doc.load_page(pageNumber - 1)

        # 定义裁剪区域 (fitz.Rect 可以接受列表或元组)
        rect = fitz.Rect(TableCoordinate)

        # 提高图片分辨率 (DPI)
        zoom = 3.125
        mat = fitz.Matrix(zoom, zoom)

        # 获取裁剪区域的像素图
        pix = page.get_pixmap(matrix=mat, clip=rect)

        output_path = os.path.abspath("temp_table_for_ai.png")
        pix.save(output_path)

        doc.close()

        # 返回千问 API 需要的本地文件格式
        return f"file://{output_path}"

    except Exception as e:
        print(f"截取表格图片时出错: {e}")
        if 'doc' in locals() and doc:
            doc.close()
        return None


def parse_json_from_response(model_output):
    """
    从大模型的文本回复中提取 JSON 数据块。
    (已加固，可以处理 None 输入)
    """
    # 加固：处理 model_output 为 None 的情况
    if not model_output:
        print("parse_json_from_response 接收到空输入。")
        return None

    # 尝试匹配 ```json ... ``` 代码块
    match = re.search(r"```json\s*([\s\S]+?)\s*```", model_output, re.DOTALL)

    json_str = ""
    if match:
        json_str = match.group(1)
    else:
        # 尝试查找第一个 '{' 或 '['
        start_index = -1
        first_brace = model_output.find('{')
        first_bracket = model_output.find('[')

        if first_brace == -1:
            start_index = first_bracket
        elif first_bracket == -1:
            start_index = first_brace
        else:
            start_index = min(first_brace, first_bracket)

        if start_index != -1:
            json_str = model_output[start_index:]
        else:
            print("未能在回复中找到 JSON 数据。")
            return None
    try:
        json_str = json_str.strip()
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败: {e}")
        print(f"原始字符串: {json_str[:200]}...")
        return None


def convert_json_to_list_of_lists(data: Union[Dict[str, Any], List[Any]]):
    """
    将AI模型返回的JSON对象（多种可能格式）转换为统一的 "list of lists" 格式。
    【已更新】: 自动移除表头末尾的点 (e.g., "REF." -> "REF")
    """
    if not data:
        return None

    # Case 1: 已经是 "list of lists" 格式
    if isinstance(data, list) and all(isinstance(row, list) for row in data):
        return data  # 已经是目标格式

    # Case 2: 是 "list of dicts" 格式 (e.g., [{"col1": "a"}, {"col1": "b"}])
    if isinstance(data, list) and len(data) > 0 and all(isinstance(row, dict) for row in data):
        try:
            # 【更新】清理表头，移除末尾的点
            original_headers = list(data[0].keys())
            cleaned_headers = [h.rstrip('.') for h in original_headers]

            rows = [list(row.values()) for row in data]
            return [cleaned_headers] + rows
        except Exception as e:
            print(f"转换 'list of dicts' 失败: {e}")
            return None

    # Case 3: 是 "dict with headers/rows" 格式 (e.g., {"headers": [...], "rows": [...]})
    if isinstance(data, dict):
        headers = data.get("headers")
        rows = data.get("rows")
        if isinstance(headers, list) and isinstance(rows, list):
            # 确保 rows 里的内容也是 list
            if all(isinstance(row, list) for row in rows):
                # 【更新】清理表头，移除末尾的点
                cleaned_headers = [h.rstrip('.') for h in headers]
                return [cleaned_headers] + rows
            else:
                print("字典格式正确，但 'rows' 内部包含非列表元素。")
                return None

    # Case 4: 只是一个 "list of strings" 或其他不支持的格式
    print(f"无法将JSON转换为'list of lists'格式。数据类型: {type(data)}")
    return None


def ai_rec(pdfPath: str, pageNumber: int, TableCoordinate):
    """
    调用千问Qwen-VL-Max模型识别表格，并返回 "list of lists" 格式。
    """
    # 1. 先把表格图片截取下来
    image_path = get_table_img(pdfPath, pageNumber, TableCoordinate)

    if not image_path:
        print("无法生成表格图片，AI识别终止。")
        return None

    # 2. 检查环境变量（将在 main 中设置）
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("错误：DASHSCOPE_API_KEY 环境变量未设置。")
        if os.path.exists("temp_table_for_ai.png"):
            os.remove("temp_table_for_ai.png")
        return None

    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_path},
                {"text": "你是一个专业的表格识别助手。请严格按照这张图片中的表格内容，将其转换为结构化的JSON格式。不要添加任何说明文字，只返回JSON。"}
            ]
        }
    ]

    try:
        # 调用 API
        response: MultiModalConversationResponse = dashscope.MultiModalConversation.call(
            model='qwen-vl-max',
            messages=messages
        )

        table_data = None
        if response.status_code == HTTPStatus.OK:
            model_output = response.output.choices[0].message.content

            if not model_output:
                print("API 调用成功，但模型返回了空内容。")
                try:
                    finish_reason = response.output.choices[0].get('finish_reason', 'unknown')
                    print(f"Finish Reason: {finish_reason}")
                    if finish_reason == 'stop' and not model_output:
                        print("提示：这通常意味着内容被安全策略过滤。")
                except Exception:
                    pass
                return None  # 提前终止

            # 从模型回复中解析 JSON
            table_json_data = parse_json_from_response(model_output)

            if table_json_data:
                # 4. 【关键】将JSON转换为 list of lists 格式 (此函数现在会清理表头)
                table_data_list = convert_json_to_list_of_lists(table_json_data)

                if table_data_list:
                    return table_data_list  # 转换成功
                else:
                    print("成功解析JSON，但无法将其转换为 'list of lists' 格式。")
                    print(f"模型返回的原始JSON结构: {json.dumps(table_json_data, indent=2, ensure_ascii=False)}")
                    return None
            else:
                print("成功调用API，但未能解析返回的JSON。")
                print(f"模型原始输出: {model_output}")

        else:
            print(f"API 调用失败: {response.code} - {response.message}")
            return None

    except Exception as e:
        print(f"AI 识别或数据处理时出错: {e}")
        return None

    finally:
        # 5. 清理临时图片文件
        temp_file_path = image_path.replace("file://", "") if image_path else ""
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            # print(f"已删除临时文件: {temp_file_path}")


# ==================================================================
#                       调试用的 Main 函数
# ==================================================================
if __name__ == "__main__":

    if DASHSCOPE_API_KEY:
        os.environ['DASHSCOPE_API_KEY'] = DASHSCOPE_API_KEY

    # --- 1. 使用你提供的PDF路径 ---
    pdf_file = r"E:\admin\Desktop\[非密]SCB18T2Gxx0AF.pdf"

    # --- 2. 使用你提供的页码 ---
    page_num = 46

    # --- 3. 使用你提供的坐标 ---
    coords = [319, 539, 485, 738]

    # --- 调试开始 ---

    if not os.path.exists(pdf_file):
        print(f"错误：测试文件 {pdf_file} 未找到。")
    else:
        print(f"正在读取文件: {pdf_file}")
        print(f"目标页码: {page_num}, 坐标: {coords}")
        print("---------------------------------")
        print("正在调用 AI 识别表格，请稍候...")

        # 调用 ai_rec，它现在会返回 list[list] 或 None
        table_list = ai_rec(pdf_file, page_num, coords)

        print("---------------------------------")
        if table_list:
            print("AI 识别成功！")
            print("识别结果 (List of Lists 格式, 表头已清理):")

            # table_list 保证是 list of lists，直接遍历打印
            for row in table_list:
                print(row)

        else:
            print("AI 识别失败，或无法转换为列表格式。")