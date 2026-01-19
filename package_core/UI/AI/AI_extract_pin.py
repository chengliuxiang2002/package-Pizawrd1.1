import os
import json
import re

from package_core.PackageExtract.yolox_onnx_py.model_paths import result_path
from package_core.UI.AI.ai_agent_pin import HuaQiuAIEngine

BOTTOM_PATH = result_path("Package_view","page","bottom.jpg")

# === 权威的 JEDEC 列表 (Python 端持有) ===
# JEDEC 列表保持不变，用于处理 type="letter" 的情况
JEDEC_ROWS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'T', 'U', 'V', 'W', 'Y',
    'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AJ', 'AK', 'AL', 'AM', 'AN', 'AP', 'AR', 'AT', 'AU', 'AV', 'AW',
    'AY',
    'BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG', 'BH', 'BJ', 'BK', 'BL', 'BM', 'BN', 'BP', 'BR', 'BT', 'BU', 'BV', 'BW',
    'BY'
]


def calculate_rows_from_letters(start_char, end_char):
    if not start_char or not end_char: return 0
    s, e = start_char.upper().strip(), end_char.upper().strip()
    try:
        return JEDEC_ROWS.index(e) - JEDEC_ROWS.index(s) + 1
    except:
        return 0


def AI_extract_pin():
    engine = HuaQiuAIEngine()
    question = "请提取该BGA封装底视图的行列PIN数"

    print(f"📸 分析图片: {BOTTOM_PATH}")
    full_text = ""
    for chunk in engine.chat(question=question, image_path=BOTTOM_PATH):
        print(chunk, end="", flush=True)
        full_text += chunk

    print("\n\n" + "-" * 30)

    try:
        json_match = re.search(r"\{.*\}", full_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))

            # === 解析行 ===
            row_type = data.get("row_type")
            final_rows = 0

            if row_type == "letter":
                print("ℹ️ 行方向：识别到字母丝印")
                final_rows = calculate_rows_from_letters(data.get("row_start"), data.get("row_end"))
            elif row_type == "number":
                print("ℹ️ 行方向：识别到数字丝印")
                final_rows = int(data.get("row_max_num", 0))
            elif row_type == "count":
                print("ℹ️ 行方向：无丝印，AI 已手动计数")
                final_rows = int(data.get("row_max_num", 0))

            # === 解析列 ===
            col_type = data.get("col_type")
            final_cols = 0

            if col_type == "letter":
                print("ℹ️ 列方向：识别到字母丝印")
                final_cols = calculate_rows_from_letters("A", data.get("col_end"))  # 假设列字母从A开始
            elif col_type == "number":
                print("ℹ️ 列方向：识别到数字丝印")
                final_cols = int(data.get("col_max_num", 0))
            elif col_type == "count":
                print("ℹ️ 列方向：无丝印，AI 已手动计数")
                final_cols = int(data.get("col_max_num", 0))

            print("-" * 30)
            print(f"✅ 最终结果: {final_rows} 行 x {final_cols} 列")
            return final_cols,final_rows

    except Exception as e:
        print(f"❌ 解析错误: {e}")
        return None,None


if __name__ == "__main__":
    AI_extract_pin()