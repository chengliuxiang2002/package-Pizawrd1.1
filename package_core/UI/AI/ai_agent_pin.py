import os
import base64
import traceback
from openai import OpenAI

# ================= 配置区域 =================
MY_API_KEY = "sk-ZvdkLPXktUgVlDuiyR64iZJr4ZBXAcWmNZnASvRqmGevMFeZ"
BASE_URL = "https://www.chataiapi.com/v1"
# MODEL_NAME = "gemini-2.5-flash-image"
MODEL_NAME = "gemini-3-pro-preview"

class HuaQiuAIEngine:
    def __init__(self):
        print("=== [DEBUG] AI引擎初始化 (全能识别版：识字+数球) ===", flush=True)
        self.api_key = os.getenv("GEMINI_API_KEY", MY_API_KEY)
        self.search_data_dir = os.path.join(os.getcwd(), "Result", "Package_extract", "data")

        try:
            self.client = OpenAI(api_key=self.api_key, base_url=BASE_URL)
        except Exception as e:
            print(f"=== [ERROR] 客户端初始化失败: {e}", flush=True)
            self.client = None

        self.base_system_prompt = "你是一名资深的电子元器件封装工程师。"

    def _check_search_data(self):
        if not os.path.exists(self.search_data_dir): return []
        try:
            return [os.path.join(self.search_data_dir, f) for f in os.listdir(self.search_data_dir) if
                    f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        except:
            return []

    def _encode_image(self, image_path):
        if not os.path.exists(image_path): return None
        with open(image_path, "rb") as image_file: return base64.b64encode(image_file.read()).decode('utf-8')

    def chat(self, question, context="", image_path=None, stream=False):
        print(f"=== [DEBUG] Chat请求: {question}", flush=True)

        if not self.client:
            yield "系统错误：API 未配置。"
            return

        try:
            # 1. 确定图片
            target_imgs = []
            if image_path and os.path.exists(image_path):
                target_imgs = [image_path]
            else:
                keywords = ["提取", "识别", "视图", "PIN", "行列"]
                if any(k in question for k in keywords):
                    target_imgs = self._check_search_data()

            user_content = []
            is_pin_mode = any(k in question.upper() for k in ["PIN", "行列", "X,Y", "XY"])

            if target_imgs and is_pin_mode:
                print("=== [DEBUG] 进入视觉提取模式...", flush=True)

                # === 核心修改：增加了“无丝印直接数球”的逻辑 ===
                instruction = """
                【任务指令】请分析这张 BGA 底视图的矩阵排列，确定行数和列数。

                请严格按照以下优先级进行判断（不要编造）：

                **第一步：检测丝印标识 (Priority High)**
                - 观察图片边缘是否有字母（A,B...）或数字（1,5,10...）。
                - 如果有，请读取**起始标识**和**结束标识**。

                **第二步：纯视觉计数 (Priority Low - 仅当无标识时使用)**
                - 如果图片边缘**没有任何文字标记**（干净的焊球矩阵），请直接**数**每一行和每一列有多少个锡球。

                **第三步：输出结论**
                请返回以下 JSON 格式（纯文本）：

                {
                    // 行方向识别结果
                    "row_type": "letter",   // 选项: "letter"(字母标识), "number"(数字标识), "count"(无标识直接计数)
                    "row_start": "A",       // 仅当 letter 时填写
                    "row_end": "N",         // 仅当 letter 时填写
                    "row_max_num": 14,      // 当 type 为 "number" 或 "count" 时，填写具体的数值

                    // 列方向识别结果
                    "col_type": "number",   // 选项: "letter", "number", "count"
                    "col_end": "",          // 仅当 letter 时填写
                    "col_max_num": 24       // 当 type 为 "number" 或 "count" 时，填写具体的数值
                }
                """
                user_content.append({"type": "text", "text": instruction})

                # 加载图片
                for img_p in target_imgs[:1]:
                    b64 = self._encode_image(img_p)
                    if b64:
                        user_content.append(
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

            else:
                user_content.append({"type": "text", "text": question})
                if context:
                    user_content.append({"type": "text", "text": f"上下文：{context}"})

            messages = [
                {"role": "system", "content": self.base_system_prompt},
                {"role": "user", "content": user_content}
            ]

            print("=== [DEBUG] 发送 API 请求...", flush=True)
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                stream=True
            )

            for chunk in response:
                if not chunk.choices: continue
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    yield delta.content

        except Exception as e:
            print(f"=== [后台错误] {e}", flush=True)
            traceback.print_exc()
            raise e