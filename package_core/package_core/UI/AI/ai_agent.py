# package_core/UI/huaqiu/ai_agent.py

import os
import base64
import traceback
from openai import OpenAI

# ================= 配置区域 =================
# 1. 填入你的 sk- 开头的 Key
MY_API_KEY = "sk-ZvdkLPXktUgVlDuiyR64iZJr4ZBXAcWmNZnASvRqmGevMFeZ"  # 替换为你真实的 Key
# 2. 填入中转服务的接口地址 (Base URL)
# 格式通常是: https://域名/v1
BASE_URL = "https://www.chataiapi.com/v1"
# 3. 模型名称
# MODEL_NAME = "gemini-3-pro-preview"
MODEL_NAME = "gemini-2.5-flash-image"
# =======================================================
class HuaQiuAIEngine:
    def __init__(self):
        print("=== [DEBUG] AI引擎初始化 ===", flush=True)
        self.api_key = os.getenv("GEMINI_API_KEY", MY_API_KEY)
        self.search_data_dir = os.path.join(os.getcwd(), "Result", "Package_extract", "data")

        try:
            self.client = OpenAI(api_key=self.api_key, base_url=BASE_URL)
        except Exception as e:
            print(f"=== [ERROR] 客户端初始化失败: {e}", flush=True)
            self.client = None

        self.system_prompt = """
        你是一名工业电子元器件专家。
        1. 普通对话：请用专业、简洁的中文回答。
        2. 提取任务：当用户请求"提取视图"、"识别数据"时，**必须且只能**返回 JSON 格式数据。
        """

    def _check_search_data(self):
        """检测自动搜索的图片"""
        if not os.path.exists(self.search_data_dir):
            return []
        try:
            return [os.path.join(self.search_data_dir, f) for f in os.listdir(self.search_data_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        except:
            return []

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def chat(self, question, context="", image_path=None, stream=False):
        """核心对话函数"""
        print(f"=== [DEBUG] Chat请求: {question}", flush=True)

        if not self.client:
            yield "系统错误：API 未配置。"
            return

        try:
            # 1. 意图识别
            keywords = ["提取", "识别", "视图数据", "数据信息"]
            is_extract = any(k in question for k in keywords)

            messages = [{"role": "system", "content": self.system_prompt}]
            user_content = []

            # 2. 处理提取逻辑
            if is_extract:
                print("=== [DEBUG] 进入提取模式...", flush=True)
                search_imgs = self._check_search_data()

                if not search_imgs:
                    yield "⚠️ **检测到当前暂无自动搜索视图数据。**\n\n请先在主界面点击 **【自动搜索】** 按钮，生成视图文件后再进行提取。"
                    return

                # 构造 JSON 指令
                json_instruction = """
                【任务指令】请根据提供的视图图片提取封装参数，并**仅返回**以下 JSON 格式（纯文本，不要Markdown代码块）：
                {
                    "is_extraction": true,
                    "package_width": <数值>,
                    "package_length": <数值>,
                    "pitch": <数值>,
                    "ball_diameter": <数值>,
                    "matrix_rows": <整数>,
                    "matrix_cols": <整数>,
                    "summary": "简短中文总结..."
                }
                """
                user_content.append({"type": "text", "text": json_instruction})
                user_content.append({"type": "text", "text": f"用户指令：{question}\n请分析以下视图文件："})

                # 加载自动搜索的图片 (限制3张)
                for img_p in search_imgs[:3]:
                    try:
                        b64 = self._encode_image(img_p)
                        fname = os.path.basename(img_p)
                        user_content.append({"type": "text", "text": f"文件名: {fname}"})
                        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                    except:
                        pass

            else:
                # 普通对话
                if context:
                    user_content.append({"type": "text", "text": f"【上下文】\n{context}"})
                user_content.append({"type": "text", "text": question})
                # 此处不再处理 main.py 传来的 image_path，因为我们已经将其设为 None

            messages.append({"role": "user", "content": user_content})

            # 3. 发送请求
            print("=== [DEBUG] 发送 API 请求...", flush=True)
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1 if is_extract else 0.7,
                stream=True
            )

            # 4. 流式返回
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            print(f"=== [后台错误] {e}", flush=True)
            # 这里抛出异常以便 chat_dialog 捕获并打印到控制台，而不是显示在界面上
            # 或者在这里直接 yield 一个友好的错误提示
            # yield "抱歉，AI 服务暂时不可用。"
            raise e