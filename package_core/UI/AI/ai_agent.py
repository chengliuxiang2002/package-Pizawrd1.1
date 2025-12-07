# package_core/UI/huaqiu/ai_agent.py

import os
import google.generativeai as genai


class HuaQiuAIEngine:
    def __init__(self):
        # 配置 Gemini API Key
        # 建议将其设置为环境变量，或者为了测试直接填入这里
        # 获取 Key 地址: https://aistudio.google.com/app/apikey
        self.api_key = os.getenv("GEMINI_API_KEY", "sk-YmgTP0OR0GOS9RJRGHUm9aHyTJ1Z6lTWSZGH8O7bf9BqJTNk")

        if not self.api_key or self.api_key == "在此处填入你的_GEMINI_API_KEY":
            print("警告: 未配置 Gemini API Key，AI 功能将无法正常工作。")

        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            print(f"Gemini 配置失败: {e}")

        # 设置系统提示词 (System Prompt)
        self.system_prompt = """
        角色设定：你是一名工业领域的电子元器件大师，也是一位经验丰富的硬件工程师。

        任务：
        1. 回答用户关于电子元器件封装（如BGA, QFN, SOP等）、PCB设计、焊接工艺等方面的问题。
        2. 如果用户提供了当前元器件的上下文信息（如引脚数、尺寸、间距等），请基于这些数据进行分析或建议。
        3. 回答风格需要专业、严谨、简洁，并尽可能提供数据支持。

        请用中文回答所有问题。
        """

    def chat(self, question, context=""):
        """
        与 AI 进行对话
        :param question: 用户的问题
        :param context: 上下文信息（例如当前选中的封装参数）
        :return: AI 的回答文本
        """
        if not self.api_key:
            return "系统提示：未配置 API Key。请在 ai_agent.py 中填入您的 Google Gemini API Key。"

        try:
            # 构建包含系统设定、上下文和用户问题的完整 Prompt
            full_prompt = f"{self.system_prompt}\n"

            if context:
                full_prompt += f"\n【当前元器件上下文数据】：\n{context}\n"

            full_prompt += f"\n【用户问题】：{question}"

            # 调用模型生成回答
            response = self.model.generate_content(full_prompt)
            return response.text

        except Exception as e:
            return f"AI 连接失败: {str(e)}。请检查网络连接或 API Key 是否正确。"