# package_core/UI/AI/chat_dialog.py

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QFrame
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont, QTextCursor
from package_core.UI.AI.ai_agent import HuaQiuAIEngine


class AIWorkerThread(QThread):
    """AI 调用工作线程"""
    stream_received = Signal(str)  # 流式片段
    response_ready = Signal(str)  # 完整响应
    error_occurred = Signal(str)  # 错误信号

    def __init__(self, engine, question, context="", image_path=None):
        super().__init__()
        self.engine = engine
        self.question = question
        self.context = context
        self.image_path = image_path

    def run(self):
        try:
            full_response = ""
            # 调用 backend 的 chat 函数 (流式)
            for chunk_text in self.engine.chat(self.question, self.context, self.image_path, stream=True):
                if chunk_text:
                    full_response += chunk_text
                    self.stream_received.emit(chunk_text)

            self.response_ready.emit(full_response)
        except Exception as e:
            # 捕获所有线程内的错误发送给主线程
            self.error_occurred.emit(str(e))


class ChatDialog(QDialog):
    """AI 对话框"""

    def __init__(self, parent=None, context="", image_path=None):
        super().__init__(parent)
        self.context = context
        self.image_path = image_path
        self.ai_engine = HuaQiuAIEngine()
        self.worker_thread = None
        self.current_ai_message_cursor = None
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle("AI 助手")
        self.resize(600, 500)
        main_layout = QVBoxLayout(self)

        # 1. 聊天记录显示区 (优化样式)
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Microsoft YaHei", 10))
        self.chat_display.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                background-color: #f9f9f9;
            }
        """)
        main_layout.addWidget(self.chat_display)

        # 2. 输入区
        input_frame = QFrame()
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(0, 0, 0, 0)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("请输入您的问题...")
        self.input_edit.setStyleSheet("QLineEdit { padding: 8px; border: 1px solid #ccc; border-radius: 4px; }")
        self.input_edit.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_edit)

        self.send_button = QPushButton("发送")
        self.send_button.setCursor(Qt.PointingHandCursor)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #0063b1; }
        """)
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)

        main_layout.addWidget(input_frame)

    def send_message(self):
        question = self.input_edit.text().strip()
        if not question: return

        # 显示用户消息
        self.append_message("你", question, "#0078d7", is_user=True)
        self.input_edit.clear()
        self.input_edit.setEnabled(False)
        self.send_button.setEnabled(False)

        # 准备 AI 回复容器
        self.start_ai_message()

        # 启动线程 (注意：image_path 这里可能为 None，在 main.py 里控制)
        self.worker_thread = AIWorkerThread(self.ai_engine, question, self.context, self.image_path)
        self.worker_thread.stream_received.connect(self.update_ai_message)
        self.worker_thread.error_occurred.connect(self.on_error)
        self.worker_thread.finished.connect(self.on_finished)
        self.worker_thread.start()

    def start_ai_message(self):
        """初始化 AI 消息块"""
        self.append_header("AI 助手", "#28a745")
        self.current_ai_message_cursor = self.chat_display.textCursor()
        self.current_ai_message_cursor.movePosition(QTextCursor.End)
        # 插入临时占位符
        self.chat_display.insertHtml('<div style="color: gray;">...</div>')
        self.chat_display.ensureCursorVisible()

    def update_ai_message(self, text):
        """流式追加内容"""
        if self.current_ai_message_cursor:
            cursor = self.chat_display.textCursor()
            cursor.movePosition(QTextCursor.End)
            # 处理换行，防止 HTML 格式错乱
            formatted = text.replace('\n', '<br>')
            cursor.insertHtml(f'<span>{formatted}</span>')
            self.chat_display.setTextCursor(cursor)
            self.chat_display.ensureCursorVisible()

    def on_error(self, err):
        # 【修改】报错时只在后台打印，不显示在界面上
        print(f"--- [后台错误] AI线程报错: {err} ---")
        # 如果需要，可以在这里让界面恢复输入状态，但不弹窗
        self.on_finished()

    def on_finished(self):
        self.input_edit.setEnabled(True)
        self.send_button.setEnabled(True)
        self.input_edit.setFocus()
        self.current_ai_message_cursor = None
        self.chat_display.append("")  # 增加空行间距

    def append_header(self, sender, color):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertHtml(f'<div style="margin-top: 15px; font-weight: bold; color: {color};">{sender}:</div>')
        self.chat_display.setTextCursor(cursor)

    def append_message(self, sender, message, color, is_user=False):
        self.append_header(sender, color)
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        msg_html = message.replace('\n', '<br>')

        # 【修改】去除 <hr>，优化背景色
        bg_style = "background-color: #e6f7ff; padding: 8px; border-radius: 5px;" if is_user else "padding: 5px;"
        cursor.insertHtml(f'<div style="margin-top:5px; {bg_style}">{msg_html}</div>')

        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()


# 便捷入口
def show_chat_dialog(parent=None, context="", image_path=None):
    dialog = ChatDialog(parent, context, image_path)
    dialog.exec()