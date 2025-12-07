"""
AI 对话框组件
提供与 AI 助手交互的图形界面
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QLabel, QSizePolicy,
    QScrollArea, QWidget, QFrame
)
from PySide6.QtCore import Qt, Signal, QThread, QSize
from PySide6.QtGui import QFont, QColor, QTextCursor

from package_core.UI.AI.ai_agent import HuaQiuAIEngine


class AIWorkerThread(QThread):
    """AI 调用工作线程，避免阻塞 UI"""
    response_ready = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, engine, question, context=""):
        super().__init__()
        self.engine = engine
        self.question = question
        self.context = context

    def run(self):
        try:
            response = self.engine.chat(self.question, self.context)
            self.response_ready.emit(response)
        except Exception as e:
            self.error_occurred.emit(str(e))


class ChatDialog(QDialog):
    """AI 对话框"""

    def __init__(self, parent=None, context=""):
        super().__init__(parent)
        self.context = context  # 可选的上下文信息
        self.ai_engine = HuaQiuAIEngine()
        self.worker_thread = None

        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        """设置 UI"""
        self.setWindowTitle("AI 助手")
        self.setMinimumSize(500, 400)
        self.resize(600, 500)

        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 标题
        title_label = QLabel("AI 电子工程师助手")
        title_label.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #333; padding: 5px;")
        main_layout.addWidget(title_label)

        # 聊天记录区域
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Microsoft YaHei", 10))
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        self.chat_display.setPlaceholderText("对话内容将显示在这里...")
        main_layout.addWidget(self.chat_display, 1)

        # 输入区域
        input_frame = QFrame()
        input_frame.setStyleSheet("""
            QFrame {
                background-color: #fff;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
        """)
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(5, 5, 5, 5)
        input_layout.setSpacing(5)

        # 输入框
        self.input_edit = QLineEdit()
        self.input_edit.setFont(QFont("Microsoft YaHei", 11))
        self.input_edit.setPlaceholderText("输入你的问题...")
        self.input_edit.setStyleSheet("""
            QLineEdit {
                border: none;
                padding: 8px;
                background: transparent;
            }
        """)
        input_layout.addWidget(self.input_edit, 1)

        # 发送按钮
        self.send_button = QPushButton("发送")
        self.send_button.setFont(QFont("Microsoft YaHei", 10))
        self.send_button.setFixedSize(70, 35)
        self.send_button.setCursor(Qt.PointingHandCursor)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        input_layout.addWidget(self.send_button)

        main_layout.addWidget(input_frame)

        # 状态标签
        self.status_label = QLabel("")
        self.status_label.setFont(QFont("Microsoft YaHei", 9))
        self.status_label.setStyleSheet("color: #666; padding: 2px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

        # 底部按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        # 清空按钮
        self.clear_button = QPushButton("清空对话")
        self.clear_button.setFont(QFont("Microsoft YaHei", 9))
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ddd;
                border-radius: 3px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        button_layout.addWidget(self.clear_button)

        # 关闭按钮
        self.close_button = QPushButton("关闭")
        self.close_button.setFont(QFont("Microsoft YaHei", 9))
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ddd;
                border-radius: 3px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        button_layout.addWidget(self.close_button)

        main_layout.addLayout(button_layout)

    def setup_connections(self):
        """设置信号连接"""
        self.send_button.clicked.connect(self.send_message)
        self.input_edit.returnPressed.connect(self.send_message)
        self.clear_button.clicked.connect(self.clear_chat)
        self.close_button.clicked.connect(self.close)

    def send_message(self):
        """发送消息"""
        question = self.input_edit.text().strip()
        if not question:
            return

        # 显示用户消息
        self.append_message("你", question, "#2196F3")
        self.input_edit.clear()

        # 禁用输入
        self.set_input_enabled(False)
        self.status_label.setText("AI 正在思考...")

        # 启动工作线程
        self.worker_thread = AIWorkerThread(self.ai_engine, question, self.context)
        self.worker_thread.response_ready.connect(self.on_response_ready)
        self.worker_thread.error_occurred.connect(self.on_error_occurred)
        self.worker_thread.finished.connect(self.on_thread_finished)
        self.worker_thread.start()

    def on_response_ready(self, response):
        """AI 响应就绪"""
        self.append_message("AI 助手", response, "#4CAF50")

    def on_error_occurred(self, error):
        """发生错误"""
        self.append_message("系统", f"发生错误: {error}", "#f44336")

    def on_thread_finished(self):
        """线程结束"""
        self.set_input_enabled(True)
        self.status_label.setText("")
        self.worker_thread = None

    def set_input_enabled(self, enabled):
        """设置输入控件状态"""
        self.input_edit.setEnabled(enabled)
        self.send_button.setEnabled(enabled)

    def append_message(self, sender, message, color):
        """添加消息到聊天记录"""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)

        # 发送者
        cursor.insertHtml(f'<p style="margin: 5px 0;"><b style="color: {color};">{sender}:</b></p>')

        # 消息内容
        # 处理换行和空格
        formatted_message = message.replace('\n', '<br>').replace('  ', '&nbsp;&nbsp;')
        cursor.insertHtml(f'<p style="margin: 5px 0 15px 10px; color: #333;">{formatted_message}</p>')

        # 分隔线
        cursor.insertHtml('<hr style="border: none; border-top: 1px solid #eee; margin: 10px 0;">')

        # 滚动到底部
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def clear_chat(self):
        """清空聊天记录"""
        self.chat_display.clear()

    def set_context(self, context):
        """设置上下文信息"""
        self.context = context


def show_chat_dialog(parent=None, context=""):
    """显示 AI 对话框的便捷函数"""
    dialog = ChatDialog(parent, context)
    dialog.exec()
    return dialog


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dialog = ChatDialog()
    dialog.show()
    sys.exit(app.exec())
