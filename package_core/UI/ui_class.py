"""一些与主窗口进行交互的ui类，相关样式表"""
from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsRectItem, \
    QHBoxLayout, QFileDialog, QMessageBox, QTableWidgetItem, QDialog, QGridLayout, QSpacerItem, \
    QSizePolicy, QLabel, QProgressBar, QPushButton

# 第三方库 PySide6 6.5.3
from PySide6.QtCore import QPoint, Qt, Signal, QEvent, QMargins, Slot, QThread, QMutex, \
    QWaitCondition, QMutexLocker, QTimer
from PySide6.QtGui import QCloseEvent, QBrush, QColor, QCursor, QPen, QFont

# 外部文件
from package_core.PDF_Processed.PDF_Processed_main import PackageDetectionPipeline
from package_core.Segment.Package_pretreat import *
from package_core.Segment.Segment_function import adjust_table_coordinates,manage_json
from package_core.Table_Processed import Table_extract

import copy

Yes = QMessageBox.Yes
No = QMessageBox.No

PuShButton_initial_qss = """
                                                                QPushButton{
                                                background-color:
                                                qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,
                                                stop: 0 rgba(240,240,240,255),
                                                stop: 0.495 rgba(200,200,200,255),
                                                stop: 0.505 rgba(190,190,190,255),
                                                stop: 1 rgba(185,185,185,255));

                                                /*color: rgb(255, 255, 255);*/
                                                border: 0.5px groove gray;
                                                border-radius: 6px;
                                                color: rgb(0, 0, 0);
                                                /*font: 75 12pt ;*/
                                                padding: 0.5px;
                                                border-style: outset;
                                                }

                                                QPushButton:hover{
                                                background-color:
                                                qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,
                                                stop: 0 rgba(200,200, 200,230),
                                                stop: 0.495 rgba(100,100,100,230),
                                                stop: 0.505 rgba(100,100,100,230),
                                                stop: 1 rgba(150,150,150,230));

                                                color: rgb(255, 255, 255);
                                                border: 0.5px groove gray;
                                                border-radius: 6px;
                                                padding: 0.5px;
                                                border-style: outset;
                                                /*color: rgb(255, 255, 255);*/
                                                /*font: 75 12pt "微软雅黑";*/
                                                }

                                                QPushButton:pressed{
                                                background-color:
                                                qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,
                                                stop: 0 rgba(200,200,200,255),
                                                stop: 0.495 rgba(150,150,150,255),
                                                stop: 0.505 rgba(145,145,145,255),
                                                stop: 1 rgba(140,140,140,255));

                                                /*color: rgb(255, 255, 255);*/
                                                border: 0.5px rgb(212, 212, 212);
                                                border-radius: 6px;
                                                color: rgb(0, 0, 0);
                                                padding: 0.5px;
                                                border-style: inset;
                                                /*font: 75 12pt "微软雅黑";
                                                padding-left:2px;
                                                padding-top:1px;*/
                                                }

                                                QPushButton:disabled{
                                                background-color:
                                                qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,
                                                stop: 0 rgba(250,250,250,125),
                                                stop: 0.495 rgba(225,225,225,125),
                                                stop: 0.505 rgba(210,210,210,125),
                                                stop: 1 rgba(190,190,190,125));

                                                /*color: rgb(255, 255, 255);*/
                                                border: 0.5px groove gray;
                                                border-radius: 6px;
                                                color: rgb(0, 0, 0);
                                                /*font: 75 12pt ;*/
                                                padding: 0.5px;
                                                border-style: outset;
                                                }
                                                            """
PuShButton_img_Draw_qss = """
                                                   QPushButton{
                                   background-color:
                                   qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,
                                   stop: 0 rgba(240,240,240,255),
                                   stop: 0.495 rgba(200,200,200,255),
                                   stop: 0.505 rgba(190,190,190,255),
                                   stop: 1 rgba(185,185,185,255));

                                   /*color: rgb(255, 255, 255);*/
                                   border: 0.5px groove gray;
                                   border-radius: 6px;
                                   color: red;
                                   /*font: 75 12pt ;*/
                                   padding: 0.5px;
                                   border-style: outset;
                                   }

                                   QPushButton:hover{
                                   background-color:
                                   qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,
                                   stop: 0 rgba(200,200, 200,230),
                                   stop: 0.495 rgba(100,100,100,230),
                                   stop: 0.505 rgba(100,100,100,230),
                                   stop: 1 rgba(150,150,150,230));

                                   color: rgb(255, 255, 255);
                                   border: 0.5px groove gray;
                                   border-radius: 6px;
                                   padding: 0.5px;
                                   border-style: outset;
                                   /*color: rgb(255, 255, 255);*/
                                   /*font: 75 12pt "微软雅黑";*/
                                   }

                                   QPushButton:pressed{
                                   background-color:
                                   qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,
                                   stop: 0 rgba(200,200,200,255),
                                   stop: 0.495 rgba(150,150,150,255),
                                   stop: 0.505 rgba(145,145,145,255),
                                   stop: 1 rgba(140,140,140,255));

                                   /*color: rgb(255, 255, 255);*/
                                   border: 0.5px rgb(212, 212, 212);
                                   border-radius: 6px;
                                   color: rgb(0, 0, 0);
                                   padding: 0.5px;
                                   border-style: inset;
                                   /*font: 75 12pt "微软雅黑";
                                   padding-left:2px;
                                   padding-top:1px;*/
                                   }

                                   QPushButton:disabled{
                                   background-color:
                                   qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,
                                   stop: 0 rgba(250,250,250,125),
                                   stop: 0.495 rgba(225,225,225,125),
                                   stop: 0.505 rgba(210,210,210,125),
                                   stop: 1 rgba(190,190,190,125));

                                   /*color: rgb(255, 255, 255);*/
                                   border: 0.5px groove gray;
                                   border-radius: 6px;
                                   color: rgb(0, 0, 0);
                                   /*font: 75 12pt ;*/
                                   padding: 0.5px;
                                   border-style: outset;
                                   }
                                               """
PuShButton_list_Draw_qss = """
                                                                QPushButton{
                                                background-color:
                                                qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,
                                                stop: 0 rgba(240,240,240,255),
                                                stop: 0.495 rgba(200,200,200,255),
                                                stop: 0.505 rgba(190,190,190,255),
                                                stop: 1 rgba(185,185,185,255));

                                                /*color: rgb(255, 255, 255);*/
                                                border: 0.5px groove gray;
                                                border-radius: 6px;
                                                color: blue;
                                                /*font: 75 12pt ;*/
                                                padding: 0.5px;
                                                border-style: outset;
                                                }

                                                QPushButton:hover{
                                                background-color:
                                                qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,
                                                stop: 0 rgba(200,200, 200,230),
                                                stop: 0.495 rgba(100,100,100,230),
                                                stop: 0.505 rgba(100,100,100,230),
                                                stop: 1 rgba(150,150,150,230));

                                                color: rgb(255, 255, 255);
                                                border: 0.5px groove gray;
                                                border-radius: 6px;
                                                padding: 0.5px;
                                                border-style: outset;
                                                /*color: rgb(255, 255, 255);*/
                                                /*font: 75 12pt "微软雅黑";*/
                                                }

                                                QPushButton:pressed{
                                                background-color:
                                                qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,
                                                stop: 0 rgba(200,200,200,255),
                                                stop: 0.495 rgba(150,150,150,255),
                                                stop: 0.505 rgba(145,145,145,255),
                                                stop: 1 rgba(140,140,140,255));

                                                /*color: rgb(255, 255, 255);*/
                                                border: 0.5px rgb(212, 212, 212);
                                                border-radius: 6px;
                                                color: rgb(0, 0, 0);
                                                padding: 0.5px;
                                                border-style: inset;
                                                /*font: 75 12pt "微软雅黑";
                                                padding-left:2px;
                                                padding-top:1px;*/
                                                }

                                                QPushButton:disabled{
                                                background-color:
                                                qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,
                                                stop: 0 rgba(250,250,250,125),
                                                stop: 0.495 rgba(225,225,225,125),
                                                stop: 0.505 rgba(210,210,210,125),
                                                stop: 1 rgba(190,190,190,125));

                                                /*color: rgb(255, 255, 255);*/
                                                border: 0.5px groove gray;
                                                border-radius: 6px;
                                                color: rgb(0, 0, 0);
                                                /*font: 75 12pt ;*/
                                                padding: 0.5px;
                                                border-style: outset;
                                                }
                                                            """

Label_initial_qss = """QLabel{
                                                                        background-color: rgb(230, 230, 230);
                                                                        color: black;
                                                                    }"""
Label_Draw_Img_qss = """QLabel{
                                                   background-color: rgb(230, 230, 230);
                                                   color: red;
                                               }"""
Label_Draw_List_qss = """QLabel{
                                        background-color: rgb(230, 230, 230);
                                        color: blue;
                                    }"""


class EnquirePopUp(QMessageBox):
    """
        询问弹窗
    """
    def __init__(self, window, title, question):
        super(EnquirePopUp, self).__init__(window)
        self.setStyleSheet("font-size: 16px;")
        self.setText(question)
        self.setWindowTitle(title)
        self.setStandardButtons(Yes | No)
        self.button(Yes).setText('确定')
        self.button(No).setText('取消')
        self.button(No).setFocus()
        self.enquire_result = self.exec()


class MyProgressDialog(QDialog):
    def __init__(self, window, dialog_title, label_text, progress_value=None):
        super(MyProgressDialog, self).__init__(window)
        self.parent_window = window
        self.dialog_title = dialog_title
        self.label_text = label_text
        self.flag = progress_value
        self.progress_value = None
        self.progress_label_value = None
        self.timer = None           # 进度条定时器
        self.timer_label = None     # 标签定时器


        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)   # 设置窗口标志，移除关闭按钮

        self.layout()        # 自定义弹窗布局
        self.setup()     # 属性初始化
        self.show()

    def layout(self):
        """自定义弹窗布局"""
        """添加布局"""
        self.gridLayout = QGridLayout(self)
        self.gridLayout.setVerticalSpacing(4)
        self.gridLayout.setContentsMargins(8, 9, -1, -1)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalSpacer_4 = QSpacerItem(138, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_4)
        """添加标签控件"""
        self.label = QLabel(self)
        font = QFont()
        font.setPointSize(12)
        self.label.setFont(font)

        self.horizontalLayout.addWidget(self.label)

        self.horizontalSpacer_5 = QSpacerItem(138, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_5)

        self.horizontalLayout.setStretch(0, 5)
        self.horizontalLayout.setStretch(1, 1)
        self.horizontalLayout.setStretch(2, 5)

        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)

        self.horizontalLayout_2 = QHBoxLayout()

        self.horizontalLayout_2.setContentsMargins(-1, 16, -1, 16)
        self.horizontalSpacer_2 = QSpacerItem(68, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_2)
        """添加进度条控件"""
        self.progressBar = QProgressBar(self)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.progressBar.sizePolicy().hasHeightForWidth())
        self.progressBar.setSizePolicy(sizePolicy)
        self.progressBar.setCursor(QCursor(Qt.ArrowCursor))
        self.progressBar.setTextVisible(False)
        self.progressBar.setOrientation(Qt.Horizontal)
        self.progressBar.setInvertedAppearance(False)

        self.horizontalLayout_2.addWidget(self.progressBar)

        self.horizontalSpacer_3 = QSpacerItem(88, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_3)

        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 2)
        self.horizontalLayout_2.setStretch(2, 1)

        self.gridLayout.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalSpacer = QSpacerItem(268, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer)
        """添加按钮控件"""
        self.pushButton = QPushButton(self)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)

        self.horizontalLayout_3.addWidget(self.pushButton)

        self.horizontalLayout_3.setStretch(0, 6)
        self.horizontalLayout_3.setStretch(1, 1)

        self.gridLayout.addLayout(self.horizontalLayout_3, 2, 0, 1, 1)

        self.gridLayout.setRowStretch(0, 1)
        self.gridLayout.setRowStretch(1, 2)
        self.gridLayout.setRowStretch(2, 1)
        self.pushButton.setText('取消')

    def setup(self):
        self.setWindowTitle(self.dialog_title)    # 设置弹窗标题
        self.label.setText(self.label_text)           # 设置弹窗标签控件标题
        self.progress_value = 0
        self.progress_label_value = 0

        self.progressBar.setValue(self.progress_value)     # 进度条设置值

        self.timer = QTimer(self)
        if not self.flag:
            self.timer.timeout.connect(self.update_progress)
            self.timer.start(10)

        self.timer_label = QTimer(self)
        self.timer_label.timeout.connect(self.update_label_progress)
        self.timer_label.start(500)

    def update_progress(self):
        """滚动进度条"""
        self.progress_value += 1
        if (self.progress_value > 100):
            self.progress_value = 0
        self.progressBar.setValue(self.progress_value)

    def update_label_progress(self):
        """进度条动态标签设置"""
        self.progress_label_value += 1
        if (self.progress_label_value % 4 == 0):
            self.label.setText(self.label_text)
        elif (self.progress_label_value % 4 == 1):
            self.label.setText(self.label_text + '.')
        elif (self.progress_label_value % 4 == 2):
            self.label.setText(self.label_text + '..')
        else:
            self.label.setText(self.label_text + '...')


class MyGraphicsView(QGraphicsView):
    """
        画框
    """
    mouse_released = Signal(list)  # 设置结束的信号

    def __init__(self, parent=None):
        super(MyGraphicsView, self).__init__(parent)  # 继承绘图控件
        self.layer = QGraphicsScene(self)
        # 绘图控件设置场景
        self.setScene(self.layer)
        # 存储画框结果
        self.rect = [0 for _ in range(4)]
        # 存储显示矩形
        self.rect_item = None
        # 设置起始点
        self.start_point = QPoint()

    def enterEvent(self, event: QEvent):
        self.setCursor(QCursor(Qt.CrossCursor))
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent):
        self.setCursor(QCursor(Qt.ArrowCursor))
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.rect[0] = event.position().x()
            self.rect[1] = event.position().y()
            self.start_point = self.mapToScene(self.rect[0], self.rect[1])  # 视图到场景

            # 创建矩形项
            self.rect_item = QGraphicsRectItem()
            self.rect_item.setPen(QPen(QColor(128, 0, 255), 2))  # 设置边框颜色为紫色
            self.rect_item.setBrush(QColor(0, 255, 0, 0))  # 设置矩形框填充
            # 给场景添加矩形项
            self.layer.addItem( self.rect_item)

    def mouseMoveEvent(self, event):
        if self.rect_item and event.buttons() & Qt.LeftButton:
            # 更新矩形的位置随鼠标移动
            x = event.position().x()
            y = event.position().y()
            current_point = self.mapToScene(x, y)
            # 去除画框起始点的影响
            left = min(self.start_point.x(), current_point.x())
            top = min(self.start_point.y(), current_point.y())
            right = max(self.start_point.x(), current_point.x())
            bottom = max(self.start_point.y(), current_point.y())

            self.rect_item.setRect(left, top, right - left, bottom - top)

    def mouseReleaseEvent(self, event):
        if self.rect_item and event.button() == Qt.LeftButton:
            self.rect[2] = event.position().x()
            self.rect[3] = event.position().y()

            # 转换坐标系
            left = min(self.rect[0], self.rect[2])
            top = min(self.rect[1], self.rect[3])
            right = max(self.rect[0], self.rect[2])
            bottom = max(self.rect[1], self.rect[3])
            # 获取矩形
            self.rect = [left, top, right, bottom]
            # 释放信号
            self.mouse_released.emit(self.rect)
            # 移除矩形
            self.layer.removeItem(self.rect_item)


class DetectThread(QThread):
    """自动搜索按钮下的检测线程"""
    # 去水印 -> 筛选含封装图页面 -> DETR -> 判断并旋转封装图页面 -> 对封装类型视图类型矫正 -> Package视图与特征视图、各视图以及封装表匹配 -> end
    # 检测线程可随时杀死
    signal_end_page = Signal(int)     # 结束去水印，筛选封装图页面，放出页面数量信号
    signal_end = Signal(list)       # 自动搜索并形成规范的json数据
    signal_error = Signal(str)  # 用于发送错误信息字符串

    def __init__(self, window, pdf_path):
        super(DetectThread, self).__init__(window)
        self.window = window
        self.mutex = QMutex()            # 设置互斥锁
        self.cond = QWaitCondition()       # 线程同步机制
        self.pdf_path = pdf_path

        self.page_list = None          # 可能含封装图页码
        self.PreProcess = None             # 执行检测工作对象

    def run(self):
        try:
            # 创建管道实例
            pipeline = PackageDetectionPipeline(self.pdf_path)
            # --- 阶段一 ---
            page_list = pipeline.step1_preprocess_pages()
            self.page_list = page_list  # 保存以便主线程获取进度
            self.signal_end_page.emit(len(page_list) if self.page_list else 0)
            # 暂停等待主线程UI更新
            self.cond.wait(self.mutex)
            # --- 阶段二 ---
            # 按顺序执行管道的每一步
            raw_detr_results = pipeline.step2_run_detr_detection(page_list)
            modified_detr_results = pipeline.step3_match_keywords(raw_detr_results, page_list)
            _, final_data, have_page , _ = pipeline.step4_group_package_components(modified_detr_results)
            self.final_package_data = final_data if final_data is not None else []
            self.final_have_page = have_page if have_page is not None else []

            # 通过信号发送最终的、干净的数据
            self.signal_end.emit(self.final_package_data)  # 假设 signal_end = Signal(list)

        except Exception as e:
            # 最佳实践：通过信号报告错误
            error_message = f"自动搜索流程出现错误: {e}"
            self.signal_error.emit(error_message)
            print(e)

            # QMessageBox.critical(self.window, '自动搜索流程出现错误', str(e))

    def resume(self):
        """
            唤醒阻塞线程
        :return:
        """
        with QMutexLocker(self.mutex):
            self.cond.wakeOne()  # 唤醒一个阻塞线程


class RecoThread(QThread):
    """识别函数工作线程"""
    signal_end = Signal(int)
    def __init__(self, window, pdf_path,pdf_page_count,current_package,package,type_dict):
        super(RecoThread, self).__init__(window)
        self.window = window
        self.pdf_path = pdf_path
        self.current_page = pdf_page_count # current_page = PDF实际当前页-1
        self.current_package = current_package
        self.type_dict = type_dict
        self.package = package # 此为所有的封装信息
        self.result = None

    def run(self):
        """识别函数接口"""
        try:
            # 识别函数
            print(f"开始识别")
            Table_Coordinate_List = []
            page_Number_List = []

            pin_num_x_serial = None
            pin_num_y_serial = None
            pin_sum = None
            # 封装类型
            package_type = self.type_dict[self.current_page]
            if package_type == 'DFN_SON'or package_type == 'DFN':
                package_type = 'SON'
            manage_data = manage_json(self.current_package)
            package_process(self.current_page, manage_data[0])  # 分割流程
            if package_type == 'BGA':
                # 如果表格类型是BGA,运行数字提取BGA引脚数量
                pin_num_x_serial, pin_num_y_serial, loss_pin, loss_color = extract_BGA_pins()
            if package_type == 'QFP':
                pin_num_x_serial, pin_num_y_serial = extract_QFP_pins()
            if package_type == 'QFN':
                pin_num_x_serial, pin_num_y_serial = extract_QFN_pins()
            if package_type == 'SOP':
                pin_sum = extract_SOP_pins()
            if package_type == 'SON':
                pin_sum = extract_SON_pins()
            #判断是否走数字流程有两个条件，一个是当前封装信息self.current_package内无Form；另一个是有Form但不是封装Form，这个就是在识别表格的时候才显示。
            if self.current_package['part_content'] is not None:
                exists = any(part['part_name'] == 'Form' for part in self.current_package['part_content'])
                if exists:
                    # 表格提取
                    current_table = manage_data[1][self.current_page]
                    page_Number_List, Table_Coordinate_List = adjust_table_coordinates(self.current_page, current_table)
                else:
                    print('数字提取')
                    self.result = package_indentify(package_type, self.current_page)
                    if package_type == 'BGA':
                        self.result[2] = ['', '', pin_num_x_serial, ''] if pin_num_x_serial is not None else self.result[2]
                        self.result[3] = ['', '', pin_num_y_serial, ''] if pin_num_y_serial is not None else self.result[3]
                        self.result[10][2] = str(loss_color)
                        self.result[10][1] = str(loss_pin)
            elif self.current_package['part_content'] is None and self.current_package['type'] == 'list':  # 说明是自动框表
                #目前只考虑识别当前框选的表，暂不考虑识别多个框选的表
                Table_Coordinate_List = [[],self.current_package['rect'],[]]
                page_Number_List = [self.current_page,self.current_page+1,self.current_page+2]
            elif self.current_package['part_content'] is None and self.current_package['type'] == 'img':  # 说明是自动框图
                #框选图流程存在争议
                pass
            if len(page_Number_List) != 0 and len(Table_Coordinate_List) != 0:
                try:
                    #表格内容提取
                    data = Table_extract.extract_table(self.pdf_path, page_Number_List, Table_Coordinate_List, package_type)
                    #后续操作只考虑了BGA表格类型
                    if package_type == 'QFP':

                        if not data:
                            # 走数字提取流程
                            print("-----表格数据提取为空-----")
                            self.result = package_indentify(package_type, self.current_page)
                        else:
                            self.result = data
                        self.result[0] = ['', '', pin_num_x_serial, ''] if pin_num_x_serial is not None else self.result[0]
                        self.result[1] = ['', '', pin_num_y_serial, ''] if pin_num_y_serial is not None else self.result[1]
                    elif package_type == 'BGA':
                        result = data[0:11]
                        result[10][2] = str(loss_color)
                        result[10][1] = str(loss_pin)
                        result[2] = ['', '', pin_num_x_serial, ''] if pin_num_x_serial is not None else self.result[2]
                        result[3] = ['', '', pin_num_y_serial, ''] if pin_num_y_serial is not None else self.result[3]
                        self.result = result
                    elif package_type == 'SON':
                        result = data[0:14]
                        result[1] = ['', '', pin_sum, ''] if pin_sum is not None else self.result[1]
                        self.result = result
                    elif package_type == 'SOP':
                        result = data[0:12]
                        result[1] = ['', '', pin_sum, ''] if pin_sum is not None else self.result[1]
                        self.result = result
                    elif package_type == 'QFN':
                        result = data
                        result[2] = ['', '', pin_num_x_serial, ''] if pin_num_x_serial is not None else self.result[2]
                        result[3] = ['', '', pin_num_y_serial, ''] if pin_num_y_serial is not None else self.result[3]
                        self.result = result
                except Exception as e:
                    print(e)
                    # 走数字提取流程
                    self.result = package_indentify(package_type, self.current_page)
                    if package_type == 'BGA':
                        self.result[2] = ['', '', pin_num_x_serial, ''] if pin_num_x_serial is not None else self.result[2]
                        self.result[3] = ['', '', pin_num_y_serial, ''] if pin_num_y_serial is not None else self.result[3]
                        self.result[10][2] = str(loss_color)
                        self.result[10][1] = str(loss_pin)
                    elif package_type == 'SON':
                        self.result[1] = ['', '', pin_sum, ''] if pin_sum is not None else self.result[1]
            self.signal_end.emit(1)
        except Exception as e:
            QMessageBox.critical(self.window, '识别出现错误', str(e))
        finally:
            # 结束对资源的访问
            pass




