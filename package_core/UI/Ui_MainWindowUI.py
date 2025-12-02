# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'Ui_MainWindow.ui'
##
## Created by: Qt User Interface Compiler version 6.5.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtPdfWidgets import QPdfView
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMainWindow, QPushButton, QSizePolicy, QSpacerItem,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(917, 596)
        MainWindow.setStyleSheet(u"background-color: rgb(245, 245, 245);")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_3 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(-1, 4, 9, -1)
        self.frame_func = QFrame(self.centralwidget)
        self.frame_func.setObjectName(u"frame_func")
        self.frame_func.setStyleSheet(u"background-color: rgb(240, 240, 240);")
        self.frame_func.setFrameShape(QFrame.StyledPanel)
        self.frame_func.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_13 = QHBoxLayout(self.frame_func)
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.horizontalLayout_13.setContentsMargins(1, 1, 1, 1)
        self.groupBox = QGroupBox(self.frame_func)
        self.groupBox.setObjectName(u"groupBox")
        font = QFont()
        font.setPointSize(10)
        self.groupBox.setFont(font)
        self.groupBox.setStyleSheet(u"background-color: rgb(240, 240, 240);\n"
"border: 1px solid;")
        self.verticalLayout = QVBoxLayout(self.groupBox)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 10, 0, 0)
        self.frame_button_pdf = QFrame(self.groupBox)
        self.frame_button_pdf.setObjectName(u"frame_button_pdf")
        self.frame_button_pdf.setStyleSheet(u"border: 0px;\n"
"background-color: rgb(240, 240, 240);")
        self.frame_button_pdf.setFrameShape(QFrame.StyledPanel)
        self.frame_button_pdf.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_12 = QHBoxLayout(self.frame_button_pdf)
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.horizontalLayout_12.setContentsMargins(-1, 3, -1, 3)
        self.frame_view_button = QFrame(self.frame_button_pdf)
        self.frame_view_button.setObjectName(u"frame_view_button")
        self.frame_view_button.setStyleSheet(u"background-color: rgb(240, 240, 240);\n"
"border: 0px;")
        self.frame_view_button.setFrameShape(QFrame.StyledPanel)
        self.frame_view_button.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.frame_view_button)
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(-1, 5, -1, 5)
        self.pushButton_open = QPushButton(self.frame_view_button)
        self.pushButton_open.setObjectName(u"pushButton_open")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_open.sizePolicy().hasHeightForWidth())
        self.pushButton_open.setSizePolicy(sizePolicy)
        self.pushButton_open.setMaximumSize(QSize(120, 16777215))
        self.pushButton_open.setStyleSheet(u"QPushButton{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(240,240,240,255),\n"
"stop: 0.495 rgba(200,200,200,255),\n"
"stop: 0.505 rgba(190,190,190,255),\n"
"stop: 1 rgba(185,185,185,255));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px groove gray;\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"/*font: 75 12pt ;*/\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"}\n"
" \n"
"QPushButton:hover{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(200,200, 200,230),\n"
"stop: 0.495 rgba(100,100,100,230),\n"
"stop: 0.505 rgba(100,100,100,230),\n"
"stop: 1 rgba(150,150,150,230));\n"
" \n"
"color: rgb(255, 255, 255);\n"
"border: 0.5px groove gray;\n"
"border-radius: 6px;\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"/*color: rgb(255, 255, 255);*/\n"
"/*font: 75 12pt \"\u5fae\u8f6f\u96c5\u9ed1\";*/\n"
"}\n"
" \n"
"QPushButton:pressed{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:"
                        "1,\n"
"stop: 0 rgba(200,200,200,255),\n"
"stop: 0.495 rgba(150,150,150,255),\n"
"stop: 0.505 rgba(145,145,145,255),\n"
"stop: 1 rgba(140,140,140,255));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px rgb(212, 212, 212);\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"padding: 0.5px;\n"
"border-style: inset;\n"
"/*font: 75 12pt \"\u5fae\u8f6f\u96c5\u9ed1\";\n"
"padding-left:2px;\n"
"padding-top:1px;*/\n"
"}\n"
"\n"
"QPushButton:disabled{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(250,250,250,125),\n"
"stop: 0.495 rgba(225,225,225,125),\n"
"stop: 0.505 rgba(210,210,210,125),\n"
"stop: 1 rgba(190,190,190,125));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px groove gray;\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"/*font: 75 12pt ;*/\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"}")

        self.horizontalLayout_2.addWidget(self.pushButton_open)

        self.pushButton_detect = QPushButton(self.frame_view_button)
        self.pushButton_detect.setObjectName(u"pushButton_detect")
        sizePolicy.setHeightForWidth(self.pushButton_detect.sizePolicy().hasHeightForWidth())
        self.pushButton_detect.setSizePolicy(sizePolicy)
        self.pushButton_detect.setMaximumSize(QSize(120, 16777215))
        self.pushButton_detect.setStyleSheet(u"QPushButton{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(240,240,240,255),\n"
"stop: 0.495 rgba(200,200,200,255),\n"
"stop: 0.505 rgba(190,190,190,255),\n"
"stop: 1 rgba(185,185,185,255));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px groove gray;\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"/*font: 75 12pt ;*/\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"}\n"
" \n"
"QPushButton:hover{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(200,200, 200,230),\n"
"stop: 0.495 rgba(100,100,100,230),\n"
"stop: 0.505 rgba(100,100,100,230),\n"
"stop: 1 rgba(150,150,150,230));\n"
" \n"
"color: rgb(255, 255, 255);\n"
"border: 0.5px groove gray;\n"
"border-radius: 6px;\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"/*color: rgb(255, 255, 255);*/\n"
"/*font: 75 12pt \"\u5fae\u8f6f\u96c5\u9ed1\";*/\n"
"}\n"
" \n"
"QPushButton:pressed{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:"
                        "1,\n"
"stop: 0 rgba(200,200,200,255),\n"
"stop: 0.495 rgba(150,150,150,255),\n"
"stop: 0.505 rgba(145,145,145,255),\n"
"stop: 1 rgba(140,140,140,255));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px rgb(212, 212, 212);\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"padding: 0.5px;\n"
"border-style: inset;\n"
"/*font: 75 12pt \"\u5fae\u8f6f\u96c5\u9ed1\";\n"
"padding-left:2px;\n"
"padding-top:1px;*/\n"
"}\n"
"\n"
"QPushButton:disabled{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(250,250,250,125),\n"
"stop: 0.495 rgba(225,225,225,125),\n"
"stop: 0.505 rgba(210,210,210,125),\n"
"stop: 1 rgba(190,190,190,125));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px groove gray;\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"/*font: 75 12pt ;*/\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"}")

        self.horizontalLayout_2.addWidget(self.pushButton_detect)

        self.pushButton_pre = QPushButton(self.frame_view_button)
        self.pushButton_pre.setObjectName(u"pushButton_pre")
        sizePolicy.setHeightForWidth(self.pushButton_pre.sizePolicy().hasHeightForWidth())
        self.pushButton_pre.setSizePolicy(sizePolicy)
        self.pushButton_pre.setMaximumSize(QSize(120, 16777215))
        self.pushButton_pre.setStyleSheet(u"QPushButton{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(240,240,240,255),\n"
"stop: 0.495 rgba(200,200,200,255),\n"
"stop: 0.505 rgba(190,190,190,255),\n"
"stop: 1 rgba(185,185,185,255));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px groove gray;\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"/*font: 75 12pt ;*/\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"}\n"
" \n"
"QPushButton:hover{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(200,200, 200,230),\n"
"stop: 0.495 rgba(100,100,100,230),\n"
"stop: 0.505 rgba(100,100,100,230),\n"
"stop: 1 rgba(150,150,150,230));\n"
" \n"
"color: rgb(255, 255, 255);\n"
"border: 0.5px groove gray;\n"
"border-radius: 6px;\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"/*color: rgb(255, 255, 255);*/\n"
"/*font: 75 12pt \"\u5fae\u8f6f\u96c5\u9ed1\";*/\n"
"}\n"
" \n"
"QPushButton:pressed{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:"
                        "1,\n"
"stop: 0 rgba(200,200,200,255),\n"
"stop: 0.495 rgba(150,150,150,255),\n"
"stop: 0.505 rgba(145,145,145,255),\n"
"stop: 1 rgba(140,140,140,255));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px rgb(212, 212, 212);\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"padding: 0.5px;\n"
"border-style: inset;\n"
"/*font: 75 12pt \"\u5fae\u8f6f\u96c5\u9ed1\";\n"
"padding-left:2px;\n"
"padding-top:1px;*/\n"
"}\n"
"\n"
"QPushButton:disabled{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(250,250,250,125),\n"
"stop: 0.495 rgba(225,225,225,125),\n"
"stop: 0.505 rgba(210,210,210,125),\n"
"stop: 1 rgba(190,190,190,125));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px groove gray;\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"/*font: 75 12pt ;*/\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"}")

        self.horizontalLayout_2.addWidget(self.pushButton_pre)

        self.pushButton_next = QPushButton(self.frame_view_button)
        self.pushButton_next.setObjectName(u"pushButton_next")
        sizePolicy.setHeightForWidth(self.pushButton_next.sizePolicy().hasHeightForWidth())
        self.pushButton_next.setSizePolicy(sizePolicy)
        self.pushButton_next.setMaximumSize(QSize(120, 16777215))
        self.pushButton_next.setSizeIncrement(QSize(0, 0))
        self.pushButton_next.setStyleSheet(u"QPushButton{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(240,240,240,255),\n"
"stop: 0.495 rgba(200,200,200,255),\n"
"stop: 0.505 rgba(190,190,190,255),\n"
"stop: 1 rgba(185,185,185,255));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px groove gray;\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"/*font: 75 12pt ;*/\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"}\n"
" \n"
"QPushButton:hover{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(200,200, 200,230),\n"
"stop: 0.495 rgba(100,100,100,230),\n"
"stop: 0.505 rgba(100,100,100,230),\n"
"stop: 1 rgba(150,150,150,230));\n"
" \n"
"color: rgb(255, 255, 255);\n"
"border: 0.5px groove gray;\n"
"border-radius: 6px;\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"/*color: rgb(255, 255, 255);*/\n"
"/*font: 75 12pt \"\u5fae\u8f6f\u96c5\u9ed1\";*/\n"
"}\n"
" \n"
"QPushButton:pressed{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:"
                        "1,\n"
"stop: 0 rgba(200,200,200,255),\n"
"stop: 0.495 rgba(150,150,150,255),\n"
"stop: 0.505 rgba(145,145,145,255),\n"
"stop: 1 rgba(140,140,140,255));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px rgb(212, 212, 212);\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"padding: 0.5px;\n"
"border-style: inset;\n"
"/*font: 75 12pt \"\u5fae\u8f6f\u96c5\u9ed1\";\n"
"padding-left:2px;\n"
"padding-top:1px;*/\n"
"}\n"
"\n"
"QPushButton:disabled{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(250,250,250,125),\n"
"stop: 0.495 rgba(225,225,225,125),\n"
"stop: 0.505 rgba(210,210,210,125),\n"
"stop: 1 rgba(190,190,190,125));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px groove gray;\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"/*font: 75 12pt ;*/\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"}")

        self.horizontalLayout_2.addWidget(self.pushButton_next)

        self.pushButton_draw_img = QPushButton(self.frame_view_button)
        self.pushButton_draw_img.setObjectName(u"pushButton_draw_img")
        sizePolicy.setHeightForWidth(self.pushButton_draw_img.sizePolicy().hasHeightForWidth())
        self.pushButton_draw_img.setSizePolicy(sizePolicy)
        self.pushButton_draw_img.setMaximumSize(QSize(120, 16777215))
        self.pushButton_draw_img.setStyleSheet(u"QPushButton{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(240,240,240,255),\n"
"stop: 0.495 rgba(200,200,200,255),\n"
"stop: 0.505 rgba(190,190,190,255),\n"
"stop: 1 rgba(185,185,185,255));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px groove gray;\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"/*font: 75 12pt ;*/\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"}\n"
" \n"
"QPushButton:hover{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(200,200, 200,230),\n"
"stop: 0.495 rgba(100,100,100,230),\n"
"stop: 0.505 rgba(100,100,100,230),\n"
"stop: 1 rgba(150,150,150,230));\n"
" \n"
"color: rgb(255, 255, 255);\n"
"border: 0.5px groove gray;\n"
"border-radius: 6px;\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"/*color: rgb(255, 255, 255);*/\n"
"/*font: 75 12pt \"\u5fae\u8f6f\u96c5\u9ed1\";*/\n"
"}\n"
" \n"
"QPushButton:pressed{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:"
                        "1,\n"
"stop: 0 rgba(200,200,200,255),\n"
"stop: 0.495 rgba(150,150,150,255),\n"
"stop: 0.505 rgba(145,145,145,255),\n"
"stop: 1 rgba(140,140,140,255));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px rgb(212, 212, 212);\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"padding: 0.5px;\n"
"border-style: inset;\n"
"/*font: 75 12pt \"\u5fae\u8f6f\u96c5\u9ed1\";\n"
"padding-left:2px;\n"
"padding-top:1px;*/\n"
"}\n"
"\n"
"QPushButton:disabled{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(250,250,250,125),\n"
"stop: 0.495 rgba(225,225,225,125),\n"
"stop: 0.505 rgba(210,210,210,125),\n"
"stop: 1 rgba(190,190,190,125));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px groove gray;\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"/*font: 75 12pt ;*/\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"}")

        self.horizontalLayout_2.addWidget(self.pushButton_draw_img)

        self.pushButton_draw_list = QPushButton(self.frame_view_button)
        self.pushButton_draw_list.setObjectName(u"pushButton_draw_list")
        sizePolicy.setHeightForWidth(self.pushButton_draw_list.sizePolicy().hasHeightForWidth())
        self.pushButton_draw_list.setSizePolicy(sizePolicy)
        self.pushButton_draw_list.setMaximumSize(QSize(120, 16777215))
        self.pushButton_draw_list.setStyleSheet(u"QPushButton{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(240,240,240,255),\n"
"stop: 0.495 rgba(200,200,200,255),\n"
"stop: 0.505 rgba(190,190,190,255),\n"
"stop: 1 rgba(185,185,185,255));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px groove gray;\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"/*font: 75 12pt ;*/\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"}\n"
" \n"
"QPushButton:hover{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(200,200, 200,230),\n"
"stop: 0.495 rgba(100,100,100,230),\n"
"stop: 0.505 rgba(100,100,100,230),\n"
"stop: 1 rgba(150,150,150,230));\n"
" \n"
"color: rgb(255, 255, 255);\n"
"border: 0.5px groove gray;\n"
"border-radius: 6px;\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"/*color: rgb(255, 255, 255);*/\n"
"/*font: 75 12pt \"\u5fae\u8f6f\u96c5\u9ed1\";*/\n"
"}\n"
" \n"
"QPushButton:pressed{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:"
                        "1,\n"
"stop: 0 rgba(200,200,200,255),\n"
"stop: 0.495 rgba(150,150,150,255),\n"
"stop: 0.505 rgba(145,145,145,255),\n"
"stop: 1 rgba(140,140,140,255));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px rgb(212, 212, 212);\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"padding: 0.5px;\n"
"border-style: inset;\n"
"/*font: 75 12pt \"\u5fae\u8f6f\u96c5\u9ed1\";\n"
"padding-left:2px;\n"
"padding-top:1px;*/\n"
"}\n"
"\n"
"QPushButton:disabled{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(250,250,250,125),\n"
"stop: 0.495 rgba(225,225,225,125),\n"
"stop: 0.505 rgba(210,210,210,125),\n"
"stop: 1 rgba(190,190,190,125));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px groove gray;\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"/*font: 75 12pt ;*/\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"}")

        self.horizontalLayout_2.addWidget(self.pushButton_draw_list)

        self.pushButton_reco = QPushButton(self.frame_view_button)
        self.pushButton_reco.setObjectName(u"pushButton_reco")
        sizePolicy.setHeightForWidth(self.pushButton_reco.sizePolicy().hasHeightForWidth())
        self.pushButton_reco.setSizePolicy(sizePolicy)
        self.pushButton_reco.setMaximumSize(QSize(120, 16777215))
        self.pushButton_reco.setStyleSheet(u"QPushButton{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(240,240,240,255),\n"
"stop: 0.495 rgba(200,200,200,255),\n"
"stop: 0.505 rgba(190,190,190,255),\n"
"stop: 1 rgba(185,185,185,255));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px groove gray;\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"/*font: 75 12pt ;*/\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"}\n"
" \n"
"QPushButton:hover{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(200,200, 200,230),\n"
"stop: 0.495 rgba(100,100,100,230),\n"
"stop: 0.505 rgba(100,100,100,230),\n"
"stop: 1 rgba(150,150,150,230));\n"
" \n"
"color: rgb(255, 255, 255);\n"
"border: 0.5px groove gray;\n"
"border-radius: 6px;\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"/*color: rgb(255, 255, 255);*/\n"
"/*font: 75 12pt \"\u5fae\u8f6f\u96c5\u9ed1\";*/\n"
"}\n"
" \n"
"QPushButton:pressed{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:"
                        "1,\n"
"stop: 0 rgba(200,200,200,255),\n"
"stop: 0.495 rgba(150,150,150,255),\n"
"stop: 0.505 rgba(145,145,145,255),\n"
"stop: 1 rgba(140,140,140,255));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px rgb(212, 212, 212);\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"padding: 0.5px;\n"
"border-style: inset;\n"
"/*font: 75 12pt \"\u5fae\u8f6f\u96c5\u9ed1\";\n"
"padding-left:2px;\n"
"padding-top:1px;*/\n"
"}\n"
"\n"
"QPushButton:disabled{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(250,250,250,125),\n"
"stop: 0.495 rgba(225,225,225,125),\n"
"stop: 0.505 rgba(210,210,210,125),\n"
"stop: 1 rgba(190,190,190,125));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px groove gray;\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"/*font: 75 12pt ;*/\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"}")

        self.horizontalLayout_2.addWidget(self.pushButton_reco)

        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 1)
        self.horizontalLayout_2.setStretch(2, 1)
        self.horizontalLayout_2.setStretch(3, 1)
        self.horizontalLayout_2.setStretch(4, 1)
        self.horizontalLayout_2.setStretch(5, 1)

        self.horizontalLayout_12.addWidget(self.frame_view_button)

        self.horizontalSpacer_5 = QSpacerItem(3, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_12.addItem(self.horizontalSpacer_5)

        self.frame_pdf_change = QFrame(self.frame_button_pdf)
        self.frame_pdf_change.setObjectName(u"frame_pdf_change")
        self.frame_pdf_change.setStyleSheet(u"background-color: rgb(240, 240, 240);")
        self.frame_pdf_change.setFrameShape(QFrame.StyledPanel)
        self.frame_pdf_change.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_11 = QHBoxLayout(self.frame_pdf_change)
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.horizontalLayout_11.setContentsMargins(-1, 4, -1, 4)
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.lineEdit_page = QLineEdit(self.frame_pdf_change)
        self.lineEdit_page.setObjectName(u"lineEdit_page")
        sizePolicy.setHeightForWidth(self.lineEdit_page.sizePolicy().hasHeightForWidth())
        self.lineEdit_page.setSizePolicy(sizePolicy)
        font1 = QFont()
        font1.setPointSize(13)
        self.lineEdit_page.setFont(font1)
        self.lineEdit_page.setStyleSheet(u"background-color: rgb(255, 255, 255);\n"
"border: 0.5px solid gray;")
        self.lineEdit_page.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_5.addWidget(self.lineEdit_page)

        self.label_total_page = QLabel(self.frame_pdf_change)
        self.label_total_page.setObjectName(u"label_total_page")

        self.horizontalLayout_5.addWidget(self.label_total_page)

        self.horizontalLayout_5.setStretch(0, 1)
        self.horizontalLayout_5.setStretch(1, 1)

        self.horizontalLayout_11.addLayout(self.horizontalLayout_5)

        self.horizontalSpacer_4 = QSpacerItem(11, 18, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_11.addItem(self.horizontalSpacer_4)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setSpacing(1)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.pushButton_zoomin = QPushButton(self.frame_pdf_change)
        self.pushButton_zoomin.setObjectName(u"pushButton_zoomin")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.pushButton_zoomin.sizePolicy().hasHeightForWidth())
        self.pushButton_zoomin.setSizePolicy(sizePolicy1)
        self.pushButton_zoomin.setMaximumSize(QSize(70, 16777215))
        font2 = QFont()
        font2.setPointSize(12)
        self.pushButton_zoomin.setFont(font2)
        self.pushButton_zoomin.setStyleSheet(u"QPushButton{\n"
"background-color: rgb(230, 230, 230);\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 2px groove gray;\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"/*font: 75 12pt ;*/\n"
"/*padding: 4px;*/\n"
"/*border-style: outset;*/\n"
"}\n"
"\n"
" \n"
"QPushButton:pressed{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(200,200,200,255),\n"
"stop: 0.495 rgba(150,150,150,255),\n"
"stop: 0.505 rgba(145,145,145,255),\n"
"stop: 1 rgba(140,140,140,255));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 2px rgb(212, 212, 212);\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"padding: 4px;\n"
"border-style: inset;\n"
"/*font: 75 12pt \"\u5fae\u8f6f\u96c5\u9ed1\";\n"
"padding-left:2px;\n"
"padding-top:1px;*/\n"
"}")

        self.horizontalLayout_4.addWidget(self.pushButton_zoomin)

        self.pushButton_zoomout = QPushButton(self.frame_pdf_change)
        self.pushButton_zoomout.setObjectName(u"pushButton_zoomout")
        sizePolicy1.setHeightForWidth(self.pushButton_zoomout.sizePolicy().hasHeightForWidth())
        self.pushButton_zoomout.setSizePolicy(sizePolicy1)
        self.pushButton_zoomout.setMaximumSize(QSize(70, 16777215))
        self.pushButton_zoomout.setFont(font2)
        self.pushButton_zoomout.setStyleSheet(u"QPushButton{\n"
"background-color: rgb(230, 230, 230);\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 2px groove gray;\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"/*font: 75 12pt ;*/\n"
"/*padding: 4px;*/\n"
"/*border-style: outset;*/\n"
"}\n"
"\n"
" \n"
"QPushButton:pressed{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(200,200,200,255),\n"
"stop: 0.495 rgba(150,150,150,255),\n"
"stop: 0.505 rgba(145,145,145,255),\n"
"stop: 1 rgba(140,140,140,255));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 2px rgb(212, 212, 212);\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"padding: 4px;\n"
"border-style: inset;\n"
"/*font: 75 12pt \"\u5fae\u8f6f\u96c5\u9ed1\";\n"
"padding-left:2px;\n"
"padding-top:1px;*/\n"
"}")

        self.horizontalLayout_4.addWidget(self.pushButton_zoomout)

        self.horizontalLayout_4.setStretch(0, 1)
        self.horizontalLayout_4.setStretch(1, 1)

        self.horizontalLayout_11.addLayout(self.horizontalLayout_4)

        self.comboBox = QComboBox(self.frame_pdf_change)
        self.comboBox.setObjectName(u"comboBox")
        self.comboBox.setMaximumSize(QSize(300, 16777215))
        self.comboBox.setStyleSheet(u"/* \u672a\u4e0b\u62c9\u65f6\uff0cQComboBox\u7684\u6837\u5f0f */\n"
"QComboBox {\n"
"    border: 2px groove gray;   /* \u8fb9\u6846 */\n"
"    border-radius: 3px;   /* \u5706\u89d2 */\n"
"    padding: 4px;   /* \u5b57\u4f53\u586b\u886c */\n"
"    color:  rgb(0, 0, 0);\n"
"	border-style: outset;\n"
"    /*font: normal normal 15px \"Microsoft YaHei\";*/\n"
"    /*background: transparent;*/\n"
"}\n"
" \n"
"")

        self.horizontalLayout_11.addWidget(self.comboBox)

        self.horizontalLayout_11.setStretch(0, 3)
        self.horizontalLayout_11.setStretch(1, 1)
        self.horizontalLayout_11.setStretch(2, 3)
        self.horizontalLayout_11.setStretch(3, 4)

        self.horizontalLayout_12.addWidget(self.frame_pdf_change)

        self.horizontalSpacer_6 = QSpacerItem(3, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_12.addItem(self.horizontalSpacer_6)

        self.horizontalLayout_12.setStretch(0, 2)
        self.horizontalLayout_12.setStretch(1, 1)
        self.horizontalLayout_12.setStretch(2, 2)
        self.horizontalLayout_12.setStretch(3, 1)

        self.verticalLayout.addWidget(self.frame_button_pdf)

        self.frame_pdf = QFrame(self.groupBox)
        self.frame_pdf.setObjectName(u"frame_pdf")
        self.frame_pdf.setStyleSheet(u"\n"
"border :0px;")
        self.frame_pdf.setFrameShape(QFrame.StyledPanel)
        self.frame_pdf.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_7 = QHBoxLayout(self.frame_pdf)
        self.horizontalLayout_7.setSpacing(3)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(1, 1, 1, 0)
        self.frame_nav_get = QFrame(self.frame_pdf)
        self.frame_nav_get.setObjectName(u"frame_nav_get")
        self.frame_nav_get.setStyleSheet(u"background-color: rgb(240, 240, 240);\n"
"border :0.4px solid;")
        self.frame_nav_get.setFrameShape(QFrame.StyledPanel)
        self.frame_nav_get.setFrameShadow(QFrame.Raised)
        self.verticalLayout_4 = QVBoxLayout(self.frame_nav_get)
        self.verticalLayout_4.setSpacing(3)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(2, 3, 2, 3)
        self.pdfView_thumbnail = QPdfView(self.frame_nav_get)
        self.pdfView_thumbnail.setObjectName(u"pdfView_thumbnail")
        sizePolicy2 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy2.setHorizontalStretch(10)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.pdfView_thumbnail.sizePolicy().hasHeightForWidth())
        self.pdfView_thumbnail.setSizePolicy(sizePolicy2)
        self.pdfView_thumbnail.setMinimumSize(QSize(122, 432))
        self.pdfView_thumbnail.setStyleSheet(u"background-color: rgb(240, 240, 240);")

        self.verticalLayout_4.addWidget(self.pdfView_thumbnail)


        self.horizontalLayout_7.addWidget(self.frame_nav_get)

        self.frame_pdf_view = QFrame(self.frame_pdf)
        self.frame_pdf_view.setObjectName(u"frame_pdf_view")
        self.frame_pdf_view.setStyleSheet(u"background-color: rgb(0, 255, 127);")
        self.frame_pdf_view.setFrameShape(QFrame.StyledPanel)
        self.frame_pdf_view.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_8 = QHBoxLayout(self.frame_pdf_view)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.pdfView = QPdfView(self.frame_pdf_view)
        self.pdfView.setObjectName(u"pdfView")
        sizePolicy2.setHeightForWidth(self.pdfView.sizePolicy().hasHeightForWidth())
        self.pdfView.setSizePolicy(sizePolicy2)
        self.pdfView.setMinimumSize(QSize(409, 432))
        self.pdfView.setStyleSheet(u"background-color: rgb(240, 240, 240);\n"
"color: rgb(255, 0, 0);")

        self.horizontalLayout_8.addWidget(self.pdfView)


        self.horizontalLayout_7.addWidget(self.frame_pdf_view)

        self.horizontalLayout_7.setStretch(0, 1)
        self.horizontalLayout_7.setStretch(1, 3)

        self.verticalLayout.addWidget(self.frame_pdf)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 45)

        self.horizontalLayout_13.addWidget(self.groupBox)

        self.groupBox_2 = QGroupBox(self.frame_func)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setFont(font)
        self.groupBox_2.setStyleSheet(u"background-color: rgb(240, 240, 240);\n"
"border : 1px solid;")
        self.verticalLayout_2 = QVBoxLayout(self.groupBox_2)
        self.verticalLayout_2.setSpacing(7)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(14, 14, 14, 13)
        self.label_type = QLabel(self.groupBox_2)
        self.label_type.setObjectName(u"label_type")
        self.label_type.setFont(font1)
        self.label_type.setStyleSheet(u"border :0px;")

        self.verticalLayout_2.addWidget(self.label_type)

        self.lineEdit_type = QLineEdit(self.groupBox_2)
        self.lineEdit_type.setObjectName(u"lineEdit_type")
        sizePolicy3 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Ignored)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.lineEdit_type.sizePolicy().hasHeightForWidth())
        self.lineEdit_type.setSizePolicy(sizePolicy3)
        font3 = QFont()
        font3.setPointSize(15)
        self.lineEdit_type.setFont(font3)
        self.lineEdit_type.setStyleSheet(u"border: 0.5px solid gray;\n"
"background-color: rgb(245, 245, 245);")

        self.verticalLayout_2.addWidget(self.lineEdit_type)

        self.label_parameter = QLabel(self.groupBox_2)
        self.label_parameter.setObjectName(u"label_parameter")
        self.label_parameter.setFont(font1)
        self.label_parameter.setStyleSheet(u"border :0px;")

        self.verticalLayout_2.addWidget(self.label_parameter)

        self.tableWidget = QTableWidget(self.groupBox_2)
        if (self.tableWidget.columnCount() < 4):
            self.tableWidget.setColumnCount(4)
        __qtablewidgetitem = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        __qtablewidgetitem2 = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, __qtablewidgetitem2)
        __qtablewidgetitem3 = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3, __qtablewidgetitem3)
        self.tableWidget.setObjectName(u"tableWidget")
        self.tableWidget.setMaximumSize(QSize(16777215, 16777215))
        self.tableWidget.setStyleSheet(u"QHeaderView::section {  \n"
"        padding: 1px;  /* \u8bbe\u7f6e\u8868\u5934\u5185\u8fb9\u8ddd */  \n"
"	   background-color: rgb(72, 116, 203);\n"
"       font-size: 14px;\n"
"	  color: rgb(255, 255, 255);\n"
"       font-weight: bold;\n"
"       text-align: center;\n"
"	  height : 32px;\n"
"       /*border: 1px solid;\n"
"       border-color: rgb(0, 0, 0);\n"
"      margin: 0px 0px;*/\n"
"    }\n"
"")
        self.tableWidget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.tableWidget.setAutoScroll(True)
        self.tableWidget.setSortingEnabled(False)
        self.tableWidget.horizontalHeader().setCascadingSectionResizes(False)
        self.tableWidget.horizontalHeader().setMinimumSectionSize(29)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(77)
        self.tableWidget.horizontalHeader().setHighlightSections(True)
        self.tableWidget.horizontalHeader().setProperty("showSortIndicator", False)
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.verticalHeader().setVisible(False)

        self.verticalLayout_2.addWidget(self.tableWidget)

        self.frame_pin_button = QFrame(self.groupBox_2)
        self.frame_pin_button.setObjectName(u"frame_pin_button")
        self.frame_pin_button.setStyleSheet(u"border: 0px;")
        self.frame_pin_button.setFrameShape(QFrame.StyledPanel)
        self.frame_pin_button.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.frame_pin_button)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 4, 0, 4)
        self.horizontalSpacer_2 = QSpacerItem(19, 15, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_2)

        self.pushButton_edit = QPushButton(self.frame_pin_button)
        self.pushButton_edit.setObjectName(u"pushButton_edit")
        sizePolicy.setHeightForWidth(self.pushButton_edit.sizePolicy().hasHeightForWidth())
        self.pushButton_edit.setSizePolicy(sizePolicy)
        font4 = QFont()
        font4.setPointSize(9)
        self.pushButton_edit.setFont(font4)
        self.pushButton_edit.setStyleSheet(u"QPushButton{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(240,240,240,255),\n"
"stop: 0.495 rgba(200,200,200,255),\n"
"stop: 0.505 rgba(190,190,190,255),\n"
"stop: 1 rgba(185,185,185,255));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px groove gray;\n"
"border-radius: 1px;\n"
"color: rgb(0, 0, 0);\n"
"/*font: 75 12pt ;*/\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"}\n"
" \n"
"QPushButton:hover{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(200,200, 200,230),\n"
"stop: 0.495 rgba(100,100,100,230),\n"
"stop: 0.505 rgba(100,100,100,230),\n"
"stop: 1 rgba(150,150,150,230));\n"
" \n"
"color: rgb(255, 255, 255);\n"
"border: 0.5px groove gray;\n"
"border-radius: 1px;\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"/*color: rgb(255, 255, 255);*/\n"
"/*font: 75 12pt \"\u5fae\u8f6f\u96c5\u9ed1\";*/\n"
"}\n"
" \n"
"QPushButton:pressed{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:"
                        "1,\n"
"stop: 0 rgba(200,200,200,255),\n"
"stop: 0.495 rgba(150,150,150,255),\n"
"stop: 0.505 rgba(145,145,145,255),\n"
"stop: 1 rgba(140,140,140,255));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px rgb(212, 212, 212);\n"
"border-radius: 1px;\n"
"color: rgb(0, 0, 0);\n"
"padding: 0.5px;\n"
"border-style: inset;\n"
"/*font: 75 12pt \"\u5fae\u8f6f\u96c5\u9ed1\";\n"
"padding-left:2px;\n"
"padding-top:1px;*/\n"
"}\n"
"\n"
"QPushButton:disabled{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(250,250,250,125),\n"
"stop: 0.495 rgba(225,225,225,125),\n"
"stop: 0.505 rgba(210,210,210,125),\n"
"stop: 1 rgba(190,190,190,125));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px groove gray;\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"/*font: 75 12pt ;*/\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"}")

        self.horizontalLayout_3.addWidget(self.pushButton_edit)

        self.horizontalSpacer = QSpacerItem(20, 15, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer)

        self.pushButton_save = QPushButton(self.frame_pin_button)
        self.pushButton_save.setObjectName(u"pushButton_save")
        sizePolicy.setHeightForWidth(self.pushButton_save.sizePolicy().hasHeightForWidth())
        self.pushButton_save.setSizePolicy(sizePolicy)
        self.pushButton_save.setFont(font4)
        self.pushButton_save.setStyleSheet(u"QPushButton{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(240,240,240,255),\n"
"stop: 0.495 rgba(200,200,200,255),\n"
"stop: 0.505 rgba(190,190,190,255),\n"
"stop: 1 rgba(185,185,185,255));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px groove gray;\n"
"border-radius: 1px;\n"
"color: rgb(0, 0, 0);\n"
"/*font: 75 12pt ;*/\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"}\n"
" \n"
"QPushButton:hover{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(200,200, 200,230),\n"
"stop: 0.495 rgba(100,100,100,230),\n"
"stop: 0.505 rgba(100,100,100,230),\n"
"stop: 1 rgba(150,150,150,230));\n"
" \n"
"color: rgb(255, 255, 255);\n"
"border: 0.5px groove gray;\n"
"border-radius: 1px;\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"/*color: rgb(255, 255, 255);*/\n"
"/*font: 75 12pt \"\u5fae\u8f6f\u96c5\u9ed1\";*/\n"
"}\n"
" \n"
"QPushButton:pressed{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:"
                        "1,\n"
"stop: 0 rgba(200,200,200,255),\n"
"stop: 0.495 rgba(150,150,150,255),\n"
"stop: 0.505 rgba(145,145,145,255),\n"
"stop: 1 rgba(140,140,140,255));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px rgb(212, 212, 212);\n"
"border-radius: 1px;\n"
"color: rgb(0, 0, 0);\n"
"padding: 0.5px;\n"
"border-style: inset;\n"
"/*font: 75 12pt \"\u5fae\u8f6f\u96c5\u9ed1\";\n"
"padding-left:2px;\n"
"padding-top:1px;*/\n"
"}\n"
"\n"
"QPushButton:disabled{\n"
"background-color:\n"
"qlineargradient(spread:pad,x1:0,x2:0, y1:0, y2:1,\n"
"stop: 0 rgba(250,250,250,125),\n"
"stop: 0.495 rgba(225,225,225,125),\n"
"stop: 0.505 rgba(210,210,210,125),\n"
"stop: 1 rgba(190,190,190,125));\n"
" \n"
"/*color: rgb(255, 255, 255);*/\n"
"border: 0.5px groove gray;\n"
"border-radius: 6px;\n"
"color: rgb(0, 0, 0);\n"
"/*font: 75 12pt ;*/\n"
"padding: 0.5px;\n"
"border-style: outset;\n"
"}")

        self.horizontalLayout_3.addWidget(self.pushButton_save)

        self.horizontalSpacer_3 = QSpacerItem(19, 15, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_3)

        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 2)
        self.horizontalLayout_3.setStretch(2, 1)
        self.horizontalLayout_3.setStretch(3, 2)
        self.horizontalLayout_3.setStretch(4, 1)

        self.verticalLayout_2.addWidget(self.frame_pin_button)

        self.verticalLayout_2.setStretch(0, 2)
        self.verticalLayout_2.setStretch(1, 2)
        self.verticalLayout_2.setStretch(2, 2)
        self.verticalLayout_2.setStretch(3, 30)
        self.verticalLayout_2.setStretch(4, 2)

        self.horizontalLayout_13.addWidget(self.groupBox_2)

        self.horizontalLayout_13.setStretch(0, 3)
        self.horizontalLayout_13.setStretch(1, 1)

        self.verticalLayout_3.addWidget(self.frame_func)

        self.frame_label = QFrame(self.centralwidget)
        self.frame_label.setObjectName(u"frame_label")
        self.frame_label.setFrameShape(QFrame.StyledPanel)
        self.frame_label.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_6 = QHBoxLayout(self.frame_label)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(0, 2, 0, 0)
        self.horizontalSpacer_7 = QSpacerItem(169, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer_7)

        self.frame_choose_img = QFrame(self.frame_label)
        self.frame_choose_img.setObjectName(u"frame_choose_img")
        self.frame_choose_img.setFrameShape(QFrame.StyledPanel)
        self.frame_choose_img.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_14 = QHBoxLayout(self.frame_choose_img)
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.horizontalLayout_14.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setSpacing(3)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.lineEdit_choose_img = QLineEdit(self.frame_choose_img)
        self.lineEdit_choose_img.setObjectName(u"lineEdit_choose_img")
        sizePolicy.setHeightForWidth(self.lineEdit_choose_img.sizePolicy().hasHeightForWidth())
        self.lineEdit_choose_img.setSizePolicy(sizePolicy)
        self.lineEdit_choose_img.setMaximumSize(QSize(40, 16777215))
        font5 = QFont()
        font5.setPointSize(12)
        font5.setItalic(False)
        self.lineEdit_choose_img.setFont(font5)
        self.lineEdit_choose_img.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout.addWidget(self.lineEdit_choose_img)

        self.label_total_img = QLabel(self.frame_choose_img)
        self.label_total_img.setObjectName(u"label_total_img")
        sizePolicy.setHeightForWidth(self.label_total_img.sizePolicy().hasHeightForWidth())
        self.label_total_img.setSizePolicy(sizePolicy)
        self.label_total_img.setFont(font2)
        self.label_total_img.setStyleSheet(u"")
        self.label_total_img.setScaledContents(True)
        self.label_total_img.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_total_img.setMargin(0)

        self.horizontalLayout.addWidget(self.label_total_img)

        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 6)

        self.horizontalLayout_14.addLayout(self.horizontalLayout)


        self.horizontalLayout_6.addWidget(self.frame_choose_img)

        self.horizontalSpacer_8 = QSpacerItem(31, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer_8)

        self.label_img_list = QLabel(self.frame_label)
        self.label_img_list.setObjectName(u"label_img_list")
        font6 = QFont()
        font6.setPointSize(14)
        self.label_img_list.setFont(font6)
        self.label_img_list.setStyleSheet(u"QLabel{\n"
"	background-color: rgb(230, 230, 230);\n"
"}")
        self.label_img_list.setScaledContents(True)
        self.label_img_list.setAlignment(Qt.AlignCenter)
        self.label_img_list.setMargin(0)

        self.horizontalLayout_6.addWidget(self.label_img_list)

        self.horizontalSpacer_9 = QSpacerItem(31, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer_9)

        self.frame_choose_list = QFrame(self.frame_label)
        self.frame_choose_list.setObjectName(u"frame_choose_list")
        self.frame_choose_list.setFrameShape(QFrame.StyledPanel)
        self.frame_choose_list.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_15 = QHBoxLayout(self.frame_choose_list)
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.horizontalLayout_15.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_16 = QHBoxLayout()
        self.horizontalLayout_16.setSpacing(3)
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.lineEdit_choose_list = QLineEdit(self.frame_choose_list)
        self.lineEdit_choose_list.setObjectName(u"lineEdit_choose_list")
        sizePolicy.setHeightForWidth(self.lineEdit_choose_list.sizePolicy().hasHeightForWidth())
        self.lineEdit_choose_list.setSizePolicy(sizePolicy)
        self.lineEdit_choose_list.setMaximumSize(QSize(40, 16777215))
        self.lineEdit_choose_list.setFont(font5)
        self.lineEdit_choose_list.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout_16.addWidget(self.lineEdit_choose_list)

        self.label_total_list = QLabel(self.frame_choose_list)
        self.label_total_list.setObjectName(u"label_total_list")
        sizePolicy.setHeightForWidth(self.label_total_list.sizePolicy().hasHeightForWidth())
        self.label_total_list.setSizePolicy(sizePolicy)
        self.label_total_list.setFont(font2)
        self.label_total_list.setStyleSheet(u"")
        self.label_total_list.setScaledContents(True)
        self.label_total_list.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_total_list.setMargin(0)

        self.horizontalLayout_16.addWidget(self.label_total_list)

        self.horizontalLayout_16.setStretch(0, 1)
        self.horizontalLayout_16.setStretch(1, 6)

        self.horizontalLayout_15.addLayout(self.horizontalLayout_16)


        self.horizontalLayout_6.addWidget(self.frame_choose_list)

        self.horizontalSpacer_10 = QSpacerItem(169, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer_10)

        self.horizontalLayout_6.setStretch(0, 9)
        self.horizontalLayout_6.setStretch(1, 3)
        self.horizontalLayout_6.setStretch(2, 1)
        self.horizontalLayout_6.setStretch(3, 6)
        self.horizontalLayout_6.setStretch(4, 1)
        self.horizontalLayout_6.setStretch(5, 3)
        self.horizontalLayout_6.setStretch(6, 9)

        self.verticalLayout_3.addWidget(self.frame_label)

        self.verticalLayout_3.setStretch(0, 30)
        self.verticalLayout_3.setStretch(1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Package Wizard V1.1", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"PDF\u9884\u89c8", None))
        self.pushButton_open.setText(QCoreApplication.translate("MainWindow", u"\u52a0\u8f7d\u624b\u518c", None))
        self.pushButton_detect.setText(QCoreApplication.translate("MainWindow", u"\u81ea\u52a8\u641c\u7d22", None))
        self.pushButton_pre.setText(QCoreApplication.translate("MainWindow", u"\u4e0a\u4e00\u4e2a", None))
        self.pushButton_next.setText(QCoreApplication.translate("MainWindow", u"\u4e0b\u4e00\u4e2a", None))
        self.pushButton_draw_img.setText(QCoreApplication.translate("MainWindow", u"\u6846\u5c01\u88c5\u56fe", None))
        self.pushButton_draw_list.setText(QCoreApplication.translate("MainWindow", u"\u6846\u5c01\u88c5\u8868", None))
        self.pushButton_reco.setText(QCoreApplication.translate("MainWindow", u"\u63d0\u53d6\u53c2\u6570", None))
        self.lineEdit_page.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_total_page.setText(QCoreApplication.translate("MainWindow", u"/0", None))
        self.pushButton_zoomin.setText(QCoreApplication.translate("MainWindow", u"-", None))
        self.pushButton_zoomout.setText(QCoreApplication.translate("MainWindow", u"+", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"\u5c01\u88c5\u53c2\u6570", None))
        self.label_type.setText(QCoreApplication.translate("MainWindow", u"\u5c01\u88c5\u7c7b\u578b", None))
        self.label_parameter.setText(QCoreApplication.translate("MainWindow", u"\u5c3a\u5bf8\u6570\u636e", None))
        ___qtablewidgetitem = self.tableWidget.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("MainWindow", u"\u53c2\u6570", None));
        ___qtablewidgetitem1 = self.tableWidget.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("MainWindow", u"Min", None));
        ___qtablewidgetitem2 = self.tableWidget.horizontalHeaderItem(2)
        ___qtablewidgetitem2.setText(QCoreApplication.translate("MainWindow", u"Type", None));
        ___qtablewidgetitem3 = self.tableWidget.horizontalHeaderItem(3)
        ___qtablewidgetitem3.setText(QCoreApplication.translate("MainWindow", u"Max", None));
        self.pushButton_edit.setText(QCoreApplication.translate("MainWindow", u"\u7f16\u8f91", None))
        self.pushButton_save.setText(QCoreApplication.translate("MainWindow", u"\u4fdd\u5b58", None))
        self.lineEdit_choose_img.setText("")
        self.label_total_img.setText(QCoreApplication.translate("MainWindow", u"/M\u4e2a\u5c01\u88c5\u56fe", None))
        self.label_img_list.setText(QCoreApplication.translate("MainWindow", u"\u6b63\u5728\u6846\u5c01\u88c5\u56fe", None))
        self.lineEdit_choose_list.setText("")
        self.label_total_list.setText(QCoreApplication.translate("MainWindow", u"/N\u4e2a\u5c01\u88c5\u8868", None))
    # retranslateUi

