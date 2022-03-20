# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'project_UI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
from PyQt5 import QtCore, QtWidgets,QtGui
from PyQt5.QtGui import QFont as qfont
from .shape_util import ShapeCanvas

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(740, 804)
        self.setStyleSheet("background-color: rgb(224, 227, 210);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(630, 650, 81, 31))
        self.pushButton.setStyleSheet("color: rgb(5,22,39)")
        self.pushButton.setFont(qfont('Times', 10))
        self.pushButton.setObjectName("pushButton")       
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(630, 720, 81, 31))
        self.pushButton_2.setStyleSheet("color: rgb(5,22,39)")
        self.pushButton_2.setFont(qfont('Times', 10))
        self.pushButton_2.setObjectName("pushButton_2")        
        
        self.label = ShapeCanvas(self)
        self.label.setGeometry(QtCore.QRect(70, 20, 600, 600))
        self.label.setAutoFillBackground(True)
        self.label.setText("")
        self.label.setObjectName("label")
        
        self.label2 = QtWidgets.QLabel(self.centralwidget)
        self.label2.setGeometry(QtCore.QRect(10, 630, 80, 15))
        self.label2.setAutoFillBackground(True)
        self.label2.setStyleSheet("color: rgb(5,22,39)")
        self.label2.setFont(qfont('Times', 10 ,QtGui.QFont.Bold))
        self.label2.setText("Question:")
        self.label2.setObjectName("label2")
        
        self.label3 = QtWidgets.QLabel(self.centralwidget)
        self.label3.setGeometry(QtCore.QRect(10, 720, 70, 31))
        self.label3.setAutoFillBackground(True)
        self.label3.setStyleSheet("color: rgb(5,22,39)")
        self.label3.setFont(qfont('Times', 10 ,QtGui.QFont.Bold))
        self.label3.setText("Answer:")
        self.label3.setObjectName("label3")
        
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setEnabled(True)
        self.comboBox.setGeometry(QtCore.QRect(30, 650, 571, 60))
        self.comboBox.setStyleSheet("color: rgb(5,22,39)")
        self.comboBox.setFont(qfont('Times', 12))
        self.comboBox.setEditable(True)
        self.comboBox.lineEdit().setFont(qfont('Times', 12, QtGui.QFont.Bold))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("1.What is the shape of the <color> object ?")
        self.comboBox.addItem("2.Is <color> object placed on the left side of the image ?")
        self.comboBox.addItem("3.Is <color> object placed on the up side of the image ?")
        self.comboBox.addItem("4.What is the shape of the object closest to the <color> object ?")
        self.comboBox.addItem("5.What is the shape of the object furthest to the <color> object ?")
        self.comboBox.addItem("6.How many objects have same shape with the <color> object ?")
        '''
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setStyleSheet("color: rgb(5,22,39)")
        self.textEdit.setFont(qfont('Times', 12))
        self.textEdit.setGeometry(QtCore.QRect(30, 650, 571, 60))
        self.textEdit.setObjectName("textEdit")
        '''
        self.label4 = QtWidgets.QLabel(self.centralwidget)
        self.label4.setStyleSheet("color: rgb(5,22,39)")
        self.label4.setFont(qfont('Times', 12))
        self.label4.setGeometry(QtCore.QRect(100, 720, 500, 31))
        self.label4.setObjectName("label4")
        # test
        # self.label4.setText("true")
        self.label4.setText("")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 852, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "OK"))
        self.pushButton_2.setText(_translate("MainWindow", "Random"))

