# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 18:03:34 2022

@author: helen
"""

from PyQt5 import QtWidgets
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap
import random

class Shape():
    def __init__(self):
        self.object_shape = []
        self.object_x = [ 17, 114, 211, 308, 405, 502 ]
        self.object_y = [ 500, 500, 500, 500, 500, 500 ]
        self.object_color = [ "#FF0000",    # r
                            "#00FF00",      # g
                            "#0000FF",      # b
                            "#ff9d00",      # o
                            "#808080",      # k
                            "#FFFF00" ]     # y
        
    def random_shape(self):
        random.shuffle(self.object_color)       
        self.object_shape.clear()
        for i in range(6):
            self.object_shape.append(random.choice(['rectangle', 'circle']))

class ShapeCanvas(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.canvas = QPixmap(600, 600)
        self.canvas.fill(QColor("white"))
        self.setPixmap(self.canvas)
        
        self.shape = Shape()
        self.shape.random_shape()
        self.shape_size = 80
        self.drag_idx = -1
        
    def paintEvent(self, event):
        super(ShapeCanvas, self).paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        for i in range(6):
            color = QColor(self.shape.object_color[i])
            pen = QPen(color, 1)
            painter.setBrush(color)
            painter.setPen(pen)
            if self.shape.object_shape[i] == 'rectangle':
                painter.drawRect(QRect(self.shape.object_x[i],self.shape.object_y[i],self.shape_size,self.shape_size))
            else:
                painter.drawEllipse(QRect(self.shape.object_x[i],self.shape.object_y[i],self.shape_size,self.shape_size))
                
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:            
            self.drag_idx = self.drag_index(event.pos())                
                # print('drag_index:', self.drag_idx) 
                
    def drag_index(self, position):
        for i in range(6):
            if self.is_pointSelected(self.shape.object_x[i],self.shape.object_y[i], position):
                return i
        return -1        
       
    def is_pointSelected(self, point_x ,point_y, position):
        # point range
        x_min =point_x - self.shape_size
        x_max = point_x + self.shape_size
        y_min = point_y - self.shape_size
        y_max = point_y + self.shape_size

        # check control points in select range position
        x, y = position.x(), position.y()
        if x_min < x < x_max and y_min < y < y_max:
            return True
        return False
    
    def mouseMoveEvent(self, event):
        if self.drag_idx != -1:
            self.shape.object_x[self.drag_idx] = event.pos().x()
            self.shape.object_y[self.drag_idx] = event.pos().y()
            self.update()
            
    def mouseReleaseEvent(self, event):
        self.drag_idx = -1
        self.update()
        
    
    def pushButton2_clicked(self): # Random
        self.shape.random_shape()
        self.repaint()
        
            
    
        