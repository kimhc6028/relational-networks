# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 10:29:41 2022

@author: helen, george chen(neural022)
"""

import sys
import cv2 
from PyQt5 import QtWidgets
from utils import Ui_MainWindow, RNPredictor
from PIL import ImageQt,Image
from numpy import asarray
import time


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        self.setupUi(self)
        self.rn_predictor = RNPredictor()
        
        self.pushButton.clicked.connect(self.pushButton_clicked)
        self.pushButton_2.clicked.connect(self.label.pushButton2_clicked)
        
    def pushButton_clicked(self):#OK       
        label_image = ImageQt.fromqpixmap(self.label.grab())
        image_RGB = Image.new("RGB", label_image.size, (255, 255, 255))       
        image_RGB.paste(label_image, mask=label_image.split()[3])        
        image_RGB = image_RGB.resize((75, 75), Image.ANTIALIAS)
        image_RGB.save('image_RGB.jpg', 'JPEG', quality=100)
        img_rgb_array = asarray(image_RGB)
        img_bgr_array = cv2.cvtColor(img_rgb_array, cv2.COLOR_RGB2BGR)
        # print(img_bgr_array.shape)#(75,75,3)
        time.sleep(0.5)
        # Question ComboBox
        # Answer label4
        if self.comboBox.currentText() != '':
            self.question = self.comboBox.currentText()
            print('Question:', self.question)
            question = self.rn_predictor.tokenize(self.question)
            self.answer = self.rn_predictor.predict((img_bgr_array/255, question))
            print('Answer:', self.answer)
            self.label4.setText(self.answer)
            

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    app.setStyle('Fusion')
    w = MainWindow()
    w.show()
    
    sys.exit(app.exec())
    