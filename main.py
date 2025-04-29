from PySide6.QtUiTools import QUiLoader
import cv2
from PySide6.QtGui import QImage, QPixmap
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QFile, Qt
from PySide6.QtWidgets import QFileDialog
from PySide6.QtGui import QPixmap
from Thresholding import thresholding
import time


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.uploded_image = None
        self.thresholding_obj = thresholding()
        self.min_distance = 20
        self.image_copy = None





        # Load the UI file
        ui_file = QFile("Task_4.ui")
        ui_file.open(QFile.ReadOnly)
        loader = QUiLoader()
        self.ui = loader.load(ui_file)
        ui_file.close()
        self.setCentralWidget(self.ui)

        self.ui.pushButton_browse.clicked.connect(self.load_image)
        self.ui.comboBox_thresholding.currentIndexChanged.connect(self.apply_thresholding)
        self.ui.QSlider.sliderReleased.connect(self.change_min_distance)
        self.ui.checkBox_local_thresholding.stateChanged.connect(self.apply_local_thresholding)







    ################################################################################################################
    def load_image(self):
        self.clear()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if file_path:
            self.uploded_image = cv2.imread(file_path)
            self.display_image(self.uploded_image,self.ui.original_image_lbl)



    def display_image(self, image, label):
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            label.setPixmap(pixmap)
            label.setScaledContents(True)

    def apply_thresholding(self,index):
        self.image_copy = self.uploded_image.copy()
        image_copy=cv2.cvtColor(self.image_copy, cv2.COLOR_BGR2GRAY)
        if index == 0:
            return
        elif index == 1:
            thresh = self.thresholding_obj.optimal_iterative_thresholding(image_copy)
            image = self.thresholding_obj.apply_iterative_thresholding(image_copy,thresh)
            self.display_image(image,self.ui.modified_image_lbl)

        elif index == 2:
            image = self.thresholding_obj.otsu_binarization(image_copy)
            self.display_image(image,self.ui.modified_image_lbl)


        else:
            image = self.thresholding_obj.multi_modal_thresholding(image_copy,self.min_distance)
            self.last_state = image
            self.display_image(image,self.ui.modified_image_lbl)




    def change_min_distance(self):
        if not self.ui.checkBox_local_thresholding.isChecked():
            self.min_distance = self.ui.QSlider.value()
            image_copy=cv2.cvtColor(self.image_copy, cv2.COLOR_BGR2GRAY)
            image = self.thresholding_obj.multi_modal_thresholding(image_copy,self.min_distance)
            self.display_image(image, self.ui.modified_image_lbl)

        else:
            self.min_distance = self.ui.QSlider.value()
            image_copy=cv2.cvtColor(self.image_copy, cv2.COLOR_BGR2GRAY)
            image = self.thresholding_obj.local_thresholding(image_copy,self.min_distance)
            self.display_image(image, self.ui.modified_image_lbl)

    def apply_local_thresholding(self,state):
        print(state)
        if state == 2:
            self.image_copy = self.uploded_image.copy()
            image_copy = cv2.cvtColor(self.image_copy, cv2.COLOR_BGR2GRAY)
            image = self.thresholding_obj.local_thresholding(image_copy,self.min_distance)
            self.display_image(image,self.ui.modified_image_lbl)
            print("enter")

        else:
            self.display_image(self.uploded_image,self.ui.modified_image_lbl)

        print("enter")

    def clear(self):
        self.ui.original_image_lbl.clear()
        self.ui.modified_image_lbl.clear()












if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()