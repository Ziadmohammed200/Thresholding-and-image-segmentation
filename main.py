from PySide6.QtUiTools import QUiLoader
import cv2
from PySide6.QtGui import QImage, QPixmap
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QFile, Qt
from PySide6.QtWidgets import QFileDialog
from Thresholding import thresholding
from mean_shift import run_mean_shift_segmentation
from agglomerative import run_agglomerative
import time
from KMeans import KMeansSegmentation
from collections import deque


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.uploded_image = None
        self.thresholding_obj = thresholding()
        self.min_distance = 20
        self.image_copy = None
        self.seed_point = None

        # Load the UI file
        ui_file = QFile("Task_4.ui")
        ui_file.open(QFile.ReadOnly)
        loader = QUiLoader()
        self.ui = loader.load(ui_file)
        ui_file.close()
        self.setCentralWidget(self.ui)
        self.resize(1400, 900)
        
        # Connect UI elements
        self.ui.pushButton_browse.clicked.connect(self.load_image)
        self.ui.comboBox_thresholding.currentIndexChanged.connect(self.apply_thresholding)
        self.ui.QSlider.sliderReleased.connect(self.change_min_distance)
        self.ui.checkBox_local_thresholding.stateChanged.connect(self.apply_local_thresholding)
        self.ui.pushButton_apply.clicked.connect(self.apply_segmentation)
        self.ui.comboBox_segmentation.currentTextChanged.connect(self.toggle_clusters_visibility)

        # Initialize UI controls visibility
        self.hide_all_segmentation_controls()

    def hide_all_segmentation_controls(self):
        """Hide all segmentation-related controls."""
        controls = [
            'label_clusters', 'spinBox_clusters', 'Max_Iteration_K', 'spinBox_Iteration_K',
            'round_label', 'doubleSpinBox', 'W_label', 'spinBox_3', 'label_4', 'spinBox_2',
            'H_label', 'spinBox_4', 'Threshold_MS', 'doubleSpinBox_MS', 'Threshold_RG',
            'spinBox_ThresholdGR', 'Seed_point_x', 'Seed_point_y','spinBox_point_x','spinBox_point_y'
        ]
        
        for control in controls:
            if hasattr(self.ui, control):
                getattr(self.ui, control).hide()

    def toggle_clusters_visibility(self, text):
        """Show only relevant controls based on selected segmentation method."""
        self.hide_all_segmentation_controls()

        if text == "K-Means Clustering":
            self.show_controls(['label_clusters', 'spinBox_clusters', 'Max_Iteration_K', 'spinBox_Iteration_K'])
        elif text == "Region Growing":
            self.show_controls(['Threshold_RG', 'spinBox_ThresholdGR', 'Seed_point_x', 'Seed_point_y','spinBox_point_x','spinBox_point_y'])
            self.display_image(self.uploded_image, self.ui.original_image_lbl)
            cv2.imshow("Select Seed Point", self.uploded_image)
            cv2.setMouseCallback("Select Seed Point", self.select_seed)
        elif text == "Mean Shift":
            self.show_controls(['Threshold_MS', 'doubleSpinBox_MS'])
        elif text == "Agglomerative":
            self.show_controls(['W_label', 'spinBox_3', 'H_label','doubleSpinBox', 'round_label', 'spinBox_4' ])


    def show_controls(self, control_names):
        """Helper function to show specified controls."""
        for name in control_names:
            if hasattr(self.ui, name):
                getattr(self.ui, name).show()

    def load_image(self):
        """Load an image from file."""
        self.clear()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if file_path:
            self.uploded_image = cv2.imread(file_path)
            self.display_image(self.uploded_image, self.ui.original_image_lbl)

    def display_image(self, image, label):
        """Display image on the specified QLabel."""
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            label.setPixmap(pixmap)
            label.setScaledContents(True)

    def apply_thresholding(self, index):
        """Apply selected thresholding method."""
        if self.uploded_image is None:
            print("No image loaded.")
            return

        self.image_copy = self.uploded_image.copy()
        image_copy = cv2.cvtColor(self.image_copy, cv2.COLOR_BGR2GRAY)

        if index == 0:
            return
        elif index == 1:
            thresh = self.thresholding_obj.optimal_iterative_thresholding(image_copy)
            image = self.thresholding_obj.apply_iterative_thresholding(image_copy, thresh)
        elif index == 2:
            image = self.thresholding_obj.otsu_binarization(image_copy)
        else:
            image = self.thresholding_obj.multi_modal_thresholding(image_copy, self.min_distance)
        
        self.display_image(image, self.ui.modified_image_lbl)

    def change_min_distance(self):
        """Handle changes to the distance threshold slider."""
        if not self.ui.checkBox_local_thresholding.isChecked():
            self.min_distance = self.ui.QSlider.value()
            image_copy = cv2.cvtColor(self.image_copy, cv2.COLOR_BGR2GRAY)
            image = self.thresholding_obj.multi_modal_thresholding(image_copy, self.min_distance)
        else:
            self.min_distance = self.ui.QSlider.value()
            image_copy = cv2.cvtColor(self.image_copy, cv2.COLOR_BGR2GRAY)
            image = self.thresholding_obj.local_thresholding(image_copy, self.min_distance)
        
        self.display_image(image, self.ui.modified_image_lbl)

    def apply_local_thresholding(self, state):
        """Toggle local thresholding."""
        if self.uploded_image is None:
            print("No image loaded.")
            return

        if state == 2:  # Checked
            self.image_copy = self.uploded_image.copy()
            image_copy = cv2.cvtColor(self.image_copy, cv2.COLOR_BGR2GRAY)
            image = self.thresholding_obj.local_thresholding(image_copy, self.min_distance)
            self.display_image(image, self.ui.modified_image_lbl)
        else:  # Unchecked
            self.display_image(self.uploded_image, self.ui.modified_image_lbl)

    def select_seed(self, event, x, y, flags, param):
        """Handle mouse click to select seed point for region growing."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.seed_point = (x, y)
            print(f"Seed point selected at: {self.seed_point}")
            cv2.circle(self.uploded_image, self.seed_point, 5, (0, 0, 255), -1)
            self.display_image(self.uploded_image, self.ui.original_image_lbl)
            cv2.destroyWindow("Select Seed Point")

    def apply_segmentation(self):
        """Apply the selected segmentation method."""
        if self.uploded_image is None:
            print("No image loaded.")
            return

        selected_method = self.ui.comboBox_segmentation.currentText()

        if selected_method == "K-Means Clustering":
            self.apply_kmeans()
        elif selected_method == "Region Growing":
            self.apply_region_growing()
        elif selected_method == "Mean Shift":
            self.apply_mean_shift()
        elif selected_method == "Agglomerative":
            self.apply_agglomerative()

    def apply_kmeans(self):
        """Apply K-Means clustering segmentation."""
        try:
            num_clusters = self.ui.spinBox_clusters.value()
            kmean = KMeansSegmentation(n_clusters=num_clusters, max_iter=100)
            original, outputimage = kmean.segment_image(self.uploded_image)
            self.display_image(outputimage, self.ui.modified_image_lbl)
        except Exception as e:
            print(f"Error during K-Means segmentation: {e}")

    def apply_region_growing(self):
        """Apply region growing segmentation."""
        if self.seed_point is None:
            print("No seed point selected. Please select a seed point on the image.")
            return

        threshold = self.ui.spinBox_ThresholdGR.value() if hasattr(self.ui, 'spinBox_ThresholdGR') else 10
        segmented_image = self.region_growing(self.uploded_image, self.seed_point, threshold)
        self.display_image(segmented_image, self.ui.modified_image_lbl)

    def region_growing(self, image, seed, threshold):
        """Region growing algorithm implementation."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        height, width = image.shape
        segmented = np.zeros((height, width), dtype=np.uint8)
        visited = np.zeros((height, width), dtype=bool)
        
        seed_x, seed_y = seed
        seed_value = image[seed_y, seed_x]
        
        stack = [(seed_x, seed_y)]
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1)]
        
        while stack:
            x, y = stack.pop()
            
            if visited[y, x] or x < 0 or x >= width or y < 0 or y >= height:
                continue
                
            visited[y, x] = True
            
            if abs(int(image[y, x]) - int(seed_value)) <= threshold:
                segmented[y, x] = 255
                
                for dx, dy in neighbors:
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < width and 0 <= new_y < height and not visited[new_y, new_x]:
                        stack.append((new_x, new_y))
        
        return segmented

    def apply_mean_shift(self):
        """Apply mean shift segmentation."""
        try:
            # Get parameters from UI controls
            bandwidth = self.ui.doubleSpinBox.value() if hasattr(self.ui, 'doubleSpinBox') else 0.1
            max_iter = self.ui.spinBox_3.value() if hasattr(self.ui, 'spinBox_3') else 100
            
            # Run mean shift segmentation
            segmented_image = run_mean_shift_segmentation(self.uploded_image, bandwidth, max_iter)
            self.display_image(segmented_image, self.ui.modified_image_lbl)
        except Exception as e:
            print(f"Error during Mean Shift segmentation: {e}")

    def apply_agglomerative(self):
        """Apply agglomerative clustering segmentation."""
        try:
            segmented_image = run_agglomerative(self.uploded_image)
            self.display_image(segmented_image, self.ui.modified_image_lbl)
        except Exception as e:
            print(f"Error during Agglomerative segmentation: {e}")

    def clear(self):
        """Clear the image displays."""
        self.ui.original_image_lbl.clear()
        self.ui.modified_image_lbl.clear()
        self.seed_point = None


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
