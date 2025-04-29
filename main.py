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
        
        # Connect UI elements to their respective methods
        self.ui.pushButton_browse.clicked.connect(self.load_image)
        self.ui.comboBox_thresholding.currentIndexChanged.connect(self.apply_thresholding)
        self.ui.QSlider.sliderReleased.connect(self.change_min_distance)
        self.ui.checkBox_local_thresholding.stateChanged.connect(self.apply_local_thresholding)
        self.ui.pushButton_apply.clicked.connect(self.apply_segmentation)

        # Connect the comboBox for Region Growing to automatically trigger seed selection
        self.ui.comboBox_segmentation.currentIndexChanged.connect(self.on_segmentation_changed)

    ################################################################################################################
    def load_image(self):
        self.clear()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if file_path:
            self.uploded_image = cv2.imread(file_path)
            self.display_image(self.uploded_image, self.ui.original_image_lbl)

    def display_image(self, image, label):
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            label.setPixmap(pixmap)
            label.setScaledContents(True)

    def apply_thresholding(self, index):
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
            self.display_image(image, self.ui.modified_image_lbl)
        elif index == 2:
            image = self.thresholding_obj.otsu_binarization(image_copy)
            self.display_image(image, self.ui.modified_image_lbl)
        else:
            image = self.thresholding_obj.multi_modal_thresholding(image_copy, self.min_distance)
            self.last_state = image
            self.display_image(image, self.ui.modified_image_lbl)

    def change_min_distance(self):
        if not self.ui.checkBox_local_thresholding.isChecked():
            self.min_distance = self.ui.QSlider.value()
            image_copy = cv2.cvtColor(self.image_copy, cv2.COLOR_BGR2GRAY)
            image = self.thresholding_obj.multi_modal_thresholding(image_copy, self.min_distance)
            self.display_image(image, self.ui.modified_image_lbl)
        else:
            self.min_distance = self.ui.QSlider.value()
            image_copy = cv2.cvtColor(self.image_copy, cv2.COLOR_BGR2GRAY)
            image = self.thresholding_obj.local_thresholding(image_copy, self.min_distance)
            self.display_image(image, self.ui.modified_image_lbl)

    def apply_local_thresholding(self, state):
        if self.uploded_image is None:
            print("No image loaded.")
            return

        if state == 2:
            self.image_copy = self.uploded_image.copy()
            image_copy = cv2.cvtColor(self.image_copy, cv2.COLOR_BGR2GRAY)
            image = self.thresholding_obj.local_thresholding(image_copy, self.min_distance)
            self.display_image(image, self.ui.modified_image_lbl)
        else:
            self.display_image(self.uploded_image, self.ui.modified_image_lbl)

        print("enter")
    
    def visualize_segments(image, labels, type):
        num_segments = np.max(labels) + 1
        color_map = np.random.randint(0, 255, size=(num_segments, 3), dtype=np.uint8)
        segmented_image = np.zeros_like(image)
        for i in range(num_segments):
            mask = labels == i
            if type == 1: # mean_shift
                segmented_image[mask, :] = color_map[i]  # Fixed: Added [:] to specify all channels
            else:
                segmented_image[mask] = color_map[i]

        # cv2.imshow('Segmented Image', segmented_image)
        # cv2.waitKey(0)
        pass

    ################################################################################################################
    # Region Growing and K-Means logic

    def on_segmentation_changed(self, index):
        selected_segmentation = self.ui.comboBox_segmentation.currentText()

        if selected_segmentation == "Region Growing":
            # Show the original image and allow seed selection
            self.display_image(self.uploded_image, self.ui.original_image_lbl)
            cv2.imshow("Select Seed Point", self.uploded_image)
            cv2.setMouseCallback("Select Seed Point", self.select_seed)
        else:
            cv2.destroyAllWindows()


    def select_seed(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.seed_point = (x, y)  # Store the seed point
            print(f"Seed point selected at: {self.seed_point}")
            cv2.circle(self.uploded_image, self.seed_point, 5, (0, 0, 255), -1)  # Visual feedback (red circle)
            self.display_image(self.uploded_image, self.ui.original_image_lbl)  # Refresh the image display

    def apply_segmentation(self):
        if self.uploded_image is None:
            print("No image loaded.")
            return

        selected_segmentation = self.ui.comboBox_segmentation.currentText()

        if selected_segmentation == "K-Means Clustering":
            try:
                num_clusters = self.ui.spinBox_clusters.value()
                kmean = KMeansSegmentation(n_clusters=num_clusters, max_iter=100)
                original, outputimage = kmean.segment_image(self.uploded_image)
                self.display_image(outputimage, self.ui.modified_image_lbl)
            except Exception as e:
                print(f"Error during segmentation: {e}")
        elif selected_segmentation == "Region Growing":
            if self.seed_point is None:
                print("No seed point selected. Please select a seed point on the image.")
                return

            # Apply region growing with the selected seed point
            threshold = 10  # Set a default threshold or get from user input
            segmented_image = self.region_growing(self.uploded_image, self.seed_point, threshold)
            self.display_image(segmented_image, self.ui.modified_image_lbl)



    def region_growing(self, image, seed, threshold):
        # Ensure image is grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get image dimensions
        height, width = image.shape
        
        # Initialize output segmented image (0 for background, 255 for region)
        segmented = np.zeros((height, width), dtype=np.uint8)
        # Initialize visited array as boolean
        visited = np.zeros((height, width), dtype=bool)
        
        # Seed point and its intensity
        seed_x, seed_y = seed
        seed_value = image[seed_y, seed_x]
        
        # Stack for DFS to store pixels to process
        stack = [(seed_x, seed_y)]
        # 7-connectivity neighbors: 4 cardinal + 3 diagonals (top-left, top-right, bottom-left)
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1)]
        
        while stack:
            # Get current pixel
            x, y = stack.pop()
            
            # Skip if already visited or out of bounds
            if visited[y, x] or x < 0 or x >= width or y < 0 or y >= height:
                continue
                
            # Mark as visited
            visited[y, x] = True
            
            # Check if pixel intensity is within threshold
            if abs(int(image[y, x]) - int(seed_value)) <= threshold:
                segmented[y, x] = 255  # Mark pixel as part of region
                
                # Add neighboring pixels to stack
                for dx, dy in neighbors:
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < width and 0 <= new_y < height and not visited[new_y, new_x]:
                        stack.append((new_x, new_y))
        
        return segmented

    def clear(self):
        self.ui.original_image_lbl.clear()
        self.ui.modified_image_lbl.clear()

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
