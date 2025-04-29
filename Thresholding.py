import numpy as np
import cv2 as cv
class thresholding:
    def __init__(self):
        pass
    def otsu_binarization(self,img):
        # make the img grayscale
        assert img is not None, "file could not be read, check with os.path.exists()"
        blur = cv.GaussianBlur(img,(5,5),0)

        hist = cv.calcHist([blur],[0],None,[256],[0,256])
        hist_norm = hist.ravel()/hist.sum()
        Q = hist_norm.cumsum()

        bins = np.arange(256)

        fn_min = np.inf
        thresh = -1

        for i in range(1,256):
            p1,p2 = np.hsplit(hist_norm,[i])
            q1,q2 = Q[i],Q[255]-Q[i]
            if q1 < 1.e-6 or q2 < 1.e-6:
                continue
            b1,b2 = np.hsplit(bins,[i])

            m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
            v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

            fn = v1*q1 + v2*q2
            if fn < fn_min:
                fn_min = fn
                thresh = i
        _, binary_img = cv.threshold(blur, thresh, 255, cv.THRESH_BINARY)
        return binary_img


    def optimal_iterative_thresholding(self,image):
        height, width = image.shape

        first_pixel = int(image[0, 0])
        second_pixel = int(image[height - 1, 0])
        third_pixel = int(image[0, width - 1])
        fourth_pixel = int(image[height - 1, width - 1])

        initial_sum_background_pixels = (first_pixel + second_pixel + third_pixel + fourth_pixel)
        initial_sum_object_pixels = np.sum(image) - initial_sum_background_pixels

        initial_mean_background = initial_sum_background_pixels / 4
        initial_mean_object = initial_sum_object_pixels / ((height*width)-4)
        initial_threshold = (initial_mean_object + initial_mean_background) / 2


        threshold = initial_threshold
        while True:
            sum_background_pixels = 0
            num_background_pixels = 0

            sum_object_pixels = 0
            num_object_pixels = 0

            for y in range(height):
                for x in range(width):
                    if image[y,x] < threshold :
                        sum_background_pixels += int(image[y, x])
                        num_background_pixels += 1
                    else:
                        sum_object_pixels += int(image[y, x])
                        num_object_pixels += 1

            new_object_mean = sum_object_pixels / num_object_pixels
            new_background_mean = sum_background_pixels / num_background_pixels

            previous_threshold = threshold
            threshold = (new_object_mean + new_background_mean) / 2

            if previous_threshold == threshold:
                return threshold

    def multi_modal_thresholding(self,image,min_distance = 50):
        image_histogram = cv.calcHist([image], [0], None, [256], [0, 256])
        image_histogram = image_histogram.flatten()

        histogram_copy = image_histogram.copy()
        max_peak_index = image_histogram.argmax()     ##find maximum peak index

        left = max(0, max_peak_index - min_distance)
        right = min(255, max_peak_index + min_distance)
        histogram_copy[left:right + 1] = 0
        second_max_peak_index = histogram_copy.argmax()       ## find maximum second peak

        sub_left = max(0, second_max_peak_index - min_distance)
        sub_right = min(255, second_max_peak_index + min_distance)
        histogram_copy[sub_left:sub_right + 1] = 0
        third_max_peak_index = histogram_copy.argmax()

        max_threshold = (max_peak_index + second_max_peak_index) / 2
        second_max_threshold = (second_max_peak_index + third_max_peak_index) / 2

        height, width = image.shape
        for y in range(height):
            for x in range(width):
                if image[y, x] < second_max_threshold:
                    image[y, x] = 0
                if image[y, x] > second_max_threshold and image[y, x] < max_threshold:
                    image[y, x] = 128
                if image[y, x] > max_threshold and image[y, x] > second_max_threshold:
                    image[y, x] = 255

        return image


    def divide_image(self,image,start_x,end_x,start_y,end_y):
        return image[start_y:end_y, start_x:end_x]


    def local_thresholding(self,image,min_dtsance = 50):
        first_half = None
        second_half = None
        third_half = None
        fourth_half = None


        height, width = image.shape
        first_half = self.divide_image(image,0,width//2,0,height//2)
        second_half = self.divide_image(image,width//2,width,0,height//2)
        third_half = self.divide_image(image,0,width//2,height//2,height)
        fourth_half = self.divide_image(image,width//2,width,height//2,height)

        first_half_thresholded = self.multi_modal_thresholding(first_half,min_dtsance)
        second_half_thresholded = self.multi_modal_thresholding(second_half,min_dtsance)
        third_half_thresholded = self.multi_modal_thresholding(third_half,min_dtsance)
        fourth_half_thresholded = self.multi_modal_thresholding(fourth_half,min_dtsance)

        image[0:height // 2, 0:width // 2] = first_half_thresholded
        image[0:height // 2, width // 2:width] = second_half_thresholded
        image[height // 2:height, 0:width // 2] = third_half_thresholded
        image[height // 2:height, width // 2:width] = fourth_half_thresholded

        return image






    def apply_iterative_thresholding(self,image,threshold):
        height, width = image.shape
        for y in range(height):
            for x in range(width):
                if image[y, x] < threshold:
                   image[y, x] = 0
                else:
                    image[y, x] = 255

        return image
























