import numpy as np
import cv2
import random

class KMeansSegmentation:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4):
        """Initialize the KMeans parameters."""
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None

    def initialize_centroids(self, X):
        """Randomly pick k points from data to be the initial centroids."""
        indices = random.sample(range(X.shape[0]), self.n_clusters)
        return X[indices]

    def compute_distances(self, X, centroids):
        """Compute distance from every point to every centroid."""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.linalg.norm(X - centroids[i], axis=1)    #For centroid i, this finds the distance from every pixel to that centroid.
        return distances

    def compute_centroids(self, X, labels):
        """Re-calculate centroids as the mean of all points assigned to each cluster."""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            if np.any(labels == i):  # avoid empty cluster
                new_centroids[i] = np.mean(X[labels == i], axis=0)
            else:
                new_centroids[i] = X[random.randint(0, X.shape[0] - 1)]  # reinitialize randomly
        return new_centroids

    def fit(self, X):
        """Fit K-means to the data."""
        self.centroids = self.initialize_centroids(X)

        for _ in range(self.max_iter):
            distances = self.compute_distances(X, self.centroids)
            self.labels = np.argmin(distances, axis=1)

            new_centroids = self.compute_centroids(X, self.labels)

            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                break

            self.centroids = new_centroids


    def segment_image(self, image, grayscale=False):
        """
        Segment an image using the fitted KMeans algorithm.

        Parameters:
            image (numpy.ndarray): Input image (already loaded, not path).
            grayscale (bool): Whether to treat the image as grayscale.

        Returns:
            original (numpy.ndarray): The original image.
            segmented (numpy.ndarray): The segmented image.
        """
        original = image.copy()

        if grayscale:
            if len(original.shape) == 3:
                original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            X = original.reshape(-1, 1)  # Flatten to one column (grayscale)
        else:
            if len(original.shape) == 2:  # If grayscale, convert to RGB
                original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
            X = original.reshape(-1, 3)  # Flatten to (n_pixels, 3)

        original_shape = original.shape  # Store the original shape for reshaping

        # Fit KMeans on the image pixels
        self.fit(X)

        # Map each pixel to the nearest centroid (assign color)
        segmented = self.centroids[self.labels]

        # Ensure color values are valid and within the range [0, 255]
        segmented = np.clip(segmented, 0, 255).astype(np.uint8)

        # Reshape back to the original image shape
        if grayscale:
            segmented = segmented.reshape(original_shape[:2])  # For grayscale, it has only two dimensions
        else:
            segmented = segmented.reshape(original_shape)  # For color image, reshape to 3D

        return original, segmented

