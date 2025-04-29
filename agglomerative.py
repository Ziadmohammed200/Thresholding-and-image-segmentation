import numpy as np

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, p):
        if self.parent[p] != p:
            self.parent[p] = self.find(self.parent[p])
        return self.parent[p]

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP == rootQ:
            return
        if self.rank[rootP] < self.rank[rootQ]:
            self.parent[rootP] = rootQ
        elif self.rank[rootP] > self.rank[rootQ]:
            self.parent[rootQ] = rootP
        else:
            self.parent[rootQ] = rootP
            self.rank[rootP] += 1

def segment_image(image, T):
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    
    H, W = image.shape[:2]
    
    edges = []
    for y in range(H):
        for x in range(W - 1):
            p1 = y * W + x
            p2 = y * W + (x + 1)
            weight = np.linalg.norm(image[y, x] - image[y, x + 1])
            edges.append((weight, p1, p2))
    for y in range(H - 1):
        for x in range(W):
            p1 = y * W + x
            p2 = (y + 1) * W + x
            weight = np.linalg.norm(image[y, x] - image[y + 1, x])
            edges.append((weight, p1, p2))
    
    edges.sort()
    
    uf = UnionFind(H * W)
    
    for weight, p1, p2 in edges:
        if weight > T:
            break
        if uf.find(p1) != uf.find(p2):
            uf.union(p1, p2)
    
    labels = np.zeros(H * W, dtype=int)
    label_map = {}
    current_label = 0
    for i in range(H * W):
        root = uf.find(i)
        if root not in label_map:
            label_map[root] = current_label
            current_label += 1
        labels[i] = label_map[root]
    
    return labels.reshape(H, W)


def visualize_segments(image, labels):
    num_segments = np.max(labels) + 1
    color_map = np.random.randint(0, 255, size=(num_segments, 3), dtype=np.uint8)

    segmented_image = np.zeros_like(image)
    for i in range(num_segments):
        mask = labels == i
        segmented_image[mask] = color_map[i]

    cv2.imshow('Segmented Image', segmented_image)
    cv2.waitKey(0)

import cv2
image = cv2.imread('test_segmentation.png')
T = 10.0  
labels = segment_image(image, T)
visualize_segments(image, labels)

