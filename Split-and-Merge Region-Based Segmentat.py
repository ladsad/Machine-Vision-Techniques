# Split-and-Merge Region-Based Segmentation
import cv2
import numpy as np

def is_homogeneous(region, threshold=20):
    return np.ptp(region) <= threshold  # Peak-to-peak intensity difference

def split_merge(image, threshold=20):
    def process(region):
        if is_homogeneous(region, threshold):
            return np.ones(region.shape, dtype=np.uint8)
        h, w = region.shape
        if h <= 2 or w <= 2:
            return np.zeros(region.shape, dtype=np.uint8)
        
        # Split into quadrants
        mid_h, mid_w = h//2, w//2
        quads = [
            region[:mid_h, :mid_w],
            region[:mid_h, mid_w:],
            region[mid_h:, :mid_w],
            region[mid_h:, mid_w:]
        ]
        
        # Process and merge
        processed = [process(q) for q in quads]
        return np.vstack((
            np.hstack((processed[0], processed[1])),
            np.hstack((processed[2], processed[3]))
        ))
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return process(gray) * 255

# Usage and display
result = split_merge(cv2.imread("input.jpg"))
cv2.imshow("Split-Merge", result)