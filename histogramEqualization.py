import cv2
import numpy as np

# Load image in grayscale
gray_img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

# Global Histogram Equalization [6][8]
equalized_global = cv2.equalizeHist(gray_img)

# CLAHE (Adaptive Histogram Equalization) [6][8]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
equalized_clahe = clahe.apply(gray_img)

# Display results
cv2.imshow("Original Grayscale", gray_img)
cv2.imshow("Global Equalization", equalized_global)
cv2.imshow("CLAHE", equalized_clahe)

def equalize_color(image):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    return cv2.cvtColor(yuv, cv2.COLOR_HSV2BGR)