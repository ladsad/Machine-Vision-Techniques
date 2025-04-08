import cv2
import numpy as np

# Load the image
image = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

# Apply Gaussian Blur to reduce noise
blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

# Sobel edge detection
sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x direction
sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y direction

# Combine gradients
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# Display results
cv2.imshow("Original Image", image)
cv2.imshow("Sobel X", np.uint8(np.abs(sobel_x)))
cv2.imshow("Sobel Y", np.uint8(np.abs(sobel_y)))
cv2.imshow("Sobel Combined", np.uint8(sobel_combined))

# Define Prewitt kernels
prewitt_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
prewitt_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

# Apply Prewitt filters
prewitt_x_edges = cv2.filter2D(blurred_image, -1, prewitt_x)
prewitt_y_edges = cv2.filter2D(blurred_image, -1, prewitt_y)

# Combine gradients
prewitt_combined = np.sqrt(prewitt_x_edges**2 + prewitt_y_edges**2).astype(np.uint8)

# Display results
cv2.imshow("Prewitt X", prewitt_x_edges)
cv2.imshow("Prewitt Y", prewitt_y_edges)
cv2.imshow("Prewitt Combined", prewitt_combined)

# Define Roberts kernels
roberts_x = np.array([[1, 0], [0, -1]])
roberts_y = np.array([[0, 1], [-1, 0]])

# Apply Roberts filters
roberts_x_edges = cv2.filter2D(image.astype(np.float32), -1, roberts_x)
roberts_y_edges = cv2.filter2D(image.astype(np.float32), -1, roberts_y)

# Combine gradients
roberts_combined = np.sqrt(roberts_x_edges**2 + roberts_y_edges**2).astype(np.uint8)

# Display results
cv2.imshow("Roberts X", np.uint8(np.abs(roberts_x_edges)))
cv2.imshow("Roberts Y", np.uint8(np.abs(roberts_y_edges)))
cv2.imshow("Roberts Combined", roberts_combined)