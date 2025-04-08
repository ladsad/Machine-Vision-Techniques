import cv2

image = cv2.imread("path_to_your_image.jpg")

if image is None:
    print("Error: Image not found or unable to read.")
    exit()

# Display the original BGR image (OpenCV uses BGR by default)
cv2.imshow("Original (BGR)", image)

# Convert BGR to RGB and display
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow("RGB", rgb_image)

# Convert BGR to HSV and display
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv_image)

# Split and display individual HSV channels
for name, channel in zip(["Hue", "Saturation", "Value"], cv2.split(hsv_image)):
    cv2.imshow(name, channel)

# Convert BGR to Grayscale and display
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", gray_image)