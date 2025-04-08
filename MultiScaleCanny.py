import cv2
import numpy as np

def multiscale_canny(image, sigmas=[1, 3, 5], auto_thresh=True, sigma=0.33):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    combined_edges = np.zeros_like(gray)
    
    for s in sigmas:
        # Apply Gaussian blur with current sigma
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=s)
        
        # Automatic threshold calculation [2]
        if auto_thresh:
            v = np.median(blurred)
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
        else:
            lower, upper = 30, 150  # Default thresholds
        
        # Detect edges and combine
        edges = cv2.Canny(blurred, lower, upper)
        combined_edges = cv2.bitwise_or(combined_edges, edges)
    
    return combined_edges

# Usage
image = cv2.imread("input.jpg")
result = multiscale_canny(image, sigmas=[1, 3, 5])

# Display results
cv2.imshow("Multiscale Edges", result)

from skimage import feature, filters
import matplotlib.pyplot as plt

def sk_multiscale_canny(image, sigmas=[1, 3, 5]):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    combined = np.zeros_like(gray)
    
    for sigma in sigmas:
        edges = feature.canny(gray, sigma=sigma)
        combined = np.logical_or(combined, edges)
    
    return combined.astype(np.uint8) * 255

# Usage with visualization
fig, ax = plt.subplots(1, len(sigmas=[1, 3, 5])+2, figsize=(15,5))

ax[0].imshow(image)
ax[0].set_title("Original")

for i, sigma in enumerate([1, 3, 5]):
    edges = feature.canny(gray, sigma=sigma)
    ax[i+1].imshow(edges, cmap='gray')
    ax[i+1].set_title(f"σ={sigma}")

combined = sk_multiscale_canny(image)
ax[-1].imshow(combined, cmap='gray')
ax[-1].set_title("Combined")
plt.show()
