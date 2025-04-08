# Multi-Feature Region Growing
import cv2
import numpy as np

def multi_feature_region_growing(img, seed, threshold=15):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Feature maps: intensity, hue, saturation
    features = np.dstack((gray, hsv[:,:,0], hsv[:,:,1]))
    
    # Initialize region and queue
    region = np.zeros_like(gray, dtype=bool)
    queue = [seed]
    region[seed] = True
    
    while queue:
        x, y = queue.pop(0)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1]:
                    if not region[nx, ny]:
                        current_feat = features[x,y]
                        neighbor_feat = features[nx,ny]
                        diff = np.sqrt(np.sum((current_feat - neighbor_feat)**2))
                        if diff < threshold:
                            region[nx, ny] = True
                            queue.append((nx, ny))
    return region.astype(np.uint8)*255

# Usage
img = cv2.imread("input.jpg")
seed_point = (100, 150)  # Set seed point
result = multi_feature_region_growing(img, seed_point)

# Display
cv2.imshow("Region Growing", result)