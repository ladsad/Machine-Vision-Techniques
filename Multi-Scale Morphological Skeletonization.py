import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, medial_axis
from skimage import data, filters, measure

def multi_scale_skeleton_analysis(image_path, scales=[1, 3, 5]):
    # Load and preprocess image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    
    # Analysis dictionaries
    skeletons = {}
    features = {}

    for sigma in scales:
        # Multi-scale smoothing
        blurred = cv2.GaussianBlur(binary.astype(float), (0,0), sigmaX=sigma)
        _, scaled_binary = cv2.threshold(blurred.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        
        # Skeletonization
        skeleton = skeletonize(scaled_binary // 255)
        skel_medial, distance = medial_axis(scaled_binary, return_distance=True)
        
        # Store results
        skeletons[sigma] = {
            'basic': skeleton,
            'medial': skel_medial,
            'distance': distance
        }
        
        # Feature extraction
        features[sigma] = {
            'branch_points': find_branch_points(skeleton),
            'end_points': find_end_points(skeleton),
            'total_length': np.sum(skeleton),
            'mean_width': np.mean(distance[skel_medial])*2  # Convert radius to diameter
        }

    return skeletons, features

def find_branch_points(skeleton):
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    conv = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    return (conv > 12).astype(np.uint8)

def find_end_points(skeleton):
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    conv = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    return (conv == 11).astype(np.uint8)

# Usage
skeletons, features = multi_scale_skeleton_analysis("plant_mask.png", scales=[1, 3, 5])

# Visualization
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for idx, sigma in enumerate([1, 3, 5]):
    # Original scaled binary
    axes[idx,0].imshow(skeletons[sigma]['basic'], cmap='gray')
    axes[idx,0].set_title(f"σ={sigma} Skeleton")
    
    # Medial axis with width
    axes[idx,1].imshow(skeletons[sigma]['distance'] * skeletons[sigma]['medial'], cmap='magma')
    axes[idx,1].set_title(f"σ={sigma} Width Map")
    
    # Feature points
    feature_img = np.zeros((*skeletons[sigma]['basic'].shape, 3))
    feature_img[...,0] = find_branch_points(skeletons[sigma]['basic'])  # Red branches
    feature_img[...,1] = find_end_points(skeletons[sigma]['basic'])     # Green endpoints
    axes[idx,2].imshow(feature_img)
    axes[idx,2].set_title(f"σ={sigma} Features")

plt.tight_layout()
plt.show()

# Print feature table
print("Scale | Branches | Endpoints | Length | Avg Width")
for sigma in [1, 3, 5]:
    f = features[sigma]
    print(f"{sigma:5} | {np.sum(f['branch_points']):8} | {np.sum(f['end_points']):9} | {f['total_length']:6} | {f['mean_width']:.2f}")
