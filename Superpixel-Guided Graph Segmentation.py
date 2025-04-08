# Superpixel-Guided Graph Segmentation

from skimage.segmentation import slic, mark_boundaries
from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2lab
from sklearn.cluster import AgglomerativeClustering

def superpixel_graph_segmentation(image_path, n_segments=200):
    image = io.imread(image_path)
    superpixels = slic(image, n_segments=n_segments, compactness=10)
    
    # Extract superpixel features
    features = []
    for sp in np.unique(superpixels):
        mask = superpixels == sp
        mean_color = np.mean(image[mask], axis=0)
        features.append(mean_color)
    
    # Cluster superpixels
    cluster = AgglomerativeClustering(n_clusters=5)
    labels = cluster.fit_predict(features)
    
    # Create final segmentation
    result = np.zeros_like(superpixels)
    for label, sp in enumerate(np.unique(superpixels)):
        result[superpixels == sp] = labels[label]
    
    # Display
    return superpixels, result

# Usage
image = io.imread("input.jpg")
superpixels, result = superpixel_graph_segmentation("input.jpg")
superpixel_graph_segmentation("input.jpg")

plt.figure(figsize=(12,6))
plt.subplot(121), plt.imshow(mark_boundaries(image, superpixels))
plt.subplot(122), plt.imshow(result, cmap='tab20')
plt.show()