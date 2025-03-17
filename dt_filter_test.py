import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import tranformations

if __name__ == "__main__":

    epsilon=1e-6

    # Load an image as guidance and src (for demonstration, use the same image)
    dirname = os.path.dirname(__file__)
    img = cv2.imread(os.path.join(dirname, 'MEFDatabase/source image sequences/Farmhouse_hdr-project.com/DSC03641.png'))  # Replace with your image path
    if img is None:
        raise FileNotFoundError("Image not found. Check the provided path.")
    
    # Convert to RGB and then normalize to [0,1]
    guidance = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    

    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    contrast = np.abs(cv2.Laplacian(gray, cv2.CV_32F))

    # Compute saturation weight: standard deviation across color channels
    saturation = np.std(img, axis=2)

    # Compute well-exposedness weight: product of Gaussian responses for each channel
    well_exposedness = np.exp(-0.5 * ((img - 0.5) / 0.2) ** 2)
    well_exposedness = np.prod(well_exposedness, axis=2)

    # Combine the weights with a small epsilon to avoid zeros
    weight = (contrast + epsilon) * (saturation + epsilon) * (well_exposedness + epsilon)

    # Apply our domain transform filter
    smooth_weight = tranformations.dt_filter(guidance, weight, sigmaSpatial=60, sigmaColor=0.4, num_iterations=3)

    # Display original and smoothed weight maps
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(weight, cmap='gray')
    plt.title("Original Weight Map")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(smooth_weight, cmap='gray')
    plt.title("Smoothed Weight Map (Domain Transform)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
