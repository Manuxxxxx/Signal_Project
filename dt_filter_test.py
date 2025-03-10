import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def dt_filter(guidance, src, sigmaSpatial=60, sigmaColor=0.4, num_iterations=3):
    """
    A naive Python implementation of the Domain Transform filter for edge-aware smoothing.
    
    This function applies recursive filtering along horizontal and vertical directions
    guided by the guidance image. It computes domain transform coefficients based on
    intensity differences (or summed color differences) and then performs forward/backward
    passes to filter the input src image.
    
    Parameters:
        guidance (np.ndarray): Guidance image as a float32 array in [0,1]. Can be single-channel
                               or multi-channel (e.g., RGB).
        src (np.ndarray): Input image to filter (float32, same shape as guidance or single-channel).
        sigmaSpatial (float): Controls the spatial extent of the filter.
        sigmaColor (float): Controls how strongly color differences affect the filtering.
        num_iterations (int): Number of recursive iterations (more iterations approximate a Gaussian).
    
    Returns:
        np.ndarray: The filtered image (same shape as src).
    """
    H, W = guidance.shape[:2]

    def compute_diff_x(g):
        diff = np.zeros((H, W), dtype=np.float32)
        for i in range(H):
            for j in range(1, W):
                if g.ndim == 3:
                    # Sum differences over channels
                    diff[i, j] = np.sum(np.abs(g[i, j] - g[i, j-1]))
                else:
                    diff[i, j] = abs(g[i, j] - g[i, j-1])
        return diff

    def compute_diff_y(g):
        diff = np.zeros((H, W), dtype=np.float32)
        for i in range(1, H):
            for j in range(W):
                if g.ndim == 3:
                    diff[i, j] = np.sum(np.abs(g[i, j] - g[i-1, j]))
                else:
                    diff[i, j] = abs(g[i, j] - g[i-1, j])
        return diff

    # --- Horizontal Filtering ---
    # Compute horizontal differences and coefficients
    dI_dx = compute_diff_x(guidance)
    dt_x = 1 + dI_dx / sigmaColor  # Domain transform along x
    a_x = np.exp(- (np.sqrt(2) / sigmaSpatial) * dt_x)
    
    # Initialize result with src (make a copy so as not to alter original data)
    result = src.copy()

    # Apply recursive filtering horizontally (for each iteration, do a forward and backward pass)
    for _ in range(num_iterations):
        # Forward pass (left to right)
        for i in range(H):
            for j in range(1, W):
                result[i, j] = a_x[i, j] * result[i, j-1] + (1 - a_x[i, j]) * result[i, j]
        # Backward pass (right to left)
        for i in range(H):
            for j in range(W-2, -1, -1):
                result[i, j] = a_x[i, j+1] * result[i, j+1] + (1 - a_x[i, j+1]) * result[i, j]
    
    # --- Vertical Filtering ---
    # Compute vertical differences and coefficients
    dI_dy = compute_diff_y(guidance)
    dt_y = 1 + dI_dy / sigmaColor  # Domain transform along y
    a_y = np.exp(- (np.sqrt(2) / sigmaSpatial) * dt_y)
    
    # Apply recursive filtering vertically
    for _ in range(num_iterations):
        # Forward pass (top to bottom)
        for j in range(W):
            for i in range(1, H):
                result[i, j] = a_y[i, j] * result[i-1, j] + (1 - a_y[i, j]) * result[i, j]
        # Backward pass (bottom to top)
        for j in range(W):
            for i in range(H-2, -1, -1):
                result[i, j] = a_y[i+1, j] * result[i+1, j] + (1 - a_y[i+1, j]) * result[i, j]
    
    return result

# --- Example usage ---
if __name__ == "__main__":


    # Load an image as guidance and src (for demonstration, use the same image)
    dirname = os.path.dirname(__file__)
    guidance_bgr = cv2.imread(os.path.join(dirname, 'MEFDatabase/source image sequences/Farmhouse_hdr-project.com/DSC03641.png'))  # Replace with your image path
    if guidance_bgr is None:
        raise FileNotFoundError("Image not found. Check the provided path.")
    
    # Convert to RGB and then normalize to [0,1]
    guidance = cv2.cvtColor(guidance_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # For this example, assume we want to filter a single-channel weight map
    # We'll create a synthetic noisy weight map for demonstration
    weight_map = np.random.rand(guidance.shape[0], guidance.shape[1]).astype(np.float32)

    # Apply our domain transform filter
    smooth_weight = dt_filter(guidance, weight_map, sigmaSpatial=60, sigmaColor=0.4, num_iterations=3)

    # Display original and smoothed weight maps
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(weight_map, cmap='gray')
    plt.title("Original Weight Map")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(smooth_weight, cmap='gray')
    plt.title("Smoothed Weight Map (Domain Transform)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
