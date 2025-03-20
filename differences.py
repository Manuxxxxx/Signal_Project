import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
import textwrap

def load_images(image_paths):
    """Load images and convert them from BGR to RGB for display."""
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)
    return images

def average_fusion(images):
    """Fuse images by taking the simple average."""
    images_float = [img.astype(np.float32) for img in images]
    fused = np.mean(images_float, axis=0)
    return ('Simple Average',np.clip(fused, 0, 255).astype(np.uint8))

def mertens_fusion(images):
    """Fuse images using OpenCV's MergeMertens exposure fusion method."""
    if not images:
        raise ValueError("Image list is empty")
    
    # Ensure all images are of the same shape
    h, w, c = images[0].shape
    for img in images:
        if img.shape != (h, w, c):
            raise ValueError("All images must have the same dimensions and channels")
    
    # Create Mertens fusion object
    merge_mertens = cv2.createMergeMertens()
    
    # Perform exposure fusion
    fused = merge_mertens.process(images)

    # Convert back to uint8 for display
    fused = np.clip(fused * 255, 0, 255).astype(np.uint8)

    return ('Mertens Fusion',fused)

def laplacian_pyramid_fusion(images, levels=4):
    """Fuse images using Laplacian Pyramid Fusion.
       Assumes all images are the same size.
    """
    # Build Gaussian pyramids for each image
    gaussian_pyramids = []
    for img in images:
        gp = [img.astype(np.float32)]
        for i in range(levels):
            img = cv2.pyrDown(img)
            gp.append(img.astype(np.float32))
        gaussian_pyramids.append(gp)

    # Build Laplacian pyramids for each image
    laplacian_pyramids = []
    for gp in gaussian_pyramids:
        lp = []
        for i in range(len(gp) - 1):
            size = (gp[i].shape[1], gp[i].shape[0])
            GE = cv2.pyrUp(gp[i+1], dstsize=size)
            L = gp[i] - GE
            lp.append(L)
        lp.append(gp[-1])
        laplacian_pyramids.append(lp)

    # Fuse Laplacian pyramids by averaging each level
    fused_pyramid = []
    for level in range(levels + 1):
        # Stack corresponding level of each pyramid and take mean
        layer_stack = np.array([lp[level] for lp in laplacian_pyramids])
        fused_layer = np.mean(layer_stack, axis=0)
        fused_pyramid.append(fused_layer)

    # Reconstruct the fused image from the fused pyramid
    fused_image = fused_pyramid[-1]
    for i in range(levels, 0, -1):
        size = (fused_pyramid[i-1].shape[1], fused_pyramid[i-1].shape[0])
        fused_image = cv2.pyrUp(fused_image, dstsize=size)
        fused_image = fused_image + fused_pyramid[i-1]

    return (('Laplacian Pyramid - levels:'+str(levels)),np.clip(fused_image, 0, 255).astype(np.uint8))

def exposure_fusion(images):
    # Convert images to float32 for processing
    images = [img.astype(np.float32) / 255.0 for img in images]

    # Compute Laplacian contrast weight (ensure it's applied to grayscale images)
    laplacian_weight = [cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F) for img in images]
    laplacian_weight = [np.abs(lap) for lap in laplacian_weight]
    laplacian_weight = [np.repeat(lap[:, :, np.newaxis], 3, axis=2) for lap in laplacian_weight]  # Convert to 3-channel

    # Compute saliency weight (make sure this is in the same shape as the image)
    saliency_weight = [cv2.saliency.StaticSaliencySpectralResidual_create().computeSaliency(img)[1] for img in images]
    saliency_weight = [np.repeat(saliency[:, :, np.newaxis], 3, axis=2) for saliency in saliency_weight]  # Convert to 3-channel

    # Compute exposure weight (ensure it's applied to RGB images)
    exposure_weight = [np.exp(-((img - 0.5) ** 2) / (2 * 0.2 ** 2)) for img in images]
    
    # Combine weights
    total_weight = sum(laplacian_weight) + sum(saliency_weight) + sum(exposure_weight)
    total_weight += 1e-12  # Prevent division by zero

    weighted_images = [(w / total_weight) * img for w, img in zip(laplacian_weight, images)]
    
    # Perform fusion
    fused_image = sum(weighted_images)

    # Convert back to uint8
    fused_image = (fused_image * 255).astype(np.uint8)

    return ("Exposure Fusion",fused_image)

def show_results(input_images, results, images_per_row=3, figsize=(10, 7)):
    """
    Display each fused result in a dedicated window, with input images shown in each window.

    Parameters:
        input_images (list of np.ndarray): List of input images to display.
        results (list of tuple): List of tuples, each containing a title (str) and a fused image (np.ndarray).
        images_per_row (int): Number of images to display per row.
        figsize (tuple): Size of each figure (width, height).
    """
    num_inputs = len(input_images)
    saturation_values = []

    for j, (title, fused_img) in enumerate(results):
        # Calculate the number of rows needed for inputs and the single result
        input_rows = (num_inputs + images_per_row - 1) // images_per_row

        # Create the figure and axes for this result
        fig, axes = plt.subplots(input_rows + 3, images_per_row, figsize=figsize)
        axes = axes.flatten()  # Flatten to easily iterate over all axes

        # Plot input images
        for i, img in enumerate(input_images):
            ax = axes[i]
            ax.imshow(img)
            ax.set_title(f'Input {i + 1}')
            ax.axis('off')

        # Calculate the index for the centered subplot in the bottom row
        result_index = num_inputs + images_per_row // 2

        # Plot the fused result in the centered subplot of the bottom row
        ax = axes[result_index]
        ax.imshow(fused_img)
        ax.set_title("\n".join(textwrap.wrap(title, 30)))
        ax.axis('off')

        # Plot color histograms for the fused image
        for c, color in enumerate(['Red', 'Green', 'Blue']):
            hist_index = result_index + images_per_row + c
            ax = axes[hist_index]
            ax.hist(fused_img[:, :, c].ravel(), bins=256, color=color.lower(), alpha=0.7)
            ax.set_title(f'{color} Channel Histogram')
            ax.set_xlim(0, 256)

        # Calculate saturation for the fused image and store it
        hsv_img = cv2.cvtColor(fused_img, cv2.COLOR_RGB2HSV)
        saturation = np.mean(hsv_img[:, :, 1])  # Saturation is the second channel in HSV
        saturation_values.append(saturation)

        # Hide any remaining empty subplots
        for k in range(len(axes)):
            if k >= num_inputs and k != result_index and k not in [result_index + images_per_row, result_index + images_per_row + 1, result_index + images_per_row + 2]:
                axes[k].axis('off')

        # Add a horizontal line to separate input images from the fused result
        y_sep = 1 - (input_rows / (input_rows + 3))
        fig.add_artist(plt.Line2D([0, 1], [y_sep, y_sep], color='black', linewidth=2, transform=fig.transFigure, clip_on=False))

        plt.tight_layout()
        plt.show()

    # Plot saturation comparison in a separate window
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(saturation_values)), saturation_values, color='purple', alpha=0.7)
    plt.xticks(range(len(saturation_values)), [f'Result {i+1}' for i in range(len(saturation_values))])
    plt.title('Saturation Comparison of All Results')
    plt.xlabel('Result')
    plt.ylabel('Saturation')
    plt.ylim(0, max(saturation_values) + 10)  # Adjust y-axis limit for better visualization
    plt.show()

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def load_images_from_folder_min_max(folder):
    images = []
    min = None
    max = None
    for filename in os.listdir(folder):
        if min is None:
            min = filename
        if max is None:
            max = filename
        if filename < min:
            min = filename
        if filename > max:
            max = filename
    images = [cv2.imread(os.path.join(folder,min)),cv2.imread(os.path.join(folder,max))]
    return images

def load_images_from_folder_distanced(folder, distance):
    images = []
    for i in range(0, len(os.listdir(folder)), distance):
        img = cv2.imread(os.path.join(folder,os.listdir(folder)[i]))
        if img is not None:
            images.append(img)
    return images

def exposure_compensation_fusion(images):
    """
    Perform exposure compensation-based fusion on a list of images.
    
    Parameters:
        images (list of numpy.ndarray): List of input images with different exposures.
        
    Returns:
        numpy.ndarray: Fused image.
    """
    # Convert images to grayscale for luminance computation
    luminances = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    
    # Calculate average luminance across all images
    avg_luminance = np.mean(luminances, axis=0)
    
    # Compute gain factors for each image (avoid division by zero)
    gain_factors = [avg_luminance / (lum + 1e-6) for lum in luminances]
    
    # Apply gain factors with NumPy multiplication (after converting to float)
    compensated_images = []
    for img, gf in zip(images, gain_factors):
        # Convert image to float for multiplication
        img_float = img.astype(np.float32)
        # Expand gain factor dimensions and convert to float32
        gf_expanded = gf[:, :, np.newaxis].astype(np.float32)
        # Multiply and clip the result
        compensated = np.clip(img_float * gf_expanded, 0, 255).astype(np.uint8)
        compensated_images.append(compensated)
    
    # Fuse images by averaging pixel values
    fused_image = np.mean(np.stack(compensated_images, axis=0), axis=0).astype(np.uint8)
    
    return ('Exposure Compensation',fused_image)

def enhanced_exposure_fusion(images, sigma=0.2, epsilon=1e-6, blur_kernel=(5,5)):
    """
    Fuse images using an enhanced exposure fusion approach with weight smoothing.
    This method computes weight maps based on three factors:
      - Contrast (using Laplacian of the grayscale image)
      - Saturation (standard deviation across RGB channels)
      - Well-exposedness (Gaussian function centered at 0.5)
    A Gaussian blur is applied to weight maps to reduce artifacts (such as black spots).
    
    Parameters:
        images (list of numpy.ndarray): Input images in the 0-255 range.
        sigma (float): Parameter for the well-exposedness weight.
        epsilon (float): Small constant to avoid division by zero.
        blur_kernel (tuple): Kernel size for Gaussian blur applied on weight maps.
    
    Returns:
        tuple: (title, fused_image) where fused_image is the final fused image.
    """
    # Normalize images to [0, 1]
    imgs = [img.astype(np.float32) / 255.0 for img in images]
    
    weight_maps = []
    for img in imgs:
        # Contrast weight: absolute Laplacian of the grayscale image
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        contrast = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        
        # Saturation weight: standard deviation across color channels
        saturation = np.std(img, axis=2)
        
        # Well-exposedness weight: Gaussian function per channel
        well_exposedness = np.exp(-0.5 * ((img - 0.5) / sigma) ** 2)
        # Combine well-exposedness across channels by taking the product
        well_exposedness = np.prod(well_exposedness, axis=2)
        
        # Combine weights, adding a small constant to avoid zeros
        weight = (contrast + epsilon) * (saturation + epsilon) * (well_exposedness + epsilon)
        # Smooth the weight map to reduce abrupt transitions (black spots)
        weight = cv2.GaussianBlur(weight, blur_kernel, 0)
        weight_maps.append(weight)
    
    # Normalize the weight maps so that they sum to 1 at each pixel
    weight_sum = np.sum(np.array(weight_maps), axis=0) + epsilon
    normalized_weights = [w / weight_sum for w in weight_maps]
    
    # Fuse the images using the normalized weight maps
    fused = np.zeros_like(imgs[0])
    for img, w in zip(imgs, normalized_weights):
        # Expand weight map to 3 channels for multiplication
        w3 = np.repeat(w[:, :, np.newaxis], 3, axis=2)
        fused += img * w3

    # Convert back to uint8 in the range [0, 255]
    fused_image = np.clip(fused * 255, 0, 255).astype(np.uint8)
    
    return ("Enhanced Exposure Fusion (Smoothed Weights)", fused_image)

def domain_transform_fusion(images, sigmaSpatial=60, sigmaColor=0.4, epsilon=1e-6, homebrew_dt=False):
    """
    Fuse images using Domain Transform filtering to refine weight maps.
    
    This method computes weight maps based on:
      - Contrast (via the absolute Laplacian on a grayscale image)
      - Saturation (standard deviation across color channels)
      - Well-exposedness (Gaussian function centered at 0.5)
      
    The computed weight maps are then smoothed using the Domain Transform filter
    (cv2.ximgproc.dtFilter) for improved edge-preservation before fusing the images.
    
    Parameters:
        images (list of numpy.ndarray): Input images in 0-255 range.
        sigmaSpatial (float): Spatial standard deviation for the domain transform filter.
        sigmaColor (float): Color standard deviation for the domain transform filter.
        epsilon (float): Small constant to avoid division by zero.
    
    Returns:
        tuple: (title, fused_image) where fused_image is the final fused result.
    """
    # Normalize images to [0, 1]
    imgs = [img.astype(np.float32) / 255.0 for img in images]
    weight_maps = []
    desc = None
    
    for img in imgs:
        # Compute contrast weight: use absolute Laplacian on grayscale image
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        contrast = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        
        # Compute saturation weight: standard deviation across color channels
        saturation = np.std(img, axis=2)
        
        # Compute well-exposedness weight: product of Gaussian responses for each channel
        well_exposedness = np.exp(-0.5 * ((img - 0.5) / 0.2) ** 2)
        well_exposedness = np.prod(well_exposedness, axis=2)
        
        # Combine the weights with a small epsilon to avoid zeros
        weight = (contrast + epsilon) * (saturation + epsilon) * (well_exposedness + epsilon)
        
        # Smooth the weight map using Domain Transform filtering.
        # The guide image can be the original color image.
        # Mode 1 (DTF_RF) applies recursive filtering.
        if(homebrew_dt):
            desc = "Simple Domain Transform"
            guidance = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            weight_map = np.random.rand(guidance.shape[0], guidance.shape[1]).astype(np.float32)
            smooth_weight = dt_filter(guidance, weight_map, sigmaSpatial, sigmaColor, num_iterations=3)
        else:
            desc ="OpenCV Domain Transform"
            smooth_weight = cv2.ximgproc.dtFilter(img, weight.astype(np.float32), sigmaSpatial, sigmaColor, mode=1)
        
        weight_maps.append(smooth_weight)
    
    # Normalize the weight maps so that they sum to 1 at every pixel
    weight_sum = np.sum(np.array(weight_maps), axis=0) + epsilon
    normalized_weights = [w / weight_sum for w in weight_maps]
    
    # Fuse the images using the normalized weight maps
    fused = np.zeros_like(imgs[0])
    for img, w in zip(imgs, normalized_weights):
        # Expand weight map to three channels for element-wise multiplication
        w3 = np.repeat(w[:, :, np.newaxis], 3, axis=2)
        fused += img * w3

    # Convert back to 0-255 range and uint8
    fused_image = np.clip(fused * 255, 0, 255).astype(np.uint8)
    
    return ("Domain Transform Fusion"+" - "+desc, fused_image)

def wavelet_fusion(images, wavelet='db1', level=2):
    """
    Fuse images using wavelet transform-based fusion.
    
    Steps:
    1. Decompose each image using discrete wavelet transform.
    2. Fuse coefficients using a chosen rule (e.g., max-abs for detail coefficients,
       averaging for approximation coefficients).
    3. Reconstruct the fused image using the inverse wavelet transform.
    
    Parameters:
        images (list of numpy.ndarray): List of images in the 0-255 range.
        wavelet (str): Wavelet type.
        level (int): Decomposition level.
        
    Returns:
        tuple: (title, fused_image)
    """
    # Convert images to float32 and normalize to [0,1]
    imgs = [img.astype(np.float32) / 255.0 for img in images]
    coeffs_list = []
    # Decompose each image
    for img in imgs:
        coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
        coeffs_list.append(coeffs)
        
    # Fuse coefficients (assumes all images have the same decomposition structure)
    fused_coeffs = []
    # Fuse the approximation coefficients using average
    fused_approx = np.mean([coeffs[0] for coeffs in coeffs_list], axis=0)
    fused_coeffs.append(fused_approx)
    
    # Fuse detail coefficients for each level
    for level_idx in range(1, level + 1):
        fused_details = []
        # Each level has 3 sets of detail coefficients (horizontal, vertical, diagonal)
        for i in range(3):
            detail_coeffs = [coeffs[level_idx][i] for coeffs in coeffs_list]
            # Use the max-absolute rule: choose coefficient with highest absolute value
            detail_coeffs = np.array(detail_coeffs)
            fused_detail = detail_coeffs[np.argmax(np.abs(detail_coeffs), axis=0)]
            fused_details.append(fused_detail)
        fused_coeffs.append(tuple(fused_details))
        
    # Reconstruct the fused image
    fused_img = pywt.waverec2(fused_coeffs, wavelet=wavelet)
    # Clip and convert back to uint8 in the range [0,255]
    fused_img = np.clip(fused_img, 0, 1)
    fused_img = (fused_img * 255).astype(np.uint8)
    
    return ("Wavelet Fusion", fused_img)

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


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    folder = os.path.join(dirname, 'MEFDatabase/source image sequences/Chinese_garden_Bartlomiej Okonek/')
    images = load_images_from_folder_distanced(folder,1)

    
    # Apply fusion techniques
    res = [
        average_fusion(images),
        mertens_fusion(images),
        laplacian_pyramid_fusion(images, levels=6),
        exposure_compensation_fusion(images),
        exposure_fusion(images),
        enhanced_exposure_fusion(images, sigma=0.2, epsilon=1e-12, blur_kernel=(3,3)),
        #wavelet_fusion(images, wavelet='db1', level=2) #SEEMS VERY VERY HARD ON THE COMPUTER,
        domain_transform_fusion(images, sigmaSpatial=60, sigmaColor=0.4, epsilon=1e-6, homebrew_dt=False),
        domain_transform_fusion(images, sigmaSpatial=60, sigmaColor=0.4, epsilon=1e-6, homebrew_dt=True)
    ];
    
    
    # Display the results for comparison
    show_results(images, res)
