import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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

def show_results(input_images, results):
    """Display the input images and the fused results side-by-side."""
    n = len(input_images)
    plt.figure(figsize=(16, 10))

    # Show input images on the first row
    for i, img in enumerate(input_images):
        plt.subplot(2, n, i + 1)
        plt.imshow(img)
        plt.title(f'Input {i+1}')
        plt.axis('off')
    
    for j, (title, fused_img) in enumerate(results):
        plt.subplot(2, len(results), len(results) + j + 1)
        plt.imshow(fused_img)
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
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



if __name__ == '__main__':
    folder = "/home/emanuele/Documenti/Università/Signal /Signal_Project/MEFDatabase/source image sequences/Chinese_garden_Bartlomiej Okonek/"
    images = load_images_from_folder_distanced(folder,1)

    
    # Apply fusion techniques
    fused_avg = average_fusion(images)
    fused_mertens = mertens_fusion(images)
    fused_laplacian = laplacian_pyramid_fusion(images, levels=6)
    fused_exposure_compensation = exposure_compensation_fusion(images)
    fused_exposure_fusion = exposure_fusion(images)
    res = [fused_avg,fused_mertens,fused_laplacian, fused_exposure_compensation, fused_exposure_fusion]
    
    # Display the results for comparison
    show_results(images, res)
