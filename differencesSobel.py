import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
import textwrap
import tranformations  
import utils

def show_results(input_images, results, images_per_row=3, figsize=(16, 10), grayscale=False):
    """
    Display the input images and the fused results in a flexible grid layout,
    with a clear division between input images and fused results.

    Parameters:
        input_images (list of np.ndarray): List of input images to display.
        results (list of tuple): List of tuples, each containing a title (str) and a fused image (np.ndarray).
        images_per_row (int): Number of images to display per row.
        figsize (tuple): Size of the entire figure (width, height).
    """
    # Number of input and result images
    num_inputs = len(input_images)
    num_results = len(results)
    
    # Calculate the number of rows needed for inputs and results
    input_rows = (num_inputs + images_per_row - 1) // images_per_row
    result_rows = (num_results + images_per_row - 1) // images_per_row
    
    # Create the figure and axes
    fig, axes = plt.subplots(input_rows + result_rows, images_per_row, figsize=figsize)
    axes = axes.flatten()  # Flatten to easily iterate over all axes
    
    # Plot input images
    for i, img in enumerate(input_images):
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(f'Input {i + 1}')
        ax.axis('off')
    
    # Plot fused results
    for j, (title, fused_img) in enumerate(results):
        ax = axes[num_inputs + j]
        ax.imshow(fused_img)
        #add text wrap to title
        ax.set_title("\n".join(textwrap.wrap(title, 30)))
        ax.axis('off')
    
    # Hide any remaining empty subplots
    for k in range(num_inputs + num_results, len(axes)):
        axes[k].axis('off')
    
    # Add a horizontal line to separate input images from fused results
    if num_results > 0:
        # Calculate the y-coordinate for the horizontal line
        # This is the normalized coordinate in figure space
        y_sep = 1 - (input_rows / (input_rows + result_rows))
        fig.add_artist(plt.Line2D([0, 1], [y_sep, y_sep], color='black', linewidth=2, transform=fig.transFigure, clip_on=False))
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    folder = os.path.join(dirname, 'MEFDatabase/source image sequences/Chinese_garden_Bartlomiej Okonek/')
    images = utils.load_images_from_folder_distanced(folder,1)

    
    # Apply fusion techniques
    res = [
        tranformations.average_fusion(images),
        tranformations.mertens_fusion(images),
        tranformations.laplacian_pyramid_fusion(images, levels=6),
        tranformations.exposure_compensation_fusion(images),
        tranformations.exposure_fusion(images),
        tranformations.enhanced_exposure_fusion(images, sigma=0.2, epsilon=1e-12, blur_kernel=(3,3)),
        #tranformations.wavelet_fusion(images, wavelet='db1', level=2) #SEEMS VERY VERY HARD ON THE COMPUTER,
        tranformations.domain_transform_fusion(images, sigmaSpatial=60, sigmaColor=0.4, epsilon=1e-6, homebrew_dt=False),
        tranformations.domain_transform_fusion(images, sigmaSpatial=60, sigmaColor=0.4, epsilon=1e-6, homebrew_dt=True)
    ]
    
    
    # Display the results for comparison
    show_results(images, res)

    # Horizontal edge detection
    res_sobel = []
    for item in res:
        str, img = item
        res_sobel.append((str, cv2.Sobel(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),cv2.CV_64F,0,1,ksize=9)))
    show_results(images, res_sobel)

    # Vertical edge detection
    res_sobel = []
    for item in res:
        str, img = item
        res_sobel.append((str, cv2.Sobel(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),cv2.CV_64F,1,0,ksize=9)))
    show_results(images, res_sobel)
