import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
import textwrap
import tranformations
import utils


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
        fig, axes = plt.subplots(input_rows + 2, images_per_row, figsize=figsize)
        axes = axes.flatten()  # Flatten to easily iterate over all axes

        # Plot input images
        for i, img in enumerate(input_images):
            ax = axes[i]
            ax.imshow(img)
            ax.set_title(f'Input {i + 1}')
            ax.axis('off')

        # Calculate the index for the centered subplot in the bottom row
        result_index = num_inputs

        # Plot the fused result in the centered subplot of the bottom row
        ax = axes[result_index]
        ax.imshow(fused_img)
        ax.set_title("\n".join(textwrap.wrap(title, 30)))
        ax.axis('off')

        # Plot color histograms for the fused image
        hist_start_index = result_index + images_per_row
        for c, color in enumerate(['Red', 'Green', 'Blue']):
            hist_index = hist_start_index + c
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
            if k >= num_inputs and k != result_index and k not in [hist_start_index, hist_start_index + 1, hist_start_index + 2]:
                axes[k].axis('off')

        # Add a horizontal line to separate input images from the fused result
        y_sep = 1 - (input_rows / (input_rows + 2))
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
    ];
    
    
    # Display the results for comparison
    show_results(images, res)
