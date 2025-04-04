import imgTransformations
import cv2
import matplotlib.pyplot as plt
import os
from ipywidgets import (
    FloatSlider,
    IntSlider,
    Output,
    VBox,
    Tab,
    interactive,
    Button,
    Label,
    SelectionSlider,
    Dropdown,
)
import ipywidgets as widgets
import textwrap
import utils
import numpy as np
from IPython.display import clear_output


def interactive_image_selector(folder):
    global images  # Access global images list in the main script

    entries = os.listdir(folder)
    subfolders = [
        entry for entry in entries if os.path.isdir(os.path.join(folder, entry))
    ]

    # Function to load and display images when a subfolder is selected
    def interactive_image_select(subfolder):
        global images  # Access global images list
        imgSet = os.path.join(folder, subfolder)
        images = utils.load_images_from_folder(imgSet)  # Update the global images list

        # Number of images per row
        images_per_row = 3
        num_inputs = len(images)

        # Calculate the number of rows needed for inputs and results
        input_rows = (num_inputs + images_per_row - 1) // images_per_row

        # Create the figure and axes
        fig, axes = plt.subplots(input_rows, images_per_row, figsize=(20, 20))
        axes = axes.flatten()  # Flatten to easily iterate over all axes

        # Plot input images
        for i, img in enumerate(images):
            ax = axes[i]
            ax.imshow(img)
            ax.set_title(f"Input {i + 1}")
            ax.axis("off")

        plt.show()

    # Create the interactive widget for selecting subfolders
    selectionWidget = interactive(
        interactive_image_select,
        subfolder=Dropdown(
            options=subfolders,
            value=subfolders[0],  # Default selected value (first subfolder)
            description="Image set",
            disabled=False,
        ),
    )

    display(selectionWidget)


def plotColourChannels(image):
    r, b, g = cv2.split(image)
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    plt.plot(hist_r, color="red", label="Red Channel")
    plt.plot(hist_g, color="green", label="Green Channel")
    plt.plot(hist_b, color="blue", label="Blue Channel")
    plt.title("RGB Histograms")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()


def interactive_fusion(fusion_method, *args, **kwargs):
    """
    Generalized interactive fusion function.

    Parameters:
        fusion_method (callable): The fusion method from imgTransformations.
        *args: Positional arguments for the fusion method.
        **kwargs: Keyword arguments for the fusion method.
    """
    title, fused = fusion_method(images, *args, **kwargs)

    # Generate additional title information from keyword arguments
    extra_info = ", ".join(f"{key}: {value}" for key, value in kwargs.items())
    if extra_info:
        title = f"{title} ({extra_info})"

    # Plotting
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(fused)
    plt.title(title)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plotColourChannels(fused)
    plt.show()


def showcase_methods_tab():
    # --------------------------
    # Create interactive widgets
    # --------------------------

    epsilon_values = np.round(np.logspace(-7, -5, num=10), decimals=10)

    # Ensure the default value is in the list
    default_epsilon = 1e-6
    if default_epsilon not in epsilon_values:
        epsilon_values = np.append(epsilon_values, default_epsilon)
        epsilon_values = np.sort(epsilon_values)  # Keep sorted order

    laplacian_widget = interactive(
        lambda levels: interactive_fusion(
            imgTransformations.laplacian_pyramid_fusion, levels=levels
        ),
        levels=IntSlider(min=1, max=10, step=1, value=4, description="Levels:"),
    )
    enhanced_widget = interactive(
        lambda sigma, epsilon, kernel_dim: interactive_fusion(
            imgTransformations.enhanced_exposure_fusion,
            sigma=sigma,
            epsilon=epsilon,
            blur_kernel=(kernel_dim, kernel_dim),
        ),
        sigma=FloatSlider(min=0.1, max=1.0, step=0.05, value=0.2, description="Sigma:"),
        epsilon=SelectionSlider(
            options=[(f"{e:.1e}", e) for e in epsilon_values],
            value=default_epsilon,
            description="Epsilon:",
        ),
        kernel_dim=IntSlider(
            min=1, max=9, step=2, value=3, description="Kernel dimension:"
        ),
    )
    domain_widget = interactive(
        lambda sigmaSpatial, sigmaColor, epsilon: interactive_fusion(
            imgTransformations.domain_transform_fusion,
            sigmaSpatial=sigmaSpatial,
            sigmaColor=sigmaColor,
            epsilon=epsilon,
            homebrew_dt=False,
        ),
        sigmaSpatial=FloatSlider(
            min=10, max=100, step=5, value=60, description="sigmaSpatial:"
        ),
        sigmaColor=FloatSlider(
            min=0.1, max=1.0, step=0.05, value=0.4, description="sigmaColor:"
        ),
        epsilon=SelectionSlider(
            options=[(f"{e:.1e}", e) for e in epsilon_values],
            value=default_epsilon,
            description="Epsilon:",
        ),
    )
    wavelet_widget = interactive(
        lambda level: interactive_fusion(
            imgTransformations.wavelet_fusion, wavelet="db1", level=level
        ),
        level=IntSlider(min=1, max=5, step=1, value=2, description="Level:"),
    )

    # For functions without adjustable numeric parameters, use buttons.
    avg_button = Button(description="Run Average Fusion")
    mertens_button = Button(description="Run Mertens Fusion")
    exposure_button = Button(description="Run Exposure Fusion")
    exp_comp_button = Button(description="Run Exposure Compensation Fusion")

    out_avg = Output()
    out_mertens = Output()
    out_exposure = Output()
    out_exp_comp = Output()

    def run_fusion(method, output):
        with output:
            output.clear_output(wait=True)
            interactive_fusion(method)

    avg_button.on_click(
        lambda b: run_fusion(imgTransformations.average_fusion, out_avg)
    )
    mertens_button.on_click(
        lambda b: run_fusion(imgTransformations.mertens_fusion, out_mertens)
    )
    exposure_button.on_click(
        lambda b: run_fusion(imgTransformations.exposure_fusion, out_exposure)
    )
    exp_comp_button.on_click(
        lambda b: run_fusion(
            imgTransformations.exposure_compensation_fusion, out_exp_comp
        )
    )

    # --------------------------
    # Layout the widgets in a Tabbed interface
    # --------------------------
    tab = Tab(
        children=[
            VBox([Label("Average Fusion (no parameters):"), avg_button, out_avg]),
            VBox(
                [Label("Mertens Fusion (no parameters):"), mertens_button, out_mertens]
            ),
            VBox([Label("Laplacian Pyramid Fusion:"), laplacian_widget]),
            VBox(
                [
                    Label("Exposure Fusion (no parameters):"),
                    exposure_button,
                    out_exposure,
                ]
            ),
            VBox(
                [
                    Label("Exposure Compensation Fusion (no parameters):"),
                    exp_comp_button,
                    out_exp_comp,
                ]
            ),
            VBox([Label("Enhanced Exposure Fusion:"), enhanced_widget]),
            VBox([Label("Domain Transform Fusion:"), domain_widget]),
            VBox([Label("Wavelet Fusion:"), wavelet_widget]),
        ]
    )
    tab.set_title(0, "Average")
    tab.set_title(1, "Mertens")
    tab.set_title(2, "Laplacian")
    tab.set_title(3, "Exposure")
    tab.set_title(4, "Exp. Comp.")
    tab.set_title(5, "Enhanced")
    tab.set_title(6, "Domain Trans.")
    tab.set_title(7, "Wavelet")

    display(tab)
