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
import pandas as pd
from IPython.display import clear_output


def interactive_image_selector(folder, images_list):
    # Remove the global declaration

    entries = os.listdir(folder)
    subfolders = [
        entry for entry in entries if os.path.isdir(os.path.join(folder, entry))
    ]

    start_entry = None
    for entry in entries:
        if entry.startswith("Chinese"):
            start_entry = entry
            break
    if start_entry is None:
        start_entry = subfolders[0]

    def interactive_image_select(subfolder):
        imgSet = os.path.join(folder, subfolder)
        loaded_images = utils.load_images_from_folder(imgSet)
        # Update the images_list in-place
        images_list.clear()
        images_list.extend(loaded_images)

        # Number of images per row
        images_per_row = 3
        num_inputs = len(images_list)

        # Calculate the number of rows needed for inputs and results
        input_rows = (num_inputs + images_per_row - 1) // images_per_row

        # Create the figure and axes
        fig, axes = plt.subplots(input_rows, images_per_row, figsize=(20, 20))
        axes = axes.flatten()  # Flatten to easily iterate over all axes

        # Plot input images
        for i, img in enumerate(images_list):
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
            value=start_entry,  # Default selected value (first subfolder)
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

def interactive_fusion(images,fusion_method, *args, **kwargs):
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

def showcase_methods_tab(images, methods):
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
            images,methods['Laplacian Pyramid'],
            levels=levels
        ),
        levels=IntSlider(min=1, max=10, step=1, value=4, description="Levels:"),
    )

    enhanced_widget = interactive(
        lambda sigma, epsilon, kernel_dim: interactive_fusion(
            images,
            methods['Enhanced Exposure'],
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
            images,
            methods['Domain Transform'],
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
            images,
            methods['Wavelet Fusion'],
            wavelet="db1",
            level=level
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
            interactive_fusion(images=images,fusion_method=method)

    avg_button.on_click(
        lambda b: run_fusion(
            methods['Average Fusion'],
            out_avg
        )
    )
    mertens_button.on_click(
        lambda b: run_fusion(
            methods['Mertens Fusion'],
            out_mertens
        )
    )
    exposure_button.on_click(
        lambda b: run_fusion(
            methods['Exposure Fusion'],
            out_exposure
        )
    )
    exp_comp_button.on_click(
        lambda b: run_fusion(
            methods['Exposure Compensation'],
            out_exp_comp
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

def showcase_weight_maps_tab(images, calculate_weight_maps):
    out = Output()
    def show_weight_maps(img, index):
        weight, smooth_weight_homebrew, smooth_weight_opencv = calculate_weight_maps(img)
        with out:
            out.clear_output(wait=True)
            plt.figure(figsize=(18, 5))

            # Original Weight Map
            plt.subplot(1, 3, 1)
            plt.imshow(weight, cmap='gray')
            plt.title(f"Original Weight Map (Image {index})")
            plt.axis('off')

            # Homebrew Domain Transform
            plt.subplot(1, 3, 2)
            plt.imshow(smooth_weight_homebrew, cmap='gray')
            plt.title(f"Smoothed Weight Map (Homebrew DT, Image {index})")
            plt.axis('off')

            # OpenCV dtFilter
            plt.subplot(1, 3, 3)
            plt.imshow(smooth_weight_opencv, cmap='gray')
            plt.title(f"Smoothed Weight Map (OpenCV dtFilter, Image {index})")
            plt.axis('off')

            plt.tight_layout()
            plt.show()

    # Create buttons dynamically
    buttons = []
    #add counter to for loop
    for i, image in enumerate(images):
        btn = Button(description=f"Show Image {i+1}")
        btn.on_click(lambda b, img=image, index=i: show_weight_maps(img, index))
        buttons.append(btn)

    # Display buttons and output widget
    display(VBox(buttons + [out]))


def compare_methods_and_metrics(spatial_frequency, calculate_entropy, calculate_metrics, images, methods):

    # Create checkboxes for method selection
    method_checkboxes = [widgets.Checkbox(description=name, value=False) for name, _ in methods.items()]
    checkboxes_box = widgets.VBox([widgets.Label("Select Fusion Methods:")] + method_checkboxes)

    # Create compare button
    compare_button = widgets.Button(description="Compare Selected Methods")
    output = widgets.Output()
    def on_compare_button_clicked(b, methods=methods):
        with output:
            clear_output(wait=True)
            selected_methods = [(name, func) for (name, func), checkbox in zip(methods.items(), method_checkboxes) if checkbox.value]

            if len(selected_methods) < 2:
                print("Please select at least 2 methods!")
                return
            
            # Generate fused images
            results = []
            for name, func in selected_methods:
                try:
                    if name == 'Domain Transform - Homebrew':
                        title, fused = func(images, homebrew_dt=True)                    
                    else:
                        title, fused = func(images)
                    results.append((title, fused))
                except Exception as e:
                    print(f"Error in {name}: {str(e)}")
                    return
            
            # Display fused images
            _, axes = plt.subplots(1, len(results), figsize=(20, 5))
            if len(results) == 1:
                axes = [axes]
                
            for ax, (title, img) in zip(axes, results):
                ax.imshow(img)
                ax.set_title(title)
                ax.axis('off')
            plt.show()
            
            # Calculate metrics
            num_images = len(results)
            similarity_matrix = np.zeros((num_images, num_images, 2))
            
            # Calculate individual metrics
            individual_metrics = []
            for title, img in results:
                entropy = calculate_entropy(img)
                sf = spatial_frequency(img)
                individual_metrics.append((entropy, sf))
            
            # Calculate pairwise metrics
            for i in range(num_images):
                for j in range(i, num_images):
                    ssim_val, psnr_val = calculate_metrics(results[i][1], results[j][1])
                    similarity_matrix[i, j] = (ssim_val, psnr_val)
                    similarity_matrix[j, i] = (ssim_val, psnr_val)
            
            # Display metrics
            df_individual = pd.DataFrame([(title, f"{entropy:.4f}", f"{sf:.4f}") 
                                        for (title, _), (entropy, sf) in zip(results, individual_metrics)],
                                    columns=['Method', 'Entropy', 'Spatial Frequency'])
            
            # Style individual metrics table
            individual_styler = (df_individual.style
                                .set_caption("Individual Image Metrics")
                                .set_properties(**{'text-align': 'left', 
                                                'font-size': '14px'})
                                .hide(axis='index')
                                .format(precision=4))
            display(individual_styler)

            # Create similarity matrices with NaN handling
            methods = [title for title, _ in results]
            
            ssim_matrix = np.zeros((len(methods), len(methods)))
            psnr_matrix = np.zeros((len(methods), len(methods)))
            for i in range(len(methods)):
                for j in range(len(methods)):
                    ssim_val, psnr_val = similarity_matrix[i, j]
                    ssim_matrix[i, j] = ssim_val
                    psnr_matrix[i, j] = psnr_val if not np.isinf(psnr_val) else np.nan

            # Create formatted DataFrames
            ssim_df = pd.DataFrame(ssim_matrix, index=methods, columns=methods)
            psnr_df = pd.DataFrame(psnr_matrix, index=methods, columns=methods)

            # Formatting functions with special handling
            def format_ssim(val):
                return f"{val:.3f}" if not np.isnan(val) else ""
                
            def format_psnr(val):
                if np.isinf(val):
                    return "∞"
                if np.isnan(val):
                    return "N/A"
                return f"{val:.2f}"

            # Create styled tables with titles
            print("\n")
            ssim_styler = (ssim_df.style
                        .set_caption("Structural Similarity Index (SSIM)")
                        .format(format_ssim)
                        .background_gradient(cmap='Blues', axis=None, 
                                            vmin=0, vmax=1)
                        .set_properties(**{'font-size': '12px'}))

            # Update the PSNR styler creation
            finite_psnr = psnr_df.replace([np.inf, -np.inf], np.nan)
            vmin = finite_psnr.min().min()
            vmax = finite_psnr.max().max()

            psnr_styler = (psnr_df.style
                        .set_caption("Peak Signal-to-Noise Ratio (PSNR)")
                        .format(format_psnr)
                        .background_gradient(cmap='Greens', axis=None,
                                            vmin=vmin, vmax=vmax,
                                            subset=pd.IndexSlice[:, :])
                        .set_properties(**{'font-size': '12px'}))
            
            display(ssim_styler)
            print("\n")
            display(psnr_styler)

    compare_button.on_click(on_compare_button_clicked)
    # Display the widgets
    display(widgets.VBox([checkboxes_box, compare_button, output]))


def show_images(image1, image2, title1="Image 1", title2="Image 2"):
    """Display two images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image1)
    axes[0].set_title(title1)
    axes[0].axis("off")
    
    axes[1].imshow(image2)
    axes[1].set_title(title2)
    axes[1].axis("off")
    
    plt.show()

def compare_methods_display(images, compute_difference, show_heatmap, methods):
    method_selector1 = widgets.Dropdown(options=methods.keys(), description="Method 1")
    method_selector2 = widgets.Dropdown(options=methods.keys(), description="Method 2")

    # Create an Output widget to manage displayed content
    output = widgets.Output()

    def compare_methods(method_name1, method_name2):
        with output:  # All display operations inside this block will go to `output`
            output.clear_output(wait=True)  # Clear previous output before showing new results

            if method_name1 == method_name2:
                print("Please select two different methods for comparison.")
                return
            
            # Get fused images from selected methods
            if method_name1 == 'Domain Transform - Homebrew':
                name1, img1 = methods[method_name1](images, homebrew_dt=True)
            else:
                name1, img1 = methods[method_name1](images)
            
            if method_name2 == 'Domain Transform - Homebrew':
                name2, img2 = methods[method_name2](images, homebrew_dt=True)
            else:
                name2, img2 = methods[method_name2](images)

            # Display the fused images
            show_images(img1, img2, name1, name2)
            
            # Compute and display the heatmap difference
            diff = compute_difference(img1, img2)
            show_heatmap(diff)

    # Display the widgets (dropdowns and button)
    display(method_selector1, method_selector2)
    button = widgets.Button(description="Compare")

    def on_button_click(b):
        compare_methods(method_selector1.value, method_selector2.value)

    button.on_click(on_button_click)
    display(button)
    display(output)  # Display the output area where results will appear