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

def plotColourChannels(ax, image):
    r, b, g = cv2.split(image)
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    ax.plot(hist_r, color="red", label="Red Channel")
    ax.plot(hist_g, color="green", label="Green Channel")
    ax.plot(hist_b, color="blue", label="Blue Channel")
    ax.set_title("RGB Histograms")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.legend()

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
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].imshow(fused)
    axes[0].set_title(title)
    axes[0].axis("off")

    # Plot RGB histograms
    plotColourChannels(axes[1], fused)

def showcase_methods_tab(images, methods):
    # Create a persistent output container for all methods
    global_output = Output()
    
    # --------------------------
    # Create interactive widgets
    # --------------------------
    epsilon_values = np.round(np.logspace(-7, -5, num=10), decimals=10)
    default_epsilon = 1e-6
    if default_epsilon not in epsilon_values:
        epsilon_values = np.append(epsilon_values, default_epsilon)
        epsilon_values = np.sort(epsilon_values)

    # Common update function for all interactive methods
    def run_interactive(method, **kwargs):
        with global_output:
            global_output.clear_output(wait=True)
            try:
                title, fused = method(images, **kwargs)
                
                # Generate parameter info
                param_info = ", ".join(f"{k}: {v}" for k, v in kwargs.items())
                full_title = f"{title} ({param_info})" if param_info else title
                
                # Create plot within output context
                fig, axes = plt.subplots(1, 2, figsize=(16, 5))
                axes[0].imshow(fused)
                axes[0].set_title(full_title)
                axes[0].axis("off")
                plotColourChannels(axes[1], fused)
                plt.show()
                
            except Exception as e:
                print(f"Error: {str(e)}")

    # Laplacian Pyramid Widget
    levels_slider = IntSlider(min=1, max=10, step=1, value=4, description="Levels:")
    laplacian_box = VBox([
        Label("Laplacian Pyramid Fusion:"),
        levels_slider
    ])
    
    def update_laplacian(change):
        run_interactive(methods['Laplacian Pyramid'], levels=change['new'])
    levels_slider.observe(update_laplacian, names='value')

    # Enhanced Exposure Widget
    sigma_slider = FloatSlider(min=0.1, max=1.0, step=0.05, value=0.2, description="Sigma:")
    epsilon_slider = SelectionSlider(
        options=[(f"{e:.1e}", e) for e in epsilon_values],
        value=default_epsilon,
        description="Epsilon:"
    )
    kernel_slider = IntSlider(min=1, max=9, step=2, value=3, description="Kernel dim:")
    enhanced_box = VBox([
        Label("Enhanced Exposure Fusion:"),
        sigma_slider,
        epsilon_slider,
        kernel_slider
    ])
    
    def update_enhanced(change):
        run_interactive(methods['Enhanced Exposure'],
                        sigma=sigma_slider.value,
                        epsilon=epsilon_slider.value,
                        blur_kernel=(kernel_slider.value, kernel_slider.value))
    
    for slider in [sigma_slider, epsilon_slider, kernel_slider]:
        slider.observe(update_enhanced, names='value')

    # Domain Transform Widget
    sigma_spatial = FloatSlider(min=10, max=100, step=5, value=60, description="sigmaSpatial:")
    sigma_color = FloatSlider(min=0.1, max=1.0, step=0.05, value=0.4, description="sigmaColor:")
    epsilon_dt = SelectionSlider(
        options=[(f"{e:.1e}", e) for e in epsilon_values],
        value=default_epsilon,
        description="Epsilon:"
    )
    domain_box = VBox([
        Label("Domain Transform Fusion:"),
        sigma_spatial,
        sigma_color,
        epsilon_dt
    ])
    
    def update_domain(change):
        run_interactive(methods['Domain Transform'],
                        sigmaSpatial=sigma_spatial.value,
                        sigmaColor=sigma_color.value,
                        epsilon=epsilon_dt.value)
    
    for slider in [sigma_spatial, sigma_color, epsilon_dt]:
        slider.observe(update_domain, names='value')

    # Wavelet Fusion Widget
    level_slider = IntSlider(min=1, max=5, step=1, value=2, description="Level:")
    wavelet_box = VBox([
        Label("Wavelet Fusion:"),
        level_slider
    ])
    
    def update_wavelet(change):
        run_interactive(methods['Wavelet Fusion'], level=change['new'])
    level_slider.observe(update_wavelet, names='value')

    # Button-based methods
    def create_button(method_name):
        btn = Button(description=f"Run {method_name}")
        def on_click(b):
            run_interactive(methods[method_name])
        btn.on_click(on_click)
        return btn

    avg_button = create_button('Average Fusion')
    mertens_button = create_button('Mertens Fusion')
    exposure_button = create_button('Exposure Fusion')
    exp_comp_button = create_button('Exposure Compensation')

    # --------------------------
    # Tabbed interface layout
    # --------------------------
    tab = Tab(children=[
        VBox([Label("Average Fusion:"), avg_button]),
        VBox([Label("Mertens Fusion:"), mertens_button]),
        laplacian_box,
        VBox([Label("Exposure Fusion:"), exposure_button]),
        VBox([Label("Exposure Compensation:"), exp_comp_button]),
        enhanced_box,
        domain_box,
        wavelet_box
    ])
    
    tab.set_title(0, "Average")
    tab.set_title(1, "Mertens")
    tab.set_title(2, "Laplacian")
    tab.set_title(3, "Exposure")
    tab.set_title(4, "Exp. Comp.")
    tab.set_title(5, "Enhanced")
    tab.set_title(6, "Domain Trans.")
    tab.set_title(7, "Wavelet")

    # Combine tabs and output in single layout
    main_layout = VBox([
        tab,
        global_output
    ])
    
    display(main_layout)

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
    """Display two images with their RGB histograms in a 2x2 grid."""
    import matplotlib.pyplot as plt
    import cv2

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top row: display images
    axes[0, 0].imshow(image1)
    axes[0, 0].set_title(title1)
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(image2)
    axes[0, 1].set_title(title2)
    axes[0, 1].axis("off")


    plotColourChannels(axes[1, 0], image1)
    plotColourChannels(axes[1, 1], image2)
    
    plt.tight_layout()
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