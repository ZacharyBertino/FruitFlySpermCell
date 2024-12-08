from PIL import Image
import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib.widgets import PolygonSelector, Button
from matplotlib.path import Path
from scipy.ndimage import label
from matplotlib.widgets import TextBox
from matplotlib.widgets import CheckButtons

import cv2

def closeBinary(binary_image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    res = cv2.morphologyEx(binary_image,cv2.MORPH_CLOSE,kernel)
    return res

def calculate_length(thinned_image, pixels_per_micrometer=3.06):
    """
    Calculate the length of a skeletonized line in micrometers.

    Parameters:
    - thinned_image: Binary image of the thinned skeleton.
    - pixels_per_micrometer: Conversion factor from pixels to micrometers.

    Returns:
    - length_in_micrometers: Length of the skeleton in micrometers.
    """
    # Count the number of foreground pixels in the thinned skeleton
    pixel_count = np.sum(thinned_image > 0)  # Count non-zero pixels

    # Convert pixel count to micrometers
    length_in_micrometers = pixel_count / pixels_per_micrometer

    return length_in_micrometers


def _thinningIteration(im, iter):
    """Perform one iteration of thinning using vectorized operations."""

    rows, cols = im.shape
    P = np.zeros((rows, cols, 8), dtype=np.uint8)

    # Neighbors: P2 to P9
    P[:, :, 0] = np.roll(im, shift=-1, axis=0)  # P2
    P[:, :, 1] = np.roll(np.roll(im, shift=-1, axis=0), shift=1, axis=1)  # P3
    P[:, :, 2] = np.roll(im, shift=1, axis=1)  # P4
    P[:, :, 3] = np.roll(np.roll(im, shift=1, axis=0), shift=1, axis=1)  # P5
    P[:, :, 4] = np.roll(im, shift=1, axis=0)  # P6
    P[:, :, 5] = np.roll(np.roll(im, shift=1, axis=0), shift=-1, axis=1)  # P7
    P[:, :, 6] = np.roll(im, shift=-1, axis=1)  # P8
    P[:, :, 7] = np.roll(np.roll(im, shift=-1, axis=0), shift=-1, axis=1)  # P9

    # Calculate A (number of 0->1 transitions in neighbors)
    transitions = ((P[:, :, :-1] == 0) & (P[:, :, 1:] == 1)).sum(axis=2) + (
        (P[:, :, -1] == 0) & (P[:, :, 0] == 1)
    )
    # Calculate B (number of 1s in neighbors)
    neighbors_sum = P.sum(axis=2)

    # Masks for conditions
    m1 = (P[:, :, 0] * P[:, :, 2] * P[:, :, 4] if iter ==
          0 else P[:, :, 0] * P[:, :, 2] * P[:, :, 6])
    m2 = (P[:, :, 2] * P[:, :, 4] * P[:, :, 6] if iter ==
          0 else P[:, :, 0] * P[:, :, 4] * P[:, :, 6])

    # Conditions for marking pixels for removal
    remove = (
        (im == 1) &  # Only consider foreground pixels
        (transitions == 1) &
        (2 <= neighbors_sum) & (neighbors_sum <= 6) &
        (m1 == 0) &
        (m2 == 0)
    )

    # Remove marked pixels
    return im & ~remove


def thinning(src):
    """Perform Zhang-Suen thinning on a binary image."""

    dst = src // 255  # Convert to binary 0/1
    prev = np.zeros_like(dst, dtype=np.uint8)
    iteration = 0

    while True:
        print(f"Starting iteration {iteration}...")
        initial_pixels = np.sum(dst)

        dst = _thinningIteration(dst, 0)
        print(f"Completed sub-iteration 0 of iteration {iteration}.")
        dst = _thinningIteration(dst, 1)
        print(f"Completed sub-iteration 1 of iteration {iteration}.")

        removed_pixels = initial_pixels - np.sum(dst)
        print(f"Iteration {iteration}: {removed_pixels} pixels removed.")

        if np.array_equal(dst, prev):  # Stop if no change
            print(
                f"No changes detected. Thinning complete after {iteration} iterations.")
            break

        prev = dst.copy()
        iteration += 1

    return dst * 255  # Convert back to 0/255

def get_largest_connected_components(binary_image, n):
    """
    Identifies the top n largest connected components in a binary image.

    Parameters:
    - binary_image: NumPy array of the binary image (values are 0 or 255).
    - n: Number of largest connected components to return.

    Returns:
    - top_components: Binary image with the n largest connected components, where each component is 255.
    """
    # Ensure format
    binary_image = (binary_image > 0).astype(np.uint8)

    # Label connected components
    labeled_array, num_features = label(binary_image)

    # If there are no components, return an empty image
    if num_features == 0:
        return np.zeros_like(binary_image, dtype=np.uint8)

    # Find the sizes of all components
    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0  # Ignore the background

    # Get the labels of the top n components
    top_labels = np.argsort(component_sizes)[-n:]

    # Create a binary mask for the top n components
    top_components = np.isin(labeled_array, top_labels).astype(np.uint8) * 255

    return top_components

def convert_to_binary(mask, image_array, threshold=None, n_components=1):
    """
    Convert an image array to a binary image based on the threshold.

    Parameters:
    - image_array: NumPy array of the image (can be grayscale or RGB)
    - threshold: Intensity value (0-255) to separate foreground from background (None for adaptive threshold)
    - n_components: Number of largest components to keep

    Returns:
    - binary_image: Binary image as a NumPy array (values are 0 or 255)
    """
    # Convert to grayscale (if needed)
    if len(image_array.shape) == 3:
        grayscale = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        grayscale = image_array

    if threshold is not None:
        # Manual thresholding
        _, binary_image = cv2.threshold(grayscale, threshold, 255, cv2.THRESH_BINARY)
    else:
        # Adaptive thresholding
        binary_image = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 1)

    # Apply mask and keep the top n components
    masked_image = np.where(mask, binary_image, 0)

    return get_largest_connected_components(masked_image, n_components)


def display(img):
    """
    Displays the given image.

    Parameters:
    - img: NumPy array of the image.

    Returns:
    - fig and ax of the subplots
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax.imshow(img, cmap='gray', interpolation='nearest')
    ax.axis("off")
    return fig, ax


def draw_roi_and_mask(image_array, threshold, background_value=0):
    image_array_base = image_array.copy()
    """
    Let the user draw a polygon around an ROI, specify number of components and/or threshold,
    and mask the outside region with a button press.

    Parameters:
    - image_array: NumPy array of the image (can be grayscale or RGB).
    - background_value: Pixel value to set outside the ROI.

    Returns:
    - None
    """
    # Store the polygon vertices, threshold, and number of components
    vertices = []
    masked_image = None
    n_components = [1]  # Default: 1 component
    custom_threshold = [40]  # Default: Use adaptive thresholding
    use_custom_threshold = [False]  # Checkbox state

    def onselect(verts):
        nonlocal vertices
        vertices = verts

    def update_threshold(text):
        """
        Update the custom threshold based on user input.
        """
        try:
            custom_threshold[0] = float(text)  # Parse input as integer
            print(f"Custom threshold updated to: {custom_threshold[0]}")
        except ValueError:
            print("Invalid threshold input. Ignoring.")

    def update_n_components(text):
        """
        Update the number of components to retain based on user input.
        """
        try:
            n_components[0] = max(1, int(text))  # Ensure n is at least 1
        except ValueError:
            print("Invalid component input. Defaulting to 1.")
            n_components[0] = 1  # Default to 1 if input is invalid

    def toggle_custom_threshold(label):
        """
        Toggle whether to use the custom threshold or adaptive threshold.
        """
        if label == "Use Custom Threshold":
            use_custom_threshold[0] = not use_custom_threshold[0]

    def redraw_with_new_params():
        """
        Reapply the masking and binary conversion with updated parameters.
        """
        if not vertices:
            h, w = image_array.shape[:2]
            mask = np.ones((h, w), dtype=bool)
        else:
            path = Path(vertices)
            h, w = image_array.shape[:2]
            y, x = np.mgrid[:h, :w]
            points = np.vstack((x.ravel(), y.ravel())).T
            mask = path.contains_points(points).reshape(h, w)

        # Determine which thresholding method to use
        final_threshold = custom_threshold[0] if use_custom_threshold[0] else None

        # Convert the masked image to binary with the specified parameters
        binary_image = convert_to_binary(mask, image_array, 
                                         final_threshold, n_components[0])

        # Close the current figure
        plt.close()

        # Display the resulting binary image with controls
        display_with_controls(binary_image, mask)

    def display_with_controls(binary_image, mask):
        """
        Display the binary image and add controls to adjust the number of components,
        use custom thresholds, and allow cutting regions.
        """
        fig, ax = display(binary_image)

        # Add a text box to input the number of components
        ax_textbox_n = plt.axes([0.915, 0.2, 0.075, 0.075])
        n_textbox = TextBox(ax_textbox_n, 'Components ', initial=str(n_components[0]))
        n_textbox.on_submit(update_n_components)

        # Add a text box for threshold
        ax_textbox_thresh = plt.axes([0.14, 0.2, 0.1, 0.075])
        threshold_textbox = TextBox(ax_textbox_thresh, 'Threshold ', initial=str(custom_threshold[0]))
        threshold_textbox.on_submit(update_threshold)

        # Add a checkbox for using custom thresholds
        ax_checkbox = plt.axes([0.01, 0.05, 0.35, 0.1])
        checkbox = CheckButtons(ax_checkbox, ["Use Custom Threshold"], use_custom_threshold)
        checkbox.on_clicked(toggle_custom_threshold)

        # Add "Enter" button to reapply the function
        ax_enter_button = plt.axes([0.4, 0.05, 0.2, 0.1])
        enter_button = Button(ax_enter_button, 'Enter')

        def on_enter(event):
            redraw_with_new_params()

        enter_button.on_clicked(on_enter)

        # Add "Undo" button
        ax_undo_button = plt.axes([0.05, 0.85, 0.15, 0.075])
        undo_button = Button(ax_undo_button, 'Undo')

        def on_undo(event):
            plt.close()
            draw_roi_and_mask(image_array_base, threshold, background_value)

        undo_button.on_clicked(on_undo)

        # Add "Find Length" button
        ax_find_length_button = plt.axes([0.82, 0.85, 0.15, 0.075])
        find_length_button = Button(ax_find_length_button, 'Find Length')

        def on_find_length(event):
            closed_image = closeBinary(binary_image)
            thinned_image = thinning(closed_image)
            leng = calculate_length(thinned_image)
            print(f"The sperm cell is {leng} micrometers")

        find_length_button.on_clicked(on_find_length)

        # Add "Cut" button
        ax_cut_button = plt.axes([0.62, 0.05, 0.35, 0.1])
        cut_button = Button(ax_cut_button, 'Cut')

        def on_cut(event):
            # Close the current display to allow polygon selection
            plt.close()

            # Display the current image for polygon selection
            fig, ax = display(binary_image)
            vertices = []

            def onselect(verts):
                nonlocal vertices
                vertices = verts

            selector = PolygonSelector(ax, onselect, props={'markersize': 8, 'markerfacecolor': 'blue'}) 

            def on_done(event):

                polygon_mask = None
                if vertices:

                    # Create a mask from the polygon
                    path = Path(vertices)
                    h, w = binary_image.shape[:2]
                    y, x = np.mgrid[:h, :w]
                    points = np.vstack((x.ravel(), y.ravel())).T
                    polygon_mask = path.contains_points(points).reshape(h, w)
                    

                    # Update the mask to remove the polygon region
                    binary_image[polygon_mask] = False
    
                plt.close()

                display_with_controls(binary_image, polygon_mask)

            # Add "Done" button to confirm the polygon
            ax_done_button = plt.axes([0.7, 0.05, 0.2, 0.075])
            done_button = Button(ax_done_button, 'Done')
            done_button.on_clicked(on_done)

            plt.show()

        cut_button.on_clicked(on_cut)

        plt.show()


    def on_mask_button_click(event):
        redraw_with_new_params()

    # Plot the image and setup PolygonSelector
    fig, ax = display(image_array)
    selector = PolygonSelector(ax, onselect, props={'markersize': 8, 'markerfacecolor': 'blue'})

    # Apply mask button
    mask_button = plt.axes([0.4, 0.05, 0.2, 0.075])
    mask_button = Button(mask_button, 'Apply Mask')
    mask_button.on_clicked(on_mask_button_click)

    # Display results
    plt.show()

    return masked_image


def preprocessing(image):
    """
    Enhance the contrast of an image represented as a NumPy array.

    Parameters:
    - image: NumPy array of the image (grayscale or RGB).

    Returns:
    - processedImage: Contrast-enhanced image as a NumPy array.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        grayscale = image

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(100, 100))
    enhanced_image = clahe.apply(grayscale)

    # Return the processed image
    return enhanced_image

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, strength=2.0):
    """
    Enhance contrast using unsharp masking.

    Parameters:
    - image: NumPy array of the image (grayscale or RGB).
    - kernel_size: Size of the Gaussian blur kernel.
    - sigma: Standard deviation for Gaussian blur.
    - strength: Factor by which the detail is amplified.

    Returns:
    - sharpened_image: Image with enhanced contrast and sharpness.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        grayscale = image

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(grayscale, kernel_size, sigma)

    # Enhance details by subtracting blurred image
    detail = grayscale - blurred
    sharpened_image = cv2.addWeighted(grayscale, 1 + strength, detail, strength, 0)

    return sharpened_image

def main():
    if len(sys.argv) < 1:
        print("Usage: python SerialThinningV2.py <filename>")

    image_path = sys.argv[1]
    image = Image.open(image_path)
    image = np.array(image)
    preppedImage = preprocessing(image)
    unsharpImage = unsharp_mask(image)
    
    # Allow the user to draw an ROI and mask the image
    draw_roi_and_mask(image, 190)

main()
