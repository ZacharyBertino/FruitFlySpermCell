from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import PolygonSelector, Button
from matplotlib.path import Path
from scipy.ndimage import label
from matplotlib.widgets import TextBox
from matplotlib.widgets import CheckButtons

import cv2

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

def get_largest_connected_component(binary_image):
    """
    Identifies the largest connected component in a binary image.

    Parameters:
    - binary_image: NumPy array of the binary image (values are 0 or 255).

    Returns:
    - largest_component: Binary image with the largest connected component all 255 and everything else all 0.
    """
    # Ensure format
    binary_image = (binary_image > 0).astype(np.uint8)

    # Label connected components
    labeled_array, num_features = label(binary_image)

    # If there are no components, return an empty image
    if num_features == 0:
        return np.zeros_like(binary_image, dtype=np.uint8)

    # Find the largest component
    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0  # Ignore the background
    largest_component_label = component_sizes.argmax()

    # Create a binary mask for the largest component
    largest_component = (
        labeled_array == largest_component_label).astype(np.uint8) * 255

    return largest_component

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

    print(threshold)

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
        print("in")
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
        Display the binary image and add controls to adjust the number of components.
        """
        fig, ax = display(binary_image)

        # Add a text box to input the number of components
        ax_textbox_n = plt.axes([0.4, 0.05, 0.2, 0.075])
        n_textbox = TextBox(ax_textbox_n, 'Num Components', initial=str(n_components[0]))
        n_textbox.on_submit(update_n_components)

        ax_textbox_thresh = plt.axes([0.1, 0.15, 0.2, 0.075])
        threshold_textbox = TextBox(ax_textbox_thresh, 'Threshold', initial=str(custom_threshold[0]))
        threshold_textbox.on_submit(update_threshold)

        ax_checkbox = plt.axes([0.7, 0.25, 0.2, 0.1])
        checkbox = CheckButtons(ax_checkbox, ["Use Custom Threshold"], use_custom_threshold)
        checkbox.on_clicked(toggle_custom_threshold)

        # Add "Enter" button to reapply the function
        ax_enter_button = plt.axes([0.7, 0.05, 0.2, 0.075])
        enter_button = Button(ax_enter_button, 'Enter')

        def on_enter(event):
            redraw_with_new_params()

        enter_button.on_clicked(on_enter)

        # Add "Undo" button
        ax_undo_button = plt.axes([0.1, 0.05, 0.2, 0.075])
        undo_button = Button(ax_undo_button, 'Undo')

        def on_undo(event):
            plt.close()
            draw_roi_and_mask(image_array, threshold, background_value)

        undo_button.on_clicked(on_undo)

        # Add "Find Length" button
        ax_find_length_button = plt.axes([0.7, 0.15, 0.2, 0.075])
        find_length_button = Button(ax_find_length_button, 'Find Length')

        def on_find_length(event):
            print("Find Length button clicked. (Implementation pending)")

        find_length_button.on_clicked(on_find_length)

        plt.show()

    def on_mask_button_click(event):
        redraw_with_new_params()

    # Plot the image and setup PolygonSelector
    fig, ax = display(image_array)
    selector = PolygonSelector(ax, onselect)

    # Add a text box to input the number of components
    ax_textbox_n = plt.axes([0.1, 0.05, 0.2, 0.075])
    n_textbox = TextBox(ax_textbox_n, 'Num Components', initial=str(n_components[0]))
    n_textbox.on_submit(update_n_components)

    # Add a text box to input the threshold
    ax_textbox_thresh = plt.axes([0.4, 0.05, 0.2, 0.075])
    threshold_textbox = TextBox(ax_textbox_thresh, 'Threshold', initial=str(custom_threshold[0]))
    threshold_textbox.on_submit(update_threshold)

    # Add a checkbox to toggle custom thresholding
    ax_checkbox = plt.axes([0.7, 0.15, 0.2, 0.1])
    checkbox = CheckButtons(ax_checkbox, ["Use Custom Threshold"], use_custom_threshold)
    checkbox.on_clicked(toggle_custom_threshold)

    # Apply mask button
    mask_button = plt.axes([0.7, 0.05, 0.2, 0.075])
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
    # image_path = "data/medium/24708.1_6 at 20X.jpg" 
    #1_4
    image_path = "data/medium/24708.1_6 at 20X.jpg" 
    # image_path = "data/hard/472.1A.1_1.jpg"
    image = Image.open(image_path)
    image = np.array(image)
    preppedImage = preprocessing(image)
    unsharpImage = unsharp_mask(image)
    
    # Allow the user to draw an ROI and mask the image
    draw_roi_and_mask(image, 190)


main()


#take all components of certain size? Take largerst percentage and see if they connect?
#Have use trace it. Every point they make take largest closest component, then combine them all
#User can threshold based on scale if too much noise.
#