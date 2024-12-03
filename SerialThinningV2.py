from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import PolygonSelector, Button
from matplotlib.path import Path
from scipy.ndimage import label


def convert_to_binary(image_array, threshold):
    """
    Convert an image array to a binary image based on the threshold.

    Parameters:
    - image_array: NumPy array of the image (can be grayscale or RGB)
    - threshold: Intensity value (0-255) to separate foreground from background

    Returns:
    - binary_image: Binary image as a NumPy array (values are 0 or 255)
    """
    # Convert to grayscale (if needed)
    if len(image_array.shape) == 3:
        grayscale = np.mean(image_array, axis=2)
    else:
        grayscale = image_array

    # Threshold image
    binary_image = np.where(grayscale > threshold, 255, 0).astype(np.uint8)

    return binary_image


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
    """
    Let the user draw a polygon around an ROI and mask the outside region with a button press.

    Parameters:
    - image_array: NumPy array of the image (can be grayscale or RGB).
    - background_value: Pixel value to set outside the ROI.

    Returns:
    - None
    """
    # Store the polygon vertices
    vertices = []
    masked_image = None

    def onselect(verts):
        nonlocal vertices
        vertices = verts

    def on_mask_button_click(event):
        nonlocal masked_image
        if not vertices:
            h, w = image_array.shape[:2]
            mask = np.ones((h, w), dtype=bool)
        else:
            # Create a mask from the polygon
            path = Path(vertices)
            h, w = image_array.shape[:2]
            y, x = np.mgrid[:h, :w]
            points = np.vstack((x.ravel(), y.ravel())).T
            mask = path.contains_points(points).reshape(h, w)

        # Apply the mask to the image
        if image_array.ndim == 3:
            # RGB image
            masked_image = np.where(
                mask[..., None], image_array, background_value)
        else:
            # Grayscale image
            masked_image = np.where(mask, image_array, background_value)

        # Convert masked image to binary
        binary_image = convert_to_binary(masked_image, threshold)

        # Close the current figure
        plt.close()

        # Display the resulting binary image
        display(binary_image)

        # Add "Undo" button
        ax_undo_button = plt.axes([0.1, 0.05, 0.2, 0.075])
        undo_button = Button(ax_undo_button, 'Undo')

        def on_undo(event):
            # When undo pressed, close image and recall process
            plt.close()
            draw_roi_and_mask(image_array, threshold, background_value)

        undo_button.on_clicked(on_undo)

        # Add "Find Length" button
        ax_find_length_button = plt.axes([0.7, 0.05, 0.2, 0.075])
        find_length_button = Button(ax_find_length_button, 'Find Length')

        def on_find_length(event):
            print("Find Length button clicked. (Implementation pending)")

        find_length_button.on_clicked(on_find_length)

        plt.show()

    # Plot the image and setup PolygonSelector
    fig, ax = display(image_array)
    selector = PolygonSelector(ax, onselect)

    # Apply mask button
    mask_button = plt.axes([0.7, 0.05, 0.2, 0.075])
    mask_button = Button(mask_button, 'Apply Mask')
    mask_button.on_clicked(on_mask_button_click)

    # Display results
    plt.show()

    return masked_image


def main():
    image_path = "./data/hard/472.1B.1_5&6.jpg"
    image = Image.open(image_path)
    image = np.array(image)

    # Allow the user to draw an ROI and mask the image
    draw_roi_and_mask(image, 200)


main()
