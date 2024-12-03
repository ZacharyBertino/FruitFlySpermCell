from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import PolygonSelector, Button
from matplotlib.path import Path
# from scipy.ndimage import label



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



    """
    Let the user draw a polygon around an ROI and mask the outside region.

    Parameters:
    - image_array: NumPy array of the image (can be grayscale or RGB).
    - background_value: Pixel value to set outside the ROI.

    Returns:
    - masked_image: Image array with the outside region set to the background.
    """
    # Display the image for interactive selection
    fig, ax = plt.subplots()
    ax.imshow(image_array, interpolation='nearest')

    # Store the polygon vertices
    vertices = []

    # Function to capture polygon vertices
    def onselect(vertices_local):
        nonlocal vertices
        vertices = vertices_local

    # Create a PolygonSelector widget
    selector = PolygonSelector(ax, onselect)

    # Wait for the user to finish drawing
    print("Draw a polygon around the area of interest and press ENTER.")
    plt.show()

    # Check if vertices were drawn
    if not vertices:
        print("No ROI selected.")
        return image_array

    # Create a mask based on the polygon
    path = Path(vertices)
    h, w = image_array.shape[:2]
    y, x = np.mgrid[:h, :w]
    points = np.vstack((x.ravel(), y.ravel())).T
    mask = path.contains_points(points).reshape(h, w)

    # Apply mask to the image
    masked_image = np.where(mask[..., None], image_array, background_value)

    return masked_image

def draw_roi_and_mask(image_array, background_value=0):
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

    # Callback for PolygonSelector
    def onselect(verts):
        nonlocal vertices
        vertices = verts

    def on_button_click(event):
        nonlocal masked_image
        if not vertices:
            print("No ROI selected. Draw a polygon first!")
            return

        # Create a mask from the polygon
        path = Path(vertices)
        h, w = image_array.shape[:2]
        y, x = np.mgrid[:h, :w]
        points = np.vstack((x.ravel(), y.ravel())).T
        mask = path.contains_points(points).reshape(h, w)

        # Apply the mask to the image
        if image_array.ndim == 3:  # RGB image
            masked_image = np.where(mask[..., None], image_array, background_value)
        else:  # Grayscale image
            masked_image = np.where(mask, image_array, background_value)

        # Convert the masked image to binary
        # grayscale = np.mean(masked_image, axis=2).astype(np.uint8)
        # percentile = 90
        # binary_image = np.percentile(grayscale, percentile)
        binary_image = convert_to_binary(masked_image, threshold=200)

        # Close the current figure
        plt.close()

        # Display the binary thresholded image
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        ax.imshow(binary_image, cmap='gray', interpolation='nearest')
        ax.set_title("Thresholded Binary Image")

        # Add "Redo" button
        ax_redo_button = plt.axes([0.1, 0.05, 0.2, 0.075])
        redo_button = Button(ax_redo_button, 'Redo')

        def on_redo(event):
            plt.close()  # Close the current image
            draw_roi_and_mask(image_array)  # Restart the process



        redo_button.on_clicked(on_redo)

        # Add "Find Length" button
        ax_find_length_button = plt.axes([0.7, 0.05, 0.2, 0.075])
        find_length_button = Button(ax_find_length_button, 'Find Length')

        def on_find_length(event):
            print("Find Length button clicked. (Implementation pending)")

        find_length_button.on_clicked(on_find_length)

        plt.show()



    # Plot the image and setup PolygonSelector
    fig, ax = plt.subplots()
    ax.imshow(image_array, interpolation='nearest')
    selector = PolygonSelector(ax, onselect)

    # Add a button
    ax_button = plt.axes([0.8, 0.05, 0.1, 0.075])  # Position of the button [x, y, width, height]
    button = Button(ax_button, 'Apply Mask')
    button.on_clicked(on_button_click)

    print("Draw a polygon around the area of interest, then click 'Apply Mask' to proceed.")
    plt.show()

    return masked_image



def show_image(image_array):
    '''
    Display image using matplotlib.

    Parameters:
    - image_array: NumPy array of the image (can be grayscale or RGB)
    '''
    
    plt.imshow(image_array, interpolation='nearest')
    plt.show()





# Import test data

def main():
    image_path = "data/hard/472.1B.1_5&6.jpg"
    image = Image.open(image_path)

    # Convert to numpy array
    image_array = np.array(image)
    binary_image_array = convert_to_binary(image_array, 200)

    # Allow the user to draw an ROI and mask the image
    # draw_roi_and_mask(binary_image_array, background_value=255)
    draw_roi_and_mask(binary_image_array) 


main()
