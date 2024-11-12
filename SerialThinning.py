from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


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
    image_path = "./data/easy/24708.1_1 at 20x.jpg"
    image = Image.open(image_path)

    # Convert to numpy array
    image_array = np.array(image)
    binary_image_array = convert_to_binary(image_array, 200)
    show_image(binary_image_array)

    print("Image shape:", binary_image_array.shape)


main()
