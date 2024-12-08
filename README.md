## Sperm Measurement Tool Installation Instructions

Follow these steps to run the measurement tool with your image file as input:

1. **Clone the Repository**  
   Open your terminal and clone this repository to your local machine:

   > git clone https://github.com/ZacharyBertino/FruitFlySpermCell.git
   
   If the command line window says the directory already exists, skip this step and proceed to step 2.

2. **Navigate to the Repository Directory**  
   Change into the directory where the repository was cloned:

   > cd FruitFlySpermCell

3. **Install Dependencies**  
   Ensure you have Python installed on your machine, then use `pip` to install the required libraries:

   > pip install pillow "numpy>=1.16.5,<1.23.0" matplotlib scipy opencv-python

   If additional dependencies are required, errors in your command line window will indicate this. Install them in the same manner.

4. **Run the Script**  
   Use the following command to run `SpermLengthFinal.py`, replacing `'path/to/your/image.jpg'` with the file path to your image. Be sure to include the file path in single quotes:

   > python SpermLengthFinal.py 'path/to/your/image.jpg'

### Example
If your image is stored at `/home/user/images/sperm_sample.jpg`, run:

> python SpermLengthFinal.py '/home/user/images/sperm_sample.jpg'


## Sperm Measurement Tool Usage Instructions

Follow these steps to use the program effectively to analyze sperm length from an image:

### 1. Launch the Program
Run the program with your image file as input. The image will open in an interactive window, along with an **"Apply Mask"** button.

### 2. (Optional) Specify a Region of Interest (ROI)
- Use the **polygon tool** to define the region of interest (ROI) by clicking to add points that outline the desired area.
- Once the shape is closed, the area inside the polygon will be treated as the ROI, and everything outside will be considered background.
- **Note**: This step is optional but highly recommended for images with a noisy background to improve accuracy.

### 3. Apply the Mask
- Press the **"Apply Mask"** button to binarize the image (or specified ROI).
- By default, the program uses an **adaptive thresholding algorithm** to create a binary image.
- If you wish to alter your region of interest after pressing the **"Apply Mask"** button, press the **"Undo"** button on the following screen.

### 4. (Optional) Adjust Threshold 
- If the adaptive thresholding results in too much noise or an unusable binary image, toggle the **"Use Custom Threshold"** checkbox.
- Enter a threshold value in the **"Threshold"** input box and press the **"Enter"** button to apply **global thresholding** with the specified value.

### 5. Adjust the Number of Components
- Use the **"Components"** input box to specify the number of connected components to keep and press the **"Enter"** button to apply the change.
- Increase the value until the entire sperm is visible in the image.
- This step helps remove unwanted artifacts and focus on the sperm structure.

### 6. Refine the Image with the Cut Tool
- Press the **"Cut"** button to activate the eraser-like tool.
- Use the tool to select and outline regions of noise or unwanted artifacts.
- Once the shape is closed, press the **"Done"** button.
- The region inside will be removed from the image, treating it like the background.
- Repeat as needed until only the sperm remains in the image.

### 7. Find the Length
- When satisfied with the cleaned binary image of the sperm, press the **"Find Length"** button.
- The program will run a **thinning algorithm** to skeletonize the sperm structure.
- Updates on the progress will be displayed in the terminal or command-line interface.
- Once completed, the estimated sperm length will be printed in the terminal.

### Final Notes
- The thinning process may take some time depending on the complexity of the image.
- The length is output in micrometers, providing an accurate estimate of the sperm's size.
