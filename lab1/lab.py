import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img = cv.imread("imageprocessing/lab1/free-nature-images.jpg")

def Display():
    img = cv.imread("imageprocessing/lab1/free-nature-images.jpg")
    cv.imshow("Image", img)
    cv.waitKey(0)
    return img

def Display_using_matplot():
    img = cv.imread("imageprocessing/lab1/free-nature-images.jpg")
    plt.imshow(img) ##swaps the blue and red color
    plt.show()
    return img

def resizeImg(img):
    if img is None:
        print("Error: Image not loaded. Check the file path.")
        return None
    
    img = cv.resize(img, (1024, 1536))  
    cv.imshow("Resized Image", img) 
    cv.waitKey(0)  # Wait for a key press
    cv.destroyAllWindows()  # Close the window
    return img  # Return resized image if needed



# Function to check if the image is loaded
def is_image_loaded(img):
    if img is None:
        print("Error: Image not loaded. Check the file path.")
        return False
    return True

# Function to add brightness
def add_operations(img, value=50):
    if not is_image_loaded(img):
        return
    M = np.ones(img.shape, dtype="uint8") * value
    added = cv.add(img, M)
    cv.imshow("Original Image", img)
    cv.imshow("Brightened (Addition)", added)

# Function to subtract brightness
def subtract_operations(img, value=50):
    if not is_image_loaded(img):
        return
    M = np.ones(img.shape, dtype="uint8") * value
    subtracted = cv.subtract(img, M)
    cv.imshow("Darkened (Subtraction)", subtracted)

# Function to increase contrast
def multiplication_operations(img, factor=1.5):
    if not is_image_loaded(img):
        return
    multiplied = cv.multiply(img, np.array([factor]))
    cv.imshow("Increased Contrast (Multiplication)", multiplied.astype("uint8"))

# Function to reduce intensity
def division_operations(img, factor=2):
    if not is_image_loaded(img):
        return
    divided = cv.divide(img, np.array([factor]))
    cv.imshow("Reduced Intensity (Division)", divided.astype("uint8"))

# Load image

# Apply operations if image is loaded
if is_image_loaded(img):
    add_operations(img)  
    subtract_operations(img)  
    multiplication_operations(img)  
    division_operations(img)  
    resizeImg(img)  # Resize the image
    # Wait for key press and close all windows
    cv.waitKey(0)
    cv.destroyAllWindows()

