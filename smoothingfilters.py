import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random

def gaussian(): 
    image = cv.imread("lab1/free-nature-images.jpg")
    if image is None:
        raise FileNotFoundError("The image file 'lab1/free-nature-images.jpg' was not found. Please check the file path.")
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    filtered_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(1, image.shape[0]-1): 
        for j in range(1, image.shape[1]-1): 
            region = image[i-1:i+2, j-1:j+2] 
            result = sum(sum(region * kernel))
            filtered_image[i, j] = result
    return image, filtered_image

def mean_filter(): 
    kernel = np.ones((3,3), np.float32) / 9
    image = cv.imread("lab1/free-nature-images.jpg")
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    filtered_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(1, image.shape[0]-1): 
        for j in range(1, image.shape[1]-1): 
            region = image[i-1:i+2, j-1:j+2] 
            result = sum(sum(region * kernel))
            filtered_image[i, j] = result
    return image, filtered_image

def weighted_average():
    kernel = np.array([[1, 2, 1], [2, 5, 2], [1, 2, 1]], dtype=np.float32)
    kernel /= kernel.sum()
    image = cv.imread("lab1/free-nature-images.jpg")
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    filtered_image = np.zeros_like(image)
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            region = image[i-1:i+2, j-1:j+2]
            filtered_image[i, j] = np.sum(region * kernel)
    return image, filtered_image

def sobel():
    image = cv.imread("lab1/free-nature-images.jpg")
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    filtered_image = np.zeros_like(image)
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            region = image[i-1:i+2, j-1:j+2]
            gx = np.sum(region * sobel_x)
            gy = np.sum(region * sobel_y)
            filtered_image[i, j] = np.sqrt(gx**2 + gy**2)
    return image, filtered_image

def laplace():
    image = cv.imread("lab1/free-nature-images.jpg")
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    filtered_image = np.zeros_like(image, dtype=np.float32)
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            region = image[i-1:i+2, j-1:j+2]
            value = np.sum(region * kernel)
            filtered_image[i, j] = value
    # Convert to 8-bit image for visualization
    filtered_image = np.abs(filtered_image)
    filtered_image = np.clip(filtered_image, 0, 255)
    filtered_image = filtered_image.astype(np.uint8)
    return image, filtered_image
def min_filter():
    image = cv.imread("lab1/free-nature-images.jpg")
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    filtered_image = np.zeros_like(image)
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            region = image[i-1:i+2, j-1:j+2]
            filtered_image[i, j] = np.min(region)
    return image, filtered_image

def max_filter():
    image = cv.imread("lab1/free-nature-images.jpg")
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    filtered_image = np.zeros_like(image)
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            region = image[i-1:i+2, j-1:j+2]
            filtered_image[i, j] = np.max(region)
    return image, filtered_image

def median_filter():
    image = cv.imread("lab1/free-nature-images.jpg")
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    filtered_image = np.zeros_like(image)
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            region = image[i-1:i+2, j-1:j+2]
            filtered_image[i, j] = np.median(region)
    return image, filtered_image

# Run and collect all filter results
original, gaussian_img = gaussian()
_, mean_img = mean_filter()
_, weighted_img = weighted_average()
_, sobel_img = sobel()
_, laplace_img = laplace()
_, min_img = min_filter()
_, max_img = max_filter()
_, median_img = median_filter()

# Plot all results
titles = ['Original', 'Gaussian', 'Mean', 'Weighted Avg', 'Sobel', 'Laplace', 'Min', 'Max', 'Median']
images = [original, gaussian_img, mean_img, weighted_img, sobel_img, laplace_img, min_img, max_img, median_img]

plt.figure(figsize=(20, 10))
for i in range(len(images)):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
