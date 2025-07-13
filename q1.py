#Contrast Stretching, Thresholding, Digital Negative and Intensity Level Slicing
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def contrast_stretching(): 
    image = cv.imread("lab1/lightimageforcontrast.jpg")
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    contrast_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    min_val = np.min(image)
    print("min_val:",min_val)
    max_val = np.max(image)
    print("max_val:",max_val)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            contrast_img[i][j] = int((image[i][j] - min_val) * 255 / (max_val - min_val))
    print("Original Image:\n", image)
    print("Stretched Image:\n", contrast_img)
    plt.figure(figsize=(15, 10))
    
    plt.subplot(231)
    plt.title('Original Image')
    plt.imshow(image, cmap = 'gray', vmin=0, vmax=255)
    plt.subplot(232)
    plt.title('Original Histogram')
    plt.hist(image.ravel(), 256, [0, 256])
    
    plt.subplot(233)
    plt.title('Stretched Image')
    plt.imshow(contrast_img,cmap='gray', vmin=0, vmax=255)
    plt.subplot(234)
    plt.title('Stretched Histogram')
    plt.hist(contrast_img.ravel(), 256, [0, 256])
    
    plt.show()
    return contrast_img
# contrast_stretching()
def intensity_slicing(): #without background
    img = cv.imread('lab1/free-nature-images.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    row, column = img.shape
    img1 = np.zeros((row,column),dtype = 'uint8')
    min_range = 100
    max_range = 200
    for i in range(row):
        for j in range(column):
            if img[i,j]>min_range and img[i,j]<max_range:
                img1[i,j] = 255
            else:
                img1[i,j] = 0
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title("Original")
    plt.subplot(122), plt.imshow(img1, cmap='gray'), plt.title("=Intensity Slicing")
    plt.show()
    return intensity_slicing
intensity_slicing()
def intensity_slicing_with_background(): #with background
    img = cv.imread('lab1/free-nature-images.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    row, column = img.shape
    img1 = np.zeros((row,column),dtype = 'uint8')
    min_range = 100
    max_range = 200
    for i in range(row):
        for j in range(column):
            if img[i,j]>min_range and img[i,j]<max_range:
                img1[i,j] = 255
            else:
                img1[i,j] = img[i,j]
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title("Original")
    plt.subplot(122), plt.imshow(img1, cmap='gray'), plt.title("=Intensity Slicing")
    plt.show()
    return intensity_slicing_with_background
intensity_slicing_with_background()

def digital_negative(): 
    image = cv.imread("lab1/free-nature-images.jpg")
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    negative_image = 255 - image
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title("Original")
    plt.subplot(122), plt.imshow(negative_image, cmap='gray'), plt.title("Digital Negative")
    plt.show()
    return negative_image
digital_negative()
def thresholding():
    image = cv.imread("lab1/free-nature-images.jpg")
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    thresholded_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > 127:
                thresholded_image[i][j] = 255
            else:
                thresholded_image[i][j] = 0
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title("Original")
    plt.subplot(122), plt.imshow(thresholded_image, cmap='gray'), plt.title("Thresholded")
    plt.show()
    return thresholded_image
    
# thresholding()