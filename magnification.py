import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

#magnification using replication
def magnify_image_replication(image, factor):   
    magnified_image = np.zeros((image.shape[0] * factor, image.shape[1] * factor), dtype=image.dtype)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            magnified_image[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor] = image[i, j]
    return magnified_image

#generate a 8x8 image to magnify to 16x16
image = plt.imread("lab1/free-nature-images.jpg")
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# print(image)
# #magnify the image to 16x16

magnified_image = magnify_image_replication(image, 2)
print(magnified_image)

cv.imshow("Original Image", image)
cv.imshow("Magnified Image", magnified_image)

cv.waitKey(0)

def magnify_image_interpolation(image,factor):
    h, w = image.shape
    new_h, new_w = factor * h, factor * w
    output = np.zeros((new_h, new_w), dtype=float)

    # Step 1: Copy original pixels to even indices
    for i in range(h):
        for j in range(w):
            output[factor*i, factor*j] = image[i, j]
    print(output)
    print("-"*100)

    # Step 2: Interpolate columns (horizontal)
    for i in range(0, new_h, 2):
        for j in range(1, new_w, 2):
            left = output[i, j - 1]
            right = output[i, j + 1] if j + 1 < new_w else 0
            output[i, j] = (left + right) / 2
    print(output)
    print("-"*100)
    # Step 3: Interpolate rows (vertical)
    for i in range(1, new_h, 2):
        for j in range(new_w):
            top = output[i - 1, j]
            bottom = output[i + 1, j] if i + 1 < new_h else 0
            output[i, j] = (top + bottom) / 2
    print(output)
    return output

magnify = magnify_image_interpolation(image, 2)
magnify = np.clip(magnify, 0, 255).astype(np.uint8)

print(np.count_nonzero(magnify==0))
cv.imshow("Original Image", image)
cv.imshow("Magnified Image", magnify)
cv.waitKey(0)


