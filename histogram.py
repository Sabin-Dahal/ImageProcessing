import cv2
# import numpy as np
import matplotlib.pyplot as plt

# def calculate_histogram(image):
#     hist = np.zeros(256)
#     for pixel in image.ravel():
#         hist[pixel] += 1
#     return hist

image = cv2.imread('lab1/free-nature-images.jpg', cv2.IMREAD_GRAYSCALE)
# histogram = calculate_histogram(image)
# equalized = cv2.equalizeHist(image)
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
# ax1.imshow(image, cmap='gray', vmin=0, vmax=255)
# ax1.set_title('Original Image')
# ax2.imshow(equalized, cmap='gray', vmin=0, vmax=255)
# ax2.set_title('Equalized Image')

# original_hist = calculate_histogram(image)
# equalized_hist = calculate_histogram(equalized)

# ax3.plot(original_hist)
# ax3.set_title('Original Histogram')
# ax3.set_xlabel('Pixel Intensity')
# ax3.set_ylabel('Frequency')

# ax4.plot(equalized_hist)
# ax4.set_title('Equalized Histogram')
# ax4.set_xlabel('Pixel Intensity')
# ax4.set_ylabel('Frequency')

# plt.tight_layout()
# plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram(image):
    """Calculates the histogram of an image."""
    histogram = [0] * 256
    for row in image:
        for pixel in row:
            histogram[pixel] += 1
    return histogram

def calculate_cdf(histogram):
    """Calculates the cumulative distribution function (CDF)."""
    cdf = [0] * 256
    cdf[0] = histogram[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + histogram[i]
    return cdf

def normalize_cdf(cdf, total_pixels):
    """Normalizes the CDF to the range 0-255."""
    return [round(((c - cdf[0]) / (total_pixels - cdf[0])) * 255) if total_pixels - cdf[0] > 0 else 0 for c in cdf]

def equalize_image(image, normalized_cdf):
    """Equalizes the image using the normalized CDF."""
    equalized_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            equalized_image[i, j] = normalized_cdf[image[i, j]]
    return equalized_image

def histogram_equalization(image):
    """Performs histogram equalization and shows results."""
    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale.")

    # Original image histogram and CDF
    original_histogram = calculate_histogram(image)
    cdf = calculate_cdf(original_histogram)
    total_pixels = image.shape[0] * image.shape[1]
    normalized_cdf = normalize_cdf(cdf, total_pixels)

    # Equalized image
    equalized_img = equalize_image(image, normalized_cdf)
    equalized_histogram = calculate_histogram(equalized_img)

    # Plot original and equalized images with histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(equalized_img, cmap='gray')
    axes[0, 1].set_title('Equalized Image')
    axes[0, 1].axis('off')

    axes[1, 0].hist(image.ravel(), bins=256, range=(0, 256), color='gray')
    axes[1, 0].set_title('Original Histogram')

    axes[1, 1].hist(equalized_img.ravel(), bins=256, range=(0, 256), color='gray')
    axes[1, 1].set_title('Equalized Histogram')

    plt.tight_layout()
    plt.show()

    return equalized_img

# Example usage:
# Load grayscale image
image = cv2.imread("lab1/free-nature-images.jpg", cv2.IMREAD_GRAYSCALE)
equalized = histogram_equalization(image)
