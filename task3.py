import numpy as np
import cv2

image = cv2.imread("E:\\OneDrive\\Documents\\4th year 1st semester\\digital image processing\\images\\DIP_Lab_task_03.bmp", cv2.IMREAD_GRAYSCALE)

# gaussian_filter = cv2.GaussianBlur(img, (5, 5), 1)
# median_filter cv2.medianBlur(img, 5)
# bilateral_filter = cv2.bilateralFilter(img, 9, 75, 75)

def median_filter(image, kernel_size):
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='reflect')
    output = np.zeros_like(image)

    rows, cols = image.shape
    for y in range(rows):
        for x in range(cols):
            window = padded[y:y + kernel_size, x:x + kernel_size]
            output[y, x] = np.median(window)

    return output

kernel_size = 3
filtered_image = median_filter(image, kernel_size)

cv2.imshow('Original', image)
cv2.imshow('Filtered', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
