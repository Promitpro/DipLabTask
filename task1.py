import cv2
import numpy as np

image_path = 'E:\\OneDrive\\Documents\\4th year 1st semester\\digital image processing\\images\\Copy of DIP_Lab_task_01.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found!")
    exit()

height, width = image.shape
print(f"Image Dimensions: Width={width}, Height={height}")

min_intensity = np.min(image)
max_intensity = np.max(image)
print(f"Intensity Range: Min={min_intensity}, Max={max_intensity}")

intensity_freq = [0] * 256
for pixel in image.flatten():
    intensity_freq[pixel] += 1

print("Frequency Distribution of Intensities:")
for intensity, freq in enumerate(intensity_freq):
    if freq >= 0:
        print(f"Intensity {intensity}: {freq}")

cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Reference
# www.geeksforgeeks.org

