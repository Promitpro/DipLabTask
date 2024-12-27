import cv2
import numpy as np

img = cv2.imread("E:\\OneDrive\\Documents\\4th year 1st semester\\digital image processing\\images\\DIP_Lab_task_04.jpg")

def reduce_brightness(image, brightness_factor):
    height, width, channels = image.shape
    new_image = np.zeros_like(image)  
    for y in range(height):
        for x in range(width):
            for c in range(channels): 
                new_pixel_value = int(image[y, x, c] * brightness_factor)
                new_pixel_value = np.clip(new_pixel_value, 0, 255) 
                new_image[y, x, c] = new_pixel_value
    return new_image

brightness_factor = 0.5
reduced_brightness_img = reduce_brightness(img, brightness_factor)

cv2.imshow("Original Image", img)
cv2.imshow("Reduced Brightness Image", reduced_brightness_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Reference
# www.stackoverflow.com
# www.geeksforgeeks.org
# www.allaboutcircuits.com
