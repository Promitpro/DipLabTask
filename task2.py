import cv2
import numpy as np

image_path = 'E:\\OneDrive\\Documents\\4th year 1st semester\\digital image processing\\images\\DIP_Lab_task_02.jpg'  
image = cv2.imread(image_path)

height, width, channels = image.shape
gray_image = np.zeros((height, width), dtype=np.uint8)

for y in range(height):
    for x in range(width):
        R = image[y, x, 2] 
        G = image[y, x, 1]
        B = image[y, x, 0]
        gray_value = int(0.299 * R + 0.587 * G + 0.114 * B)
        gray_image[y, x] = gray_value

top_left_x = 50  
top_left_y = 50  
sub_region_size = 5 

sub_region = gray_image[top_left_y:top_left_y + sub_region_size, 
                        top_left_x:top_left_x + sub_region_size]

print("Sub-region intensity values (5x5):")
print(sub_region)

cv2.imshow('Original RGB Image', image)
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
