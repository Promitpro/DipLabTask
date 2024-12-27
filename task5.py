import cv2
import numpy as np

# Load the image
image = cv2.imread("E:\\OneDrive\\Documents\\4th year 1st semester\\digital image processing\\images\\DIP_Lab_task_05.jpg")

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define range for background (snow - adjust these values as needed)
lower_white = np.array([0, 0, 150])  # Adjust V (Value) to capture the snow
upper_white = np.array([180, 50, 255]) # Adjust S (Saturation) and V

# Create a mask for the background
mask = cv2.inRange(hsv, lower_white, upper_white)

# Invert the mask to get the dog
mask_inv = cv2.bitwise_not(mask)

# Morphological operations to clean the mask
kernel = np.ones((5, 5), np.uint8)
mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel, iterations=2) # Remove noise
mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel, iterations=2) # Close gaps

# Apply the mask to the original image
result = cv2.bitwise_and(image, image, mask=mask_inv)

# Find contours
contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image (optional)
image_with_contours = image.copy()
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 3)

# Display the results
cv2.imshow("Original Image", image)
cv2.imshow("Mask", mask)
cv2.imshow("Mask Inverted", mask_inv)
cv2.imshow("Segmented Dog", result)
cv2.imshow("Contours", image_with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()
