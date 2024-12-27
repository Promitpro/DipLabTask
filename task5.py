import cv2
import numpy as np

image = cv2.imread("E:\\OneDrive\\Documents\\4th year 1st semester\\digital image processing\\images\\DIP_Lab_task_05.jpg")
def hsv_conversion(image):
    height, width, _ = image.shape
    hsv_image = np.zeros_like(image, dtype=np.float32)

    for y in range(height):
        for x in range(width):
            b, g, r = image[y, x]
            b, g, r = b / 255.0, g / 255.0, r / 255.0
            cmax = max(r, g, b)
            cmin = min(r, g, b)
            delta = cmax - cmin

            if delta == 0:
                h = 0
            elif cmax == r:
                h = (60 * ((g - b) / delta) + 360) % 360
            elif cmax == g:
                h = (60 * ((b - r) / delta) + 120) % 360
            elif cmax == b:
                h = (60 * ((r - g) / delta) + 240) % 360

            if cmax == 0:
                s = 0
            else:
                s = delta / cmax

            v = cmax
            hsv_image[y, x] = [h, s * 255, v * 255]
    return hsv_image.astype(np.uint8)

def hsv_thresholding(hsv_image, lower_hsv, upper_hsv):
    height, width, _ = hsv_image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            h, s, v = hsv_image[y, x]
            if (lower_hsv[0] <= h <= upper_hsv[0] and
                lower_hsv[1] <= s <= upper_hsv[1] and
                lower_hsv[2] <= v <= upper_hsv[2]):
                mask[y, x] = 255
    return mask

def morphological_operations(mask, kernel, operation="open"):
    height, width = mask.shape
    kernel_height, kernel_width = kernel.shape
    pad_y = kernel_height // 2
    pad_x = kernel_width // 2
    padded_mask = np.pad(mask, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant')
    output_mask = np.zeros_like(mask)
    for y in range(height):
        for x in range(width):
            n = padded_mask[y:y+kernel_height, x:x+kernel_width]
            if operation == "open":
                eroded = np.min(n)
                if eroded == 255:
                    output_mask[y, x] = 255
            elif operation == "close":
                d = np.max(n)
                output_mask[y, x] = d
    return output_mask

hsv_image = hsv_conversion(image)
lower_white = np.array([0, 0, 60])
upper_white = np.array([180, 50, 255])
mask = hsv_thresholding(hsv_image, lower_white, upper_white)
mask_inv = 255 - mask
kernel = np.ones((5, 5), dtype=np.uint8)
mask_inv = morphological_operations(mask_inv, kernel, "open")
mask_inv = morphological_operations(mask_inv, kernel, "close")

result = np.zeros_like(image)
height, width, _ = image.shape
for y in range(height):
    for x in range(width):
        if mask_inv[y, x] == 255:
            result[y, x] = image[y, x]

cv2.imshow("Original Image", image)
cv2.imshow("Mask Inverted", mask_inv)
cv2.imshow("Segmented Dog", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Reference
# www.allaboutcircuits.com
# www.gemini.ai
