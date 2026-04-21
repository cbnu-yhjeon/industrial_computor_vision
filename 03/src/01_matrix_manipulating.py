import cv2
import numpy as np

# White image
image = np.full((300, 300, 3), 255, np.uint8)
cv2.imshow("white", image)
cv2.waitKey()
cv2.destroyAllWindows()

# Red image
image = np.full((300, 300, 3), [0, 0, 255], np.uint8)
cv2.imshow("red", image)
cv2.waitKey()
cv2.destroyAllWindows()

# Black image
image.fill(0)
cv2.imshow("black", image)
cv2.waitKey()
cv2.destroyAllWindows()

# Partial manipulation using indexing
image[200, 100] = [255, 255, 255]
cv2.imshow("image", image)
cv2.waitKey()
cv2.destroyAllWindows()

image[:, 100] = 255
cv2.imshow("blue with white line", image)
cv2.waitKey()
cv2.destroyAllWindows()

image[100:600, 100:200, 1] = 255
cv2.imshow("image", image)
cv2.waitKey()
cv2.destroyAllWindows()
