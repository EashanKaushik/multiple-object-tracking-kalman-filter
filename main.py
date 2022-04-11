
import numpy as np

# X_hat = np.array([[0], [1], [2], [3]])

# print(X_hat[1][0])

# B_t = np.array([[10], [20], [30], [40]])

# print((B_t.dot(1).reshape(B_t.shape[0], -1)))

# a = np.array([[[1.,  1.,  1.,  1.],
#               [1.,  1.,  1.,  1.],
#               [1.,  1.,  1.,  1.],
#               [1.,  1.,  1.,  1.]], [[1.,  1.,  1.,  1.],
#               [1.,  1.,  1.,  1.],
#               [1.,  1.,  1.,  1.],
#               [1.,  1.,  1.,  1.]], [[1.,  1.,  1.,  1.],
#               [1.,  1.,  1.,  1.],
#               [1.,  1.,  1.,  1.],
#               [1.,  1.,  1.,  1.]]])

# print(a.shape)

# print(np.pad(a, (4, 4, 4),  mode='constant', constant_values=0).shape)


# Python program to explain cv2.copyMakeBorder() method

# importing cv2
import cv2

# path
path = r'geeks.png'

# Reading an image in default mode
image = cv2.imread(path)

# Window name in which image is displayed
window_name = 'Image'

print(image.shape)

# Using cv2.copyMakeBorder() method
image = cv2.copyMakeBorder(image, 0, 0, 0, 0,
                           cv2.BORDER_CONSTANT, None, value=0)

print(image.shape)

# Displaying the image
cv2.imshow(window_name, image)
cv2.waitKey(0)
