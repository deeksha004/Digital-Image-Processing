import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
def kmeans_cl(image, k=5):
	
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	pixel_values = image.reshape((-1, 3))
	pixel_values = np.float32(pixel_values)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
	t0=time.time()
	_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
	t1=time.time()
	centers = np.uint8(centers)
	segmented_image = centers[labels.flatten()]
	segmented_image = segmented_image.reshape(image.shape)
	#segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
	plt.imshow(segmented_image)
	plt.title("Kmeans Segmented Image")
	plt.show()
	#plt.imsave('kmeans_color.png', segmented_image)
	print('Time taken by kmeans to segment image of size {} is: {}'.format(image.shape, t1-t0))

image = cv2.imread(sys.argv[1])
kmeans_cl(image, k=int(sys.argv[2]))
