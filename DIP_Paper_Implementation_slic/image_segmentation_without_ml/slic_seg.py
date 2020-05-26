from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb, quickshift
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import time
import numpy as np
import pandas as pd
import skimage.color as color
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
import sys

def slic_algo(image, numSegments, m=10.0, sigma = 0):

	# loop over the number of segments
	#for numSegments in (100):
	# apply SLIC and extract (approximately) the supplied number
	# of segments
	t0=time.time()
	segments = slic(image, n_segments = numSegments, sigma = sigma,  compactness=m)# Sigma is smoothing Gaussian kernel 
	t1=time.time()
	print(segments.shape)
	print(type(segments))
	# show the output of SLIC
	#plt.imsave("slic.png", mark_boundaries(image, segments))
	#print('Time taken to slic segment {} size image is: {}'.format(image.shape, t1-t0))
	return(segments)
def gs04(image, scale=150, sigma=0):
    t0=time.time()
    segments = felzenszwalb(image, scale=scale, sigma=sigma, min_size=20)
    #t1=time.time()
    # show the output 
    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1)
    #ax.imshow(mark_boundaries(image, segments))
    #plt.axis("off")
    # show the plots
    #plt.show()
    #plt.imsave("gs04.png", mark_boundaries(image, segments))
    #print('Time taken by slic to segment image of size {} is: {}'.format(image.shape, t1-t0))
    return(segments)
    
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
	plt.show()
	#plt.imsave('kmeans_color.png', segmented_image)
	#print('Time taken by kmeans to segment image of size {} is: {}'.format(image.shape, t1-t0))
	
	return(segmented_image)

"""
#Accessing Individual Superpixel Segmentations
"""
"""
image = plt.imread('images/rice.png')
segments = slic_algo(image, 256)
for (i, segVal) in enumerate(np.unique(segments)):
	# construct a mask for the segment
	print ("[x] inspecting segment %d" % (i))
	mask = np.zeros(image.shape[:2], dtype = "uint8")
	mask[segments == segVal] = 255
	# show the masked region
	cv2.imshow("Mask", mask)
	cv2.imshow("Applied", cv2.bitwise_and(image, image, mask = mask))
	cv2.waitKey(0)

"""

image = plt.imread(sys.argv[1])#'images/car.jpg'
plt.imshow(image)
#ret, img_gt = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
img_gt = plt.imread(sys.argv[2])#'masked/car.jpg'
ret, img_gt = cv2.threshold(img_gt,127,255,cv2.THRESH_BINARY)

segments = slic_algo(image, int(sys.argv[3]), float(sys.argv[4]), 5)
#segments = gs04(image, sigma=5)

new_img = color.label2rgb(segments, image, kind='avg');
#blur = cv2.GaussianBlur(new_img,(5,5),0)
ret, img_pred = cv2.threshold(cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#ret, img_pred = cv2.threshold(new_img,127,255,cv2.THRESH_BINARY)
"""
for kmeans

image = plt.imread('images/car.jpg')
plt.imshow(image)
#ret, img_gt = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
img_gt = plt.imread('masked/car.jpg')
ret, img_gt = cv2.threshold(img_gt,127,255,cv2.THRESH_BINARY)

new_img = kmeans_cl(image, k=64)
ret, img_pred = cv2.threshold(cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
"""
#fig = plt.figure("Superpixels -- %d segments" % (numSegments))
plt.subplot(2, 3, 1),plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis("off")
plt.subplot(2, 3, 2),plt.imshow(img_gt, cmap='gray')
plt.title('Binary Ground Truth Image')
plt.axis("off")
plt.subplot(2, 3, 3),plt.imshow(mark_boundaries(image, segments), cmap='gray')
#plt.subplot(2, 3, 3),plt.imshow(new_img, cmap='gray') # for kmeans
plt.title('kmeans Segmented Image')
plt.axis("off")
plt.subplot(2, 3, 4),plt.imshow(new_img, cmap='gray')
plt.title('Superpixels Averaged Image')
plt.axis("off")
plt.subplot(2, 3, 5),plt.imshow(img_pred, cmap='gray')
plt.title('Binary Predicted Image')
plt.axis("off")
# show the plots
plt.show()


iou_0 = jaccard_score(img_gt.flatten(), img_pred.flatten(), pos_label=0)
iou_1 = jaccard_score(img_gt.flatten(), img_pred.flatten(), pos_label=255)
#accuracy = accuracy_score(img_gt, img_pred)
#print(accuracy)
print('IOU(255): {}'.format(iou_1))
print('IOU(0): {}'.format(iou_0))
#plt.imshow(bw_img, cmap='gray')
#plt.show()


