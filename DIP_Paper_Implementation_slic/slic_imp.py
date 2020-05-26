# import the necessary packages
import cv2
from skimage.segmentation import slic
from skimage.segmentation import felzenszwalb, quickshift
from sklearn.cluster import SpectralClustering
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import argparse
from skimage.transform import resize
import time
import numpy as np
import sys
import skimage.color as color

#from scipy.misc import imresize

# construct the argument parser and parse the arguments
image = plt.imread(sys.argv[1])

def slic_algo(image, numSegments, m=10.0, sigma = 0):

	t0=time.time()
	segments = slic(image, n_segments = numSegments, sigma = sigma,  compactness=m)# Sigma is smoothing Gaussian kernel 
	t1=time.time()
	fig = plt.figure("Superpixels -- %d segments" % (numSegments))
	ax = fig.add_subplot(1, 1, 1)
	ax.imshow(mark_boundaries(image, segments))
	new_img = color.label2rgb(segments, image, kind='avg');
	#plt.imshow(new_img)
	plt.title("SLIC Segmented")
	plt.axis("off")
	# show the plots
	plt.show()
	#plt.imsave("slic(b)_64.png", mark_boundaries(image, segments))
	print('Time taken by slic to segment image of size {} is: {}'.format(image.shape, t1-t0))
	    
def gs04(image, scale=150, sigma=0):
    t0=time.time()
    segments = felzenszwalb(image, scale=scale, sigma=sigma, min_size=20)
    t1=time.time()
    # show the output 
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segments))
    plt.title("Felzenswalb Segmented")
    plt.axis("off")
    # show the plots
    plt.show()
    #plt.imsave("gs04.png", mark_boundaries(image, segments))
    print('Time taken by slic to segment image of size {} is: {}'.format(image.shape, t1-t0))
	
def qs09(image, sigma=0, max_dist=10):
    t0 = time.time()
    segments = quickshift(image, sigma=sigma, max_dist=max_dist)
    t1 = time.time()
    # show the output 
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segments))
    plt.title("Quick Shift Segmented")
    plt.axis("off")
    # show the plots
    plt.show()
    plt.imsave("qs09.png", mark_boundaries(image, segments))
    print('Time taken to qs segment {} size image is: {}'.format(image.shape, t1-t0))
	
def nc05(img, n_clusters=5,
                          n_neighbors=5,
                          gamma=1,
                          affinity='nearest_neighbors',
                          visualize=True,
                          include_spatial=False):
    """
    Normalized cut algorithm for image segmentation
    include_spatial : (Bonus)
    """
    img = resize(img, (int(img.shape[0] * 0.3), int(img.shape[1] * 0.3)), anti_aliasing=True)
    #img = imresize(img, 0.3) / 255
    n = img.shape[0]
    m = img.shape[1]

    if include_spatial:
        xx = np.arange(n)
        yy = np.arange(m)
        X, Y = np.meshgrid(yy, xx)
        img = np.concatenate((Y.reshape(n, m, 1), X.reshape(n, m, 1), img), axis=2)
        print("spectral_segment(:include_spatial) img.shape = {}".format(img.shape))

    img = img.reshape(-1, img.shape[-1])

    # Notes:
    # gamma is ignored for affinity='nearest_neighbors'
    # n_neighbors is ignore for affinity='rbf'
    # n_jobs = -1 means using all processors :D
    t0 = time.time()
    labels = SpectralClustering(n_clusters=n_clusters, affinity=affinity,gamma=gamma,n_neighbors=n_neighbors,n_jobs=-1,eigen_solver='arpack').fit_predict(img)
    t1 = time.time()
    labels = labels.reshape(n, m)
    if visualize == True:
    	fig = plt.figure()
    	ax = fig.add_subplot(1, 1, 1)
    	ax.imshow(labels)
    	plt.axis("off")
    	plt.title("Normal Cut Segmentation")
    	plt.show()
    	plt.imsave("ns05.png", labels)
    print('Time taken to nc segment {} size image is: {}'.format(img.shape, t1-t0))
    return (labels)
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
		
def line_pl(x, y, title, xlabel, ylabel):
	plt.plot(x, y, marker='o')
	plt.title(title)
	plt.xticks(x)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()
	
labels = nc05(img=image)
qs09(image)
#kmeans_cl(image, k=int(sys.argv[2]))
gs04(image, scale=256)
slic_algo(image, numSegments = int(sys.argv[2]), m=int(sys.argv[3]), sigma=0)
x= ['(481, 321, 3)', '(853, 1280, 3)', '(1333, 2000, 3)', '(2225, 3320, 3)']
y_slic= [0.726, 3.83, 5.30, 73.47]
x_kmean= ['(481, 321, 3)']
y_kmean = [121.27]
x_gs = ['(481, 321, 3)', '(853, 1280, 3)', '(1333, 2000, 3)']
y_gs = [0.395, 13.62, 37.18]
plt.plot(x, y_slic, marker='o', label='SLIC')
plt.plot(x_kmean, y_kmean, marker='o', label='Kmeans')
plt.plot(x_gs, y_gs, marker='o', label='Felzenswalb')
plt.legend()
plt.title('Time required to generate superpixels for images of increasing size.')
plt.xticks(x)
plt.xlabel('Image Size')
plt.ylabel('Time in Seconds')
plt.show()

