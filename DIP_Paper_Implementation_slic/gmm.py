import numpy as np
import cv2
from sklearn.mixture import GaussianMixture as GMM
import glob
import skimage.color as color
import matplotlib.pyplot as plt
import sys

#Use plant cells to demo the GMM on 2 components
#Use BSE_Image to demo it on 4 components
#USe alloy.jpg to demonstrate bic and how 2 is optimal for alloy

#print(file)     #just stop here to see all file names printed
image = plt.imread(sys.argv[1])

#Convert MxNx3 image into Kx3 where K=MxN
img2 = image.reshape((-1,3))
#covariance choices, full, tied, diag, spherical
gmm_model = GMM(n_components=int(sys.argv[2]),covariance_type='full').fit(img2)
segments= gmm_model.predict(img2)
segmented = segments.reshape(image.shape[0], image.shape[1])

img1 = color.label2rgb(segmented, image, kind='avg');
#Put numbers back to original shape so we can reconstruct segmented image
#original_shape = image.shape

plt.imshow(img1)
plt.title("GMM Segmented Image")
plt.show()
#cv2.imwrite("images/segmented_gmm.jpg", segmented)
