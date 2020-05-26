import numpy as np
import cv2
from sklearn.mixture import GaussianMixture as GMM
import glob
import skimage.color as color

#Use plant cells to demo the GMM on 2 components
#Use BSE_Image to demo it on 4 components
#USe alloy.jpg to demonstrate bic and how 2 is optimal for alloy
path = "images/test/*.jpg"
for file in glob.glob(path):
    print(file)     #just stop here to see all file names printed
    image = cv2.imread(file)
    
    #Convert MxNx3 image into Kx3 where K=MxN
    img2 = image.reshape((-1,3))
    #covariance choices, full, tied, diag, spherical
    gmm_model = GMM(n_components=2,covariance_type='tied').fit(img2)
    segments= gmm_model.predict(img2)
    #img1 = color.label2rgb(segments, image, kind='avg');
    #Put numbers back to original shape so we can reconstruct segmented image
    original_shape = image.shape
    segmented = segments.reshape(original_shape[0], original_shape[1])
    
    cv2.imwrite("images/segmented_gmm.jpg", segmented)
