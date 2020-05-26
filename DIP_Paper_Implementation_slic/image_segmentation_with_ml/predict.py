import numpy as np
import cv2
import pandas as pd
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import skimage.color as color
from sklearn.metrics import jaccard_score
from skimage.segmentation import slic

radius = 5
radius_lis = [2, 3, 4, 5, 6, 7]
n_points = 8 * radius
METHOD = 'uniform'

def slic_algo(image, numSegments=256, m=10.0, sigma = 0):

	# loop over the number of segments
	#for numSegments in (100):
	# apply SLIC and extract (approximately) the supplied number
	# of segments
	segments = slic(image, n_segments = numSegments, sigma = sigma,  compactness=m)# Sigma is smoothing Gaussian kernel 
	return(segments)

def feature_extraction(img):
    df = pd.DataFrame()


#All features generated must match the way features are generated for TRAINING.
#Feature1 is our original image pixels
    img2 = img.reshape(-1) # it is a gray image
    df['Original Image'] = img2

#Generate Gabor features
    num = 1
    kernels = []
    for theta in range(2):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for lamda in np.arange(0, np.pi, np.pi / 4):
                for gamma in (0.05, 0.5):
#               print(theta, sigma, , lamda, frequency)
                
                    gabor_label = 'Gabor' + str(num)
#                    print(gabor_label)
                    ksize=5
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                    kernels.append(kernel)
                    #Now filter image and add values to new column
                    fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img  #Modify this to add new column for each gabor
                    num += 1
########################################
#Geerate OTHER FEATURES and add them to the data frame
#Feature 3 is canny edge
    edges = cv2.Canny(img, 100,200)   #Image, min and max values
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1 #Add column to original dataframe

    from skimage.filters import roberts, sobel, scharr, prewitt

#Feature 4 and 5 is Y-cbCr and LAB
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    #edge_roberts1 = edge_roberts.reshape(-1)
    df['Y'] = img_ycbcr[:,:,0].reshape(-1)
    df['Cb'] = img_ycbcr[:,:,1].reshape(-1)
    df['Cr'] = img_ycbcr[:,:,2].reshape(-1)
    df['L'] = img_lab[:,:,0].reshape(-1)
    df['A'] = img_lab[:,:,1].reshape(-1)
    df['B'] = img_lab[:,:,2].reshape(-1)

#Feature 6 is Sobel
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1

#Feature 7 is LBP
    lbp = local_binary_pattern(img, n_points, radius, METHOD)
    df['lbp'] = lbp.reshape(-1)
    
#Feature 8 is Gaussian with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1

#Feature 9 is Gaussian with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3
    return df


#########################################################

#Applying trained model to segment multiple files. 

import glob
import pickle
import sys
from matplotlib import pyplot as plt
def help_message():
   print("Usage: [Input_Image] [Model Name]")
   print("Enter Path to the input image")
   print("Enter model name:")

if __name__ == '__main__':  
	# validate the input arguments
	filename = sys.argv[2]
	loaded_model = pickle.load(open(filename, 'rb'))

	path = sys.argv[1]
	for file in glob.glob(path):
		print(file)     #just stop here to see all file names printed
		img1= cv2.imread(file)
		img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

	#Call the feature extraction function.
		X = feature_extraction(img)
		result = loaded_model.predict(X)
		segmented = result.reshape((img.shape))
		
		name = file.split("_")
		plt.imsave('images/predicted_results/'+ name[1], segmented, cmap ='gray')
		print("The results are stored in directory 'images/predicted_results/'")
	#Above, we are splitting the file path into 2 -> creates a list with 2 entries
	#Then we are taking the second half of name to save segmented images with that name
