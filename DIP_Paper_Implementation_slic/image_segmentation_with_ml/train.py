import numpy as np
import cv2
import pandas as pd
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import skimage.color as color
from sklearn.metrics import jaccard_score
from skimage.segmentation import slic
from skimage.segmentation import felzenszwalb, quickshift

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
def gs04(image, scale=150, sigma=0):
    segments = felzenszwalb(image, scale=scale, sigma=sigma, min_size=20)
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
from matplotlib import pyplot as plt

df_train = pd.DataFrame()

path = "images/train/*.jpg"
for file in glob.glob(path):
    print(file)     #just stop here to see all file names printed
    image= cv2.imread(file)
    
    #segments = slic_algo(image)
    #img1 = color.label2rgb(segments, image, kind='avg');
    segments = gs04(image)
    img1 = color.label2rgb(segments, image, kind='avg');
    plt.imshow(img1, cmap='gray')
    img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_NEAREST)
    #plt.imshow(img, cmap='gray')

#Call the feature extraction function.
    X = feature_extraction(img)
    #result = loaded_model.predict(X)
    print(X)

    #segmented = result.reshape((img.shape))
    df_train = df_train.append(X, ignore_index = True, sort=False)
    print(df_train.shape) 
    #name = file.split("e_")
    #plt.imsave('images/Segmented/'+ name[1], segmented, cmap ='jet')

#Above, we are splitting the file path into 2 -> creates a list with 2 entries
#Then we are taking the second half of name to save segmented images with that name

# Generating the labels
path_mask = "images/train_masks/*.gif"
for file in glob.glob(path_mask):
    print(file)     #just stop here to see all file names printed
    #Convert it to 3d format
    img1= plt.imread(file)
    img1 = img1[:,:,:3]
    #print(img1.shape)
    #plt.imshow(img1)
    #plt.show()
    img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_NEAREST)
    labeled_img1 = img.reshape(-1)
    #print(labeled_img1)
    df_train['Labels'] = labeled_img1
    #print(df_train.shape)
df_train.reset_index(drop=True, inplace=True)
print(df_train.shape)
print(df_train.head())


#Saving the data on the disk
df_train.to_csv("training_data.csv", index=False)

df_train = pd.read_csv("training_data.csv")
#Define the dependent variable that needs to be predicted (labels)
Y = df_train["Labels"].values

#Define the independent variables
X = df_train.drop(labels = ["Labels"], axis=1)
 
# Splitting the X into 70:30
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20)

# Aplying Random Forest Algorithm to train the model
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with n number of decision trees
model = RandomForestClassifier(n_estimators = 50, random_state = 42)

# Train the model on training data
model.fit(X_train, y_train)

#First test prediction on the training data itself. SHould be good. 
prediction_test_train = model.predict(X_train)

#Test prediction on testing data. 
prediction_test = model.predict(X_test)

#Let us check the accuracy on test data
from sklearn import metrics
#First check the accuracy on training data. This will be higher than test data prediction accuracy.
print ("Accuracy on training data = ", metrics.accuracy_score(y_train, prediction_test_train))
#Check accuracy on test dataset. If this is too low compared to train it indicates overfitting on training data.
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))



# Plot the feature importance graph

feature_list = list(X.columns)
feature_imp = pd.DataFrame({'feature_name': feature_list, 
							'feature_imp': model.feature_importances_}).sort_values(by = 'feature_imp', ascending=False)
print(feature_imp)
feature_imp.plot.bar(x= 'feature_name', y='feature_imp')
plt.show()

import pickle

#Save the trained model as pickle string to disk for future use
filename = "seg_rf_model_felzenswalb"
pickle.dump(model, open(filename, 'wb'))

#To test the model on future datasets
#loaded_model = pickle.load(open(filename, 'rb'))

#result = loaded_model.predict(X)

#segmented = result.reshape((img.shape))

#from matplotlib import pyplot as plt
#plt.imshow(segmented, cmap ='jet')
#plt.imsave('segmented_rock_RF_100_estim.jpg', segmented, cmap ='jet')

