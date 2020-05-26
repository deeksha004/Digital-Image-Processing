import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import glob
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score

path_fel = 'images/segemented_felzenswalb/*.jpg'
path_slic = 'images/segemented_slic/*.jpg'
path_ori = 'images/segmented_original/*.jpg'
path_gmm = 'images/segmented_gmm/*.jpg'

img_gt = plt.imread('/home/iiitb/Documents/DIP/slic/image_segmentation/images/train_masks_extra/00087a6bd4dc_02_mask.gif')
img_gt = img_gt[:,:,:3]
img_gt = cv2.cvtColor(img_gt,cv2.COLOR_RGB2GRAY)
#img_gt = cv2.resize(img_gt, (256, 256), interpolation = cv2.INTER_NEAREST)

ret, img_gt = cv2.threshold(img_gt,127,255,cv2.THRESH_BINARY)


for file in glob.glob(path_gmm):
	print(file)     #just stop here to see all file names printed
	image= cv2.imread("images/segmented_gmm_test.jpg")
	img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	#img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_NEAREST)
	rt, img_pred = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
	iou_1 = jaccard_score(img_gt.flatten(), img_pred.flatten(), pos_label=255)
	accuracy = accuracy_score(img_gt, img_pred)
	print("Accuracy_Original", accuracy)
	print('IOU(255): {}'.format(iou_1))
