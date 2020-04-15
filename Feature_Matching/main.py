#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 21:23:53 2020

@author: Deeksha Aggarwal
"""

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from Feature_Extraction import Feature_Extraction as fe
import pandas as pd

train_dir = './train/'
test_dir = './test/'

train_imgs = os.listdir(train_dir)
test_imgs = os.listdir(test_dir)

kp_dict = dict()

des_dict = dict()

train_matches = list()

train_num_matches = list()


#Extract features from test images
test_img = plt.imread('./test/light.jpeg')
kp, des = fe.orb(test_img)
img_kp=cv2.drawKeypoints(test_img,kp[:10],None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('test_keypoints.jpg',cv2.cvtColor(img_kp,cv2.COLOR_BGR2RGB))
# create BFMatcher object


# Match descriptors with each image in the dataset.
for i in range(0, 4):

	# Read the train image
	train_image = plt.imread(train_dir+train_imgs[i])
	train_kp, train_des = fe.orb(train_image)
	
	img_kp=cv2.drawKeypoints(train_image,train_kp[:10],None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv2.imwrite('train_keypoints.jpg',cv2.cvtColor(img_kp,cv2.COLOR_BGR2RGB))
	plt.imshow(img_kp),plt.show()
	# Apply bf matcher
	"""bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des,train_des)
	#Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)
	"""
	#Apply knn bf matcher
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des,train_des, k=2)
	# Apply ratio test
	good = []
	for m,n in matches:
		if m.distance < 0.80*n.distance:
			good.append([m])
	#train_num_matches.append(len(matches))

	
	#Draw the matches
	#img3 = np. zeros(shape=[512, 512, 3], dtype=np. uint8)
	#img3 = cv2.drawMatches(test_img,kp,train_image,train_kp,matches[:10], None, flags=2)
	img3 = cv2.drawMatchesKnn(test_img,kp,train_image,train_kp,good, None, flags=2)
	
	plt.imshow(img3),plt.show()
	#plt.imsave('bf_matcher_results/{}_{}.png'.format(train_image, test_img), img3)

print(train_num_matches)

# Draw first 10 matches.
#img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)

#img3 = cv2.drawMatches(image,kp_dict_class2[train_class2_imgs[4]],test_img,kp,matches[:10], test_img, flags=0)

#plt.imshow(img3)
#plt.show()
