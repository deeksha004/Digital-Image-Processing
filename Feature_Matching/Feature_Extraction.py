#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:03:15 2020

@author: Deeksha Aggarwal
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

class Feature_Extraction(object):
    
    def orb(img):
        orb = cv2.ORB_create()

        # find the keypoints with ORB
        #kp = orb.detect(img,None)
        
        # compute the descriptors with ORB
        kp, des = orb.detectAndCompute(img, None)
        
        # draw only keypoints location,not size and orientation
        #img2 = cv2.drawKeypoints(img,kp,None, flags=0)
        #plt.imshow(img2)
        #plt.show()
        return (kp, des)
        
    def sift(img):
        sift_ob = cv2.xfeatures2d.SIFT_create()
        
        # compute the keypoints and descriptors with SIFT
        kp, des = sift_ob.detectAndCompute(img, None)
    
        return (kp, des)
