#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:04:34 2020

@author: iiitb
"""


import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import json
import numpy as np
import math
#import imgaug as ia
#from imgaug import augmenters as iaa


# In[2]:


def plot_pair(images, gray=False):

    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(10,8))
    i=0
    
    for y in range(2):
        if gray:
            axes[y].imshow(images[i], cmap='gray')
        else:
            axes[y].imshow(images[i])
        axes[y].get_xaxis().set_visible(False)
        axes[y].get_yaxis().set_visible(False)
        i+=1
    
    plt.show()


# In[3]:


def get_poly(ann_path):
    
    with open(ann_path) as handle:
        data = json.load(handle)
    
    shape_dicts = data['shapes']
    
    return shape_dicts


# In[4]:


def create_binary_masks(im, shape_dicts):
    
    blank = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=np.float32)
    
    for shape in shape_dicts:
        points = np.array(shape['points'], dtype=np.int32)
        #cv2.fillPoly(blank, [points], 255)
        
        if shape['shape_type']=='circle':
            center = tuple(points[0])
            p1 = tuple(points[1])
            radius = int(math.sqrt( ((center[0]-p1[0])**2)+((center[1]-p1[1])**2) ))
            blank = cv2.circle(blank, center, radius, 255, -1)
        else:
            cv2.fillPoly(blank, [points], 255)
    return blank


# ### Create Masks for Binary Classification

# In[5]:


image_list = sorted(os.listdir('./images'), key=lambda x: x.split('.')[0])
annot_list = sorted(os.listdir('./annotated'), key=lambda x: x.split('.')[0])

im_fn = image_list[0]

ann_fn = annot_list[0]

for im_fn, ann_fn in zip(image_list, annot_list):
    
    im = cv2.imread(os.path.join('./images', im_fn), 0)
    
    ann_path = os.path.join('./annotated', ann_fn)
    shape_dicts = get_poly(ann_path)
    im_binary = create_binary_masks(im, shape_dicts)
    
    cv2.imwrite('./masked/{}'.format(im_fn), im_binary)
    plot_pair([im, im_binary], gray=True)
    plt.show()
    
    break


