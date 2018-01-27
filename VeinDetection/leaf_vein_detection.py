# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:48:52 2017
# Accomplish soma reconstruction,


@author: xli63

"""
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import util,segmentation,exposure,filters, morphology,measure,feature,io,draw
from scipy import ndimage,stats,cluster,misc,spatial

import sys
import os
import cv2


def generate_single_cellMask  (OriginalImg ):
    
    '''create whole_mask masks''' 
    gray = skimage.color.rgb2gray (OriginalImg)                                  # plt.figure(),plt.imshow(gray, cmap='gray')
    whole_mask = gray <  (filters.threshold_otsu(gray)  )                 # plt.figure(),plt.imshow(whole_mask, cmap='gray')
    labels = measure.label (whole_mask)
    #labels = morphology.dilation (labels, morphology.disk(2))             #plt.figure(),plt.imshow(labels)
    
    ''' find singleCell(leaf) mask'''    
    #find the label ID of the center components (supposed to be leaf)
    centerImg = np.array( [np.uint(gray.shape[0]/2),np.uint(gray.shape[1]/2)])
    centerLabelID = labels[centerImg[0],centerImg[1]]    
    if centerLabelID == 0 :  # just in case the center component does't cover the center point of img
        blobs = np.zeros((labels.max() ,2)) 
        for obj_id , obj in enumerate( measure.regionprops (labels)  ) :
            if obj.area > 500:
                blobs[obj_id, :] = obj.centroid
        Y = spatial.distance.cdist(blobs[:,0:2], [centerImg], 'euclidean')
        centerLabelID = np.argmin(Y) + 1
        
    prop_center = measure.regionprops (labels) [centerLabelID-1]    
    singleCellMask = (labels==centerLabelID) *1                          #plt.figure(),plt.imshow(singleCellMask)
    
#    if measure.label (singleCellMask)
    singleCellMask = morphology.remove_small_holes(singleCellMask, 20)    # remove small objects   #plt.figure(),plt.imshow(singleCellMask)
   
    return singleCellMask



def vein_detection(OriginalImg, singleCellMask = []):
    gray = skimage.color.rgb2gray (OriginalImg) 
    
    if singleCellMask != []:
        thres =  filters.threshold_minimum(gray)         
        leaf_mask = ~(gray > thres )                         #  plt. figure(),plt.imshow(leaf_mask)
        leaf_mask =  morphology.binary_closing (leaf_mask, morphology.disk(2))    # remove small dark spots
    else:
        leaf_mask = generate_single_cellMask  (OriginalImg )
        
    vein_image =  - (gray *leaf_mask)
    imadjusted = exposure.rescale_intensity (vein_image)
    vein = feature.canny(imadjusted)
    
    vein = vein * 255
    return vein


if __name__ == '__main__':
    
    Img_loc_root = 'D:\materials of courses of Rebecca\Deeplearning Hien[ECE6397]\Project\dataset'
    write_path = os.path.join(Img_loc_root, 'vein')

    
    read_path = os.path.join(Img_loc_root, 'Original')
    for OriginalImg_fileName  in os.listdir(read_path) :
        OriginalImg = io.imread( read_path + './'+  OriginalImg_fileName)                                         # plt.figure()  , plt.imshow(OriginalImg,cmap = 'gray')
        veinImg = vein_detection(OriginalImg)
        cv2.imwrite(write_path + './' + OriginalImg_fileName,veinImg ) 
#        print (OriginalImg_fileName)1
