# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 18:39:35 2017

@author: xli63
"""
import os
os.chdir(r'D:\materials of courses of Rebecca\Deeplearning Hien[ECE6397]\Project')  # set current working directory
import matplotlib.pyplot as plt
import cv2
import numpy as np
import skimage
from skimage import util,segmentation,exposure,filters, morphology,measure,feature,io
from scipy import ndimage,stats,cluster,misc,spatial

from sklearn.model_selection import train_test_split


read_img_loc_root = 'D:\materials of courses of Rebecca\Deeplearning Hien[ECE6397]\Project\Image_prepared'
write_img_loc_root = 'D:\materials of courses of Rebecca\Deeplearning Hien[ECE6397]\Project\dataset'


write_loc = {}
for key  in ['Original','border', 'GANinput']:
    write_loc[key]  = write_img_loc_root + './' + key       
    if os.path.isdir(write_loc[key] ) == False:
        os.makedirs(write_loc[key] )           

# put all order leaf in one folder and bring order name to the file name
for orderName  in os.listdir(read_img_loc_root) :    
    for imageType in os.listdir(read_img_loc_root + './' + orderName):
        print(imageType)
        subfolder_order = read_img_loc_root + './' + orderName + './' + imageType
        for  sampleID_inclass, image_Name in enumerate(os.listdir(subfolder_order)):
            image_fullName = (subfolder_order + './' + image_Name)
            new_image_Name = image_Name
            new_image_Name = new_image_Name.replace(image_Name.split('-')[0],'')
            new_fullName = write_loc[imageType] + './' + orderName + '_' + str(sampleID_inclass+1) + new_image_Name  # Image ID, sample index, time (1...4)
            os.rename (image_fullName,new_fullName )  # move it into here 
            

# split it up to  train,test and val          
X = []

X_train, X_test_val, y_train, y_test_val = train_test_split( X, y, test_size=0.33, random_state=42)
X_val, X_test, y_val, y_test = train_test_split( X_test_val, y_train, test_size=0.5, random_state=42)
            
            
        if 'Original' in child_folder: # get the child_folder 
            read_img_loc  = read_img_loc_root + './' + subfolder_order + './' + child_folder
            
            for key  in ['Original','border', 'GANinput']:
                write_loc[key]  = write_img_loc_root + './' + subfolder_order +  './' +key       
                if os.path.isdir(write_loc[key] ) == False:
                    os.makedirs(write_loc[key] )           
 

            for fileName_img in os.listdir(read_img_loc):
                
#                fileName_img = 'ny1072-08-2.jpg'
                
                # generate cropped image 
                canvas_size = [256,256]      
                
                OriginalImg = io.imread(read_img_loc +'./' + fileName_img)                
                Gan_input , cropped_image, border = image_preparing (OriginalImg , canvas_size,    visual=False   )
                
                
                
                # write cropped image 
                writeImgdone = cv2.imwrite( write_loc['GANinput'   ]  + './' + fileName_img, Gan_input)  
                writeImgdone = cv2.imwrite( write_loc['border'      ]  + './' + fileName_img, border) 
                writeImgdone = cv2.imwrite( write_loc['Original'    ]  + './' + fileName_img, cropped_image) 


                if writeImgdone == True:
                    print('Generate Cropped Images for '+ fileName_img +' done!')
                else:
                    print('[Caucious! ]  Generate Cropped Images  failed!!!! for originalImgName')   
                
