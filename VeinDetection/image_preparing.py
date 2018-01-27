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



def cropImg (original_image,  cropRange):
    
    if len(cropRange) == 4 :  # in the form of    (min_row, min_col, max_row, max_col), need to change  to  [[ymin, ymax],[xmin,xmax]]    
        xmin = cropRange[0]
        xmax = cropRange[2]
        ymin = cropRange[1]
        ymax = cropRange[3]
        cropRange_temp = [[xmin,xmax],[ymin, ymax]] 
        cropRange = cropRange_temp
#			print (cropRange)       
            
    if len(original_image.shape) ==3 :
        image = original_image[ int ( cropRange[0][0] ) :int (cropRange[0][1] ),
                                int ( cropRange[1][0] ) :int (cropRange[1][1] ),          
                                :]    
    else:
        image = original_image[ int ( cropRange[0][0] ) :int (cropRange[0][1] ),
                                int ( cropRange[1][0] ) :int (cropRange[1][1] )          ]     
    
    return image


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
   
    return prop_center , singleCellMask


def vein_detection(OriginalImg, leaf_mask = []):
    gray = skimage.color.rgb2gray (OriginalImg) 
    
#    if singleCellMask != []:
#        thres =  filters.threshold_minimum(gray)         
#        leaf_mask = ~(gray > thres )                         #  plt. figure(),plt.imshow(leaf_mask)
#        leaf_mask =  morphology.binary_closing (leaf_mask, morphology.disk(2))    # remove small dark spots
#    else:  # this one is better
#        __, leaf_mask = generate_single_cellMask  (OriginalImg ) 
#        
    vein_image =  - (gray *leaf_mask)
    imadjusted = exposure.rescale_intensity (vein_image)
    vein = feature.canny(imadjusted)
    
    vein = vein * 255
    return vein

def image_preparing (OriginalImg , canvas_size  ,visual = False):
    
    '''initalization '''
#    Gan_input_size =  [canvas_size[0],canvas_size[1] *2, 3]
#    Gan_input = np.zeros(Gan_input_size)                 
    
    inital_cropRange = (0,0,np.shape(OriginalImg)[0] - 200, np.shape(OriginalImg)[1] - 190)         #bbox (x1,y1, x2, y2)

    # Crop the original image and generate mase
    OriginalImg_noRuler = cropImg (OriginalImg,  inital_cropRange)                       #  plt.figure(),plt.imshow(OriginalImg_noRuler)
    
    prop_center , singleCellMask = generate_single_cellMask  (OriginalImg_noRuler)    #  plt.figure(),plt.imshow(singleCellMask)
    
    
#   '''Crop or resize the image '''
#    if       ( prop_center.bbox[2] - prop_center.bbox[0]  ) <= canvas_size [0]  and ( prop_center.bbox[3] - prop_center.bbox[1] ) <= canvas_size [1]  and  singleCellMask.sum() > 150 :
#        print ('just crop')    
#        # bbox size with in crop image   bbox (x1,y1, x2, y2)
#        cropRange = [  [ prop_center.centroid[0] - canvas_size [0] /2  ,   prop_center.centroid[0] + canvas_size [0] /2    ],
#                       [ prop_center.centroid[1] - canvas_size [1] /2  ,   prop_center.centroid[1] + canvas_size [1] /2    ] ]
#        cropped_image =  cropImg (OriginalImg,  cropRange)
#        cropped_mask =  cropImg (singleCellMask,  cropRange)
#        
#        border = np.uint8(  skimage.segmentation.find_boundaries (cropped_mask) * 255  )
#        border_3CH = cv2.merge((border,border,border))         
#        
#     # leaf size  larger than canvas   # resize and shrink it     
#    else: 
#        
#        if singleCellMask.sum() > 150 :  # succeuss 
    cropRange = np.array( prop_center.bbox)  # orignial bbox
    
    # englarge bbox
    bbox_erode_wid = 20
    cropRange[0] = max(0, cropRange[0] - bbox_erode_wid )
    cropRange[1] = max(0, cropRange[1] - bbox_erode_wid )
    cropRange[2] = cropRange[2] + bbox_erode_wid
    cropRange[3] = cropRange[3] + bbox_erode_wid
    
    # keep the crop ratio to the ratio of canvas 
    width =  cropRange[2] - cropRange[0] 
    height =   cropRange[3] - cropRange[1] 
    if  width  >  height :  # enlarge the width
        cropRange[3] = cropRange[1]  + width
    else:
        cropRange[2] = cropRange[0]  + height                      
            

    cropped_image =   cropImg (OriginalImg_noRuler,  cropRange)      #  plt.figure(),plt.imshow(cropped_image)
    cropped_mask =    cropImg (singleCellMask,  cropRange)        #  plt.figure(),plt.imshow(cropped_mask)
    
    
    # binary images
    border  = np.uint8(  skimage.segmentation.find_boundaries (cropped_mask) * 255  )
    vein    = vein_detection(cropped_image,leaf_mask =cropped_mask )
    

    border_3CH = cv2.merge((border,border,border)) 
    vein_3CH = cv2.merge((vein,vein,vein)) 

    
    # resize the original imaage , border,and vein, to make it exact the same as the canvas
    cropped_image_resized   = misc.imresize (  cropped_image,  [canvas_size[0],canvas_size[1], 3])
    border_3CH_resized      = misc.imresize (  border_3CH   ,  [canvas_size[0],canvas_size[1], 3])
    vein_3CH_resized        = misc.imresize (  vein_3CH   ,  [canvas_size[0],canvas_size[1], 3])
    
    
    GANinput_border = np.concatenate ( (cropped_image_resized , border_3CH_resized ), axis = 1)  #  plt.figure(),plt.imshow(Gan_input)
    GANinput_vein   = np.concatenate ( (cropped_image_resized , vein_3CH_resized   ), axis = 1)  #  plt.figure(),plt.imshow(Gan_input)

    #

    ''' visulization intermediate results'''    
    if visual == True: 
        f, ((ax1, ax2), (ax3, ax4),(ax5, ax6)) = plt.subplots( 2, 3, sharex=True, sharey=True )
                
        ax2.imshow(GANinput_border)
        ax2.set_title('GANinput_border')
        ax2.axis('off')
                
        ax3.imshow(border_3CH  ,cmap = 'gray')
        ax3.set_title('singleCellMask')
        ax3.axis('off')

        ax5.imshow(GANinput_vein)
        ax5.set_title('GANinput_vein')
        ax5.axis('off')
                
        ax6.imshow(vein_3CH  ,cmap = 'gray')
        ax6.set_title('vein_3CH')
        ax6.axis('off')
        
        ax1.imshow(OriginalImg_noRuler)
        ax1.set_title('OriginalImg_noRuler')
        ax1.axis('off')
        
        ax4.imshow(OriginalImg,cmap = 'gray' )
        ax4.set_title('OriginalImg')
        ax4.axis('off')    
    #    plt.close ('all')    
    
    return cropped_image, border,vein, GANinput_border,GANinput_vein
    


if __name__ == '__main__':
   
    
    read_img_loc_root = 'D:\materials of courses of Rebecca\Deeplearning Hien[ECE6397]\Project\Project_Dataset'
    write_img_loc_root = 'D:\materials of courses of Rebecca\Deeplearning Hien[ECE6397]\Project\Image_prepared'
    
    write_loc = {}
    
    X_Original = [] # dataset
    X_border = [] # dataset
    X_vein =[]
    X_GANinput_border = []
    X_GANinput_vein = []
    y = [] # label
       
    
    for key  in ['Original','border','vein', 'GANinput_border','GANinput_vein']:
        write_loc[key]  = write_img_loc_root + './' + key       
        if os.path.isdir(write_loc[key] ) == False:
            os.makedirs(write_loc[key] )          
    
    for classID , subfolder_order  in enumerate(os.listdir(read_img_loc_root) ):   # order Name (class name)

    #    print (subfolder_order)
        for child_folder in os.listdir(read_img_loc_root + './' + subfolder_order):
            # use the Original image to create all the outputs
            if 'Original' in child_folder: # get the child_folder 
                read_img_loc  = read_img_loc_root + './' + subfolder_order + './' + child_folder
#                print (child_folder)            
 
     
    
                for sampleID_inclass, fileName_img in enumerate( os.listdir(read_img_loc) ):
                    
    #                fileName_img = 'ny1072-08-2.jpg'
                    
                    # generate cropped image 
                    canvas_size = [256,256]      
                    
                    
                    OriginalImg = io.imread(read_img_loc +'./' + fileName_img)                
                    cropped_image, border,vein, GANinput_border,GANinput_vein = image_preparing (OriginalImg , canvas_size,    visual=False   )
                    
                                    
                    # abbr of fileName
                    new_image_Name = fileName_img
                    new_image_Name = subfolder_order + '_' + str(sampleID_inclass+1)  + new_image_Name.replace(fileName_img.split('-')[0],'')
                    
#                    new_fullName = write_loc[imageType] + './' + orderName + '_' + str(sampleID_inclass+1) + new_image_Name  # Image ID, sample index, time (1...4)

                    
                    # write cropped image                 
                    X_Original.append (cropped_image) # dataset
                    X_border .append(border)
                    X_vein.append (vein) 
                    X_GANinput_border.append(GANinput_border)
                    X_GANinput_vein.append (GANinput_vein)             
                    y .append (classID+1)
                    
                    writeImgdone = cv2.imwrite( write_loc['Original'        ]  + './' + new_image_Name, cropped_image)  
                    writeImgdone = cv2.imwrite( write_loc['border'          ]  + './' + new_image_Name, border) 
                    writeImgdone = cv2.imwrite( write_loc['vein'            ]  + './' + new_image_Name, vein) 
                    writeImgdone = cv2.imwrite( write_loc['GANinput_border' ]  + './' + new_image_Name, GANinput_border) 
                    writeImgdone = cv2.imwrite( write_loc['GANinput_vein'   ]  + './' + new_image_Name, GANinput_vein) 
    
    
                    if writeImgdone == True:
                        print('Generate Cropped Images for '+ fileName_img +' done!')
                    else:
                        print('[Caucious! ]  Generate Cropped Images  failed!!!! for originalImgName')   
                    
