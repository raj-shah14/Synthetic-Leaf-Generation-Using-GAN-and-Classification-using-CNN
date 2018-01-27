from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from scipy.cluster.hierarchy import dendrogram, linkage
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import sys, pickle
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import fcluster
import os, glob
from shutil import copyfile
import scipy.io as sio
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pylab
import itertools
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
import random
import math
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.misc import imresize

# Building 'AlexNet'
#Image Augmentation
img_aug=ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_blur(sigma_max=3.)
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_flip_updown()

network = input_data(shape=[None, 227, 227, 3], data_augmentation=img_aug)
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 5, activation='softmax')
network = regression(network, optimizer='momentum',loss='categorical_crossentropy',learning_rate=0.001)

#read all folders
all_folders=os.listdir('Image_prepared')

#store, targets and resized images
    

targets=[]
images=[]

target_id=0

for folder_name in all_folders:
    all_images=os.listdir('Image_prepared/'+folder_name)
    for img in all_images:
        infile='Image_prepared/'+folder_name+'/'+img
        im=Image.open(infile)
        im=imresize(im,(227,227,3))
        pix=np.array(im)
        images.append(pix)
        targets.append(str(target_id))
    target_id=target_id+1

#saving the images and targets

np.save('targets.npy',targets)
np.save('images.npy',images)

#splitting train, test and validation indices
N=len(targets)
Ntrain=math.floor(0.64*N)
Nvalidate=math.floor(0.16*N)
Ntest=math.floor(0.2*N)

indices=[i for i in range(0,N)]

train_indices=random.sample(indices,(int)(Ntrain))
indices=[x for x in indices if x not in train_indices]

validate_indices=random.sample(indices,(int)(Nvalidate))
indices=[x for x in indices if x not in validate_indices]

test_indices=random.sample(indices,(int)(Ntest))

train_t=[targets[x] for x in train_indices]
validate_t=[targets[x] for x in validate_indices]
test_t=[targets[x] for x in test_indices]

train_X=[images[x] for x in train_indices]
validate_X=[images[x] for x in validate_indices]
test_X=[images[x] for x in test_indices]

#t1=to_categorical(train_t,5)
#f1=np.array(train_X)

#t2=to_categorical(validate_t,5)
#f2=np.array(validate_X)

t1=to_categorical(targets,5)
f1=np.array(images)

model = tflearn.DNN(network)
model.fit(f1,t1, n_epoch=150, validation_set=0.2, shuffle=True,show_metric=True, batch_size=300, snapshot_step=200, snapshot_epoch=False, run_id='training')

#model.fit(f1,t1, n_epoch=150, validation_set=(f2,t2), shuffle=True,show_metric=True, batch_size=300, snapshot_step=200, snapshot_epoch=False, run_id='training')
#go to each of the indices in these clusters

model.save('model_alex_full.tflearn')
model.load('model_alex_full.tflearn')

#model.save('model_full_1200.tflearn')
#v=np.array(validate_X)
#model performance on validation set
#validate_pred1=[]
#for i in range(len(v)):
#    validate_pred1.append(model.predict(v[i].reshape(1,227,227,3)))

#validate_pred=[]
#for pred in validate_pred1:
#    y1=pred[0]
#    max_val=0
#    idx=0
#    for i in range(len(y1)):
#        if(y1[i]>max_val):
#            max_val=y1[i]
#            idx=i
#    validate_pred.append(str(idx))

#print('validation accuracy:')
#print(accuracy_score(validate_t,validate_pred))


#v=np.array(test_X)
#model performance on validation set
#validate_pred1=[]
#for i in range(len(v)):
#    validate_pred1.append(model.predict(v[i].reshape(1,227,227,3)))

#validate_pred=[]
#for pred in validate_pred1:
#    y1=pred[0]
#    max_val=0
#    idx=0
#    for i in range(len(y1)):
#        if(y1[i]>max_val):
#            max_val=y1[i]
#            idx=i
#    validate_pred.append(str(idx))

#print('Test accuracy:')
#print(accuracy_score(test_t,validate_pred))
