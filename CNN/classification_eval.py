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

model = tflearn.DNN(network)
model.load('model_alex_full.tflearn')

output_path='/GAN/GANoutput_vein'
input_path='/GAN/GANinput_vein_test'

input_files=os.listdir(input_path)
output_files=os.listdir(output_path)

files_out=[]

pred_label=[]
true_label=[]


#4923 files
cnt=0
for filename in output_files:
     print(cnt)
     cnt=cnt+1
     if filename.endswith('.png'):
         files_out.append(filename)
         #read the file
         infile=output_path+'/'+filename
         im=Image.open(infile)
         im=imresize(im,(227,227,3))
         probs=model.predict(im.reshape(1,227,227,3))
         pred_label.append(str(probs[0].index(max(probs[0]))))
         idx_num=filename[-8:-4]
         for x in input_files:
            if x[:4]==idx_num:
                if x[5:8]=='Elm':
                    true_label.append('0')
                if x[5:8]=='Mag':
                    true_label.append('1')
                if x[5:8]=='Map':
                    true_label.append('2')
                if x[5:8]=='Oak':
                    true_label.append('3')
                if x[5:8]=='Pin':
                    true_label.append('4')
            filename1='misclassification_86_vein/'+filename[:-4]+'_'+true_label[-1]+'_'+pred_label[-1]+'.png'
            copyfile(infile, filename1)


print('Accuracy score:')
print(accuracy_score(true_label,pred_label)) #53.99%
print('Confusion matrix:')
print(confusion_matrix(true_label,pred_label))
# [[  29   11   70  586  236]
#  [  27  206   62  515   70]
#  [   0    0  362  529  104]
#  [   0    0    5  985   50]
#  [   0    0    0    0 1076]]


#Vein epoch 86:
# Accuracy:0.825512898639
# Confusion matrix:
# [[424 336 146  19   7]
#  [ 55 772  24  27   2]
#  [  1   0 949  41   4]
#  [  1   3  68 967   1]
#  [  1   1  83  39 952]]

#Border epoch 86:
#Accuracy: 0.871419865935
# Confusion matrix:
# [[ 637  270   14   10    1]
#  [  70  799    2    7    2]
#  [   2    1  747  239    6]
#  [   0    1    4 1034    1]
#  [   0    0    2    1 1073]]

