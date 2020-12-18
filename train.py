no_of_epochs = 20

#Importing libraries
import glob
import argparse
from glob import glob
from os import getcwd, chdir
from random import shuffle
import os,shutil
from shutil import move
from shutil import copyfile

print("Preparing data...")
#Creating folder tree
get = os.getcwd()
dir = get + "/img"
if os.path.exists(dir):
    shutil.rmtree(dir)

os.mkdir("img")

os.chdir(get+"/img")
os.mkdir("train")
os.mkdir("test")
os.mkdir("valid")

os.chdir(get+"/img/train")
os.mkdir("positive")
os.mkdir("negative")

os.chdir(get+"/img/test")
os.mkdir("positive")
os.mkdir("negative")

os.chdir(get+"/img/valid")
os.mkdir("positive")
os.mkdir("negative")

os.chdir(get)

#Creating runtime parser for taking values in argument
parser = argparse.ArgumentParser()

parser.add_argument('--pos', dest = 'pos', default = None, type = str)
parser.add_argument('--neg', dest = 'neg', default = None, type = str)


#Storing the arguments
args = parser.parse_args()
Positive = args.pos
Negative = args.neg


#creating list That will contain path for all the images
pos_img_list = []
neg_img_list = []


#Creating List of Positive and Negative images from the provided path 
extension = "jpg"

with open (Positive,"r") as txt:
	for single_path in txt.readlines():
		single_path = single_path[:-1]
		
		saved = getcwd()
		chdir(single_path)
		local_pos = glob('*.' + extension)
		complete_pos = [single_path+ "/" + e for e in local_pos]

		for imag in complete_pos:
			pos_img_list.append(imag)

		chdir(saved)

with open (Negative,"r") as txt2:
	for single_npath in txt2.readlines():
		single_npath = single_npath[:-1]
		saved = getcwd()

		chdir(single_npath)
		local_npos = glob('*.' + extension)
		complete_npos = [single_npath+ "/" + e for e in local_npos]

		for imag2 in complete_npos:
			neg_img_list.append(imag2)

		chdir(saved)

shuffle(pos_img_list)
shuffle(neg_img_list)

##Seprating the test and validation Images (Positive and negative)
#Providing size for test and valid split
pos_test_size = len(pos_img_list)//10
pos_valid_size = len(pos_img_list)//10

neg_test_size = len(neg_img_list)//10
neg_valid_size = len(neg_img_list)//10

#Creating Test, Train and Valid image list
pos_test_data = pos_img_list[:pos_test_size]
pos_valid_data = pos_img_list[-pos_valid_size:]
pos_train_data = pos_img_list[pos_test_size:-pos_valid_size]

neg_test_data = neg_img_list[:neg_test_size]
neg_valid_data = neg_img_list[-neg_valid_size:]
neg_train_data = neg_img_list[neg_test_size:-neg_valid_size]

train_size = len(pos_train_data) + len(neg_train_data)
valid_size = len(pos_valid_data) + len(neg_valid_data)

#Creating corresponding text files
# with open('positive_training.txt', 'w') as f:
n=0
for item in pos_train_data:
    n = n+1
    dest = get+"/img/train/positive/"+str(n)+".jpg"
    copyfile(item,dest)                

n=0
for item in pos_test_data:
    n = n+1
    dest = get+"/img/test/positive/"+str(n)+".jpg"
    copyfile(item,dest)

n=0
for item in pos_valid_data:
    n = n+1
    dest = get+"/img/valid/positive/"+str(n)+".jpg"
    copyfile(item,dest)

n=0
for item in neg_train_data:
    n = n+1
    dest = get+"/img/train/negative/"+str(n)+".jpg"
    copyfile(item,dest)

n=0
for item in neg_test_data:
    n = n+1
    dest = get+"/img/test/negative/"+str(n)+".jpg"
    copyfile(item,dest)

n=0
for item in neg_valid_data:
    n = n+1
    dest = get+"/img/valid/negative/"+str(n)+".jpg"
    copyfile(item,dest)
# _______________________________________________________________________________________________________

#importing libraries
import numpy as np
import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import os,shutil

#importing the model  
print("Creating Model..")
mobile = keras.applications.mobilenet.MobileNet()

#specifying path for train and valid locations
train_path = "img/train"
val_path = "img/valid"

#creating batches to feed to the model
train_batches = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(train_path,target_size=(224,224),batch_size=64)
valid_batch = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input).flow_from_directory(val_path,target_size=(224,224),batch_size=64)
   
#creating architecture of the new model
x=mobile.layers[-6].output
prediction = Dense(2,activation='softmax')(x)
model = Model(inputs=mobile.input, outputs=prediction)

print("Training model...")
#Defining the new model with training
train_epochs = train_size/64
valid_epochs = valid_size/64
model.compile(Adam(lr=.0001), loss = 'categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(train_batches,steps_per_epoch = train_epochs ,validation_data=valid_batch,validation_steps=valid_epochs,epochs=no_of_epochs,verbose=2)

print("Saving model..")
#saving the mmodel
model.save('Model.h5')

print("Model Saved!!")
get = os.getcwd()
dir = get + "/img/train"
if os.path.exists(dir):
    shutil.rmtree(dir)

dir = get + "/img/valid"
if os.path.exists(dir):
    shutil.rmtree(dir)

print("Done")
