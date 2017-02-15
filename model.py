import tensorflow as tf
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.misc
from scipy.ndimage import rotate

DATA_PATH = './data/'

#col_names = ['center', 'left','right','steering','throttle','brake','speed']
data = pd.read_csv(DATA_PATH + 'driving_log.csv')
num_of_img = len(data)

STEERING_CORRECTION = 0.09 # used to estimate the correction of the steering angle for left and right camera images

# image pipeline (to be applied in drive.py too)
def process_img(img):
	height = img.shape[0]
	top = int(np.ceil(height * 0.30))
	bottom = height - int(np.ceil(height * 0.2))
	img = img[top:bottom, :]
	return scipy.misc.imresize(img, (64,64))
	
# Source: https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk
def random_flip(image,steering):
    if np.random.randint(0,2)==0:
        image,steering=cv2.flip(image,1),-steering
    return image,steering
        
# Source: https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk
def random_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = 0.8 + 0.4*(2*np.random.uniform()-1.0)    
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1
	
# Source: https://medium.com/@ksakmann/behavioral-cloning-make-a-car-drive-like-yourself-dc6021152713#.7k8vfppvk
def random_shear(image,steering,shear_range):
    rows,cols,ch = image.shape
    dx = np.random.randint(-shear_range,shear_range+1)
    random_point = [cols/2+dx,rows/2]
    pts1 = np.float32([[0,rows],[cols,rows],[cols/2,rows/2]])
    pts2 = np.float32([[0,rows],[cols,rows],random_point])
    dsteering = dx/(rows/2) * 360/(2*np.pi*25.0) / 6.0    
    M = cv2.getAffineTransform(pts1,pts2)
    image = cv2.warpAffine(image,M,(cols,rows),borderMode=1)
    steering +=dsteering
    return image,steering
    
def new_img(image, steering_angle):
	image, steering_angle = random_shear(image, steering_angle, 115)
	image = process_img(image)
	image, steering_angle = random_flip(image, steering_angle)
	image = random_brightness(image)
	return image, steering_angle
    
def get_next_images(batch_size=64):
    pos = np.random.randint(0, num_of_img, batch_size)

    image_files_and_angles = []
    for index in pos:
        rnd_image = np.random.randint(0, 3)
        if rnd_image == 0:
            img = data.iloc[index]['left'].strip()
            angle = data.iloc[index]['steering'] + STEERING_CORRECTION
            image_files_and_angles.append((img, angle))

        elif rnd_image == 1:
            img = data.iloc[index]['center'].strip()
            angle = data.iloc[index]['steering']
            image_files_and_angles.append((img, angle))
        else:
            img = data.iloc[index]['right'].strip()
            angle = data.iloc[index]['steering'] - STEERING_CORRECTION
            image_files_and_angles.append((img, angle))

    return image_files_and_angles
    
# image generator
def generator(batch_size=64):
    while True:
        x = []
        y = []
        images = get_next_images(batch_size)
        for img_file, steering_angle in images:
            image = plt.imread(DATA_PATH + img_file)
            angle = steering_angle
            new_image, new_angle = new_img(image, angle)
            x.append(new_image)
            y.append(new_angle)

        yield np.array(x), np.array(y)
        

# hyperparameters
nb_epochs = 10
samples_per_epoch = 25600
nb_val_samples = 5120
learning_rate = 0.001 
batch_size = 64

# model creation (similar to the Nvidia one)
model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2,2)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2,2)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
#model.add(Dropout(0.25))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))


# compile the model
model.compile(optimizer=Adam(lr=learning_rate), loss="mse")

# leave training earlier if loss on validation does not improve for two consecutive epochs

if os.path.isfile('best.h5'):
	os.remove('best.h5')

callbacks = [EarlyStopping(monitor='val_loss',patience=1,verbose=0) , ModelCheckpoint('best.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)]

# train the model
history = model.fit_generator(generator(batch_size),
                              samples_per_epoch=samples_per_epoch,
                              nb_epoch=nb_epochs,
                              validation_data=generator(batch_size),
                              nb_val_samples=nb_val_samples,
			      callbacks=callbacks)

# save model.json and model.h5 (weights)
if os.path.isfile('model.json'):
	os.remove('model.json')
if os.path.isfile('model.h5'):
	os.remove('model.h5')
with open('model.json', 'w') as outfile:
	json.dump(model.to_json(), outfile)
model.save_weights('model.h5')
