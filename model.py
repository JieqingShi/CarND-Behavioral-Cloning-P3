#!/usr/bin/env python
# coding: utf-8

# ### Notebook for training used on the provided dataset
# 
# 
# Using the NVIDIA architecture
# This is mainly used for looking at the potential preprocessing steps, model architecture etc.

# In[57]:


import os
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from scipy import ndimage
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Cropping2D
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
from sklearn.utils import shuffle
import sklearn
import cv2


# In[2]:


path_label = "data/driving_log.csv"
path_img = "data/IMG"


# In[3]:


def get_labels(path_label="data/driving_log.csv"):
    df = pd.read_csv(path_label)
    labels = df["steering"].values
    print("Got labels")
    return labels


# In[4]:


def get_images(path_img="data/IMG"):
    images = []
    fnames = os.listdir(path_img)
    for fname in fnames:
        # only use CENTER images first
        if fname.startswith("center"):
            img = plt.imread(os.path.join(path_img, fname))
            images.append(img)
    images = np.array(images)
    print("Got images")
    return images


# In[87]:


def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))  # normalize
    # add cropping
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Conv2D(24,(5,5), strides=(2,2), activation="relu"))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation="relu"))
    model.add(Conv2D(48, (5,5), strides=(2,2), activation="relu"))
    model.add(Conv2D(64, (3,3), strides=(1,1), activation="relu"))
    model.add(Conv2D(64, (3,3), strides=(1,1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


# In[101]:


samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples = samples[1:]
        
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = plt.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# In[102]:


# Set our batch size
batch_size=128

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


# In[53]:


#y_train = get_labels()
#X_train = get_images()


# In[103]:


model = build_model()


# In[104]:


model.compile(loss=keras.losses.mse,
              optimizer=keras.optimizers.Adam())


# In[105]:


# model.fit(X_train, y_train, validation_split=0.2, batch_size=128, epochs=5, shuffle=True)
model.fit_generator(train_generator, steps_per_epoch=np.ceil(len(train_samples)/batch_size),             validation_data=validation_generator, validation_steps=np.ceil(len(validation_samples)/batch_size),             epochs=5, verbose=1)


# In[106]:


# model.save("model.h5")
model.save("model.h5")
print("model saved")


# In[12]:


K.clear_session()


# In[ ]:


# ToDos for first step

# write generator for iterative loading of images
# add regularization stuff (batch_norm, dropout)
# train and save model
# deploy model and test


# #### Further Experimentations:
# 
# 
# - different processing strategies in lambda layer
# - Image augmentation (flipping)
# - cropping images (top and bottom)
# - use left and right images for recovery
# - modify model architecture
# - iterative, "transfer" learning approach (see https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/Behavioral+Cloning+Cheatsheet+-+CarND.pdf)
# 

# In[ ]:




