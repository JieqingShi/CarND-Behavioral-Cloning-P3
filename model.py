#!/usr/bin/env python
# coding: utf-8

# ### Notebook for training used on the provided dataset
# 
# 
# Using the NVIDIA architecture
# This is mainly used for looking at the potential preprocessing steps, model architecture etc.


import os
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from scipy import ndimage
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Cropping2D
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf

path_label = "data/driving_log.csv"
path_img = "data/IMG"
df = pd.read_csv(path_label)
labels = df["steering"].values


images = []
fnames = os.listdir(path_img)
# img = cv2.imread(os.path.join(path_img, fnames[0]), cv2.COLOR_BGR2RGB)
for fname in fnames:
    # only use CENTER images first
    if fname.startswith("center"):
        img = plt.imread(os.path.join(path_img, fname))
        images.append(img)
images = np.array(images)


X_train = images
y_train = labels
assert len(X_train) == len(y_train)

# define model using NVIDIA architecture
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

model.compile(loss=keras.losses.mse,
              optimizer=keras.optimizers.Adam())

model.fit(X_train, y_train, validation_split=0.2, batch_size=128, epochs=5, shuffle=True)

model.save_weights("model.h5")
K.clear_session()




