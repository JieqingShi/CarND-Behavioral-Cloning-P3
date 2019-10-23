import csv
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

batch_size=128  # module level variable
    
def get_labels(path_label="data/driving_log.csv"):
    df = pd.read_csv(path_label)
    labels = df["steering"].values
    print("Got labels")
    return labels

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

def get_train_val_samples(driving_log_path="data/driving_log.csv"):
    samples = []
    with open(driving_log_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    samples = samples[1:]

    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples

def flip_image(image, angle):
    image_flipped = np.fliplr(image)
    angle_flipped = -angle
    return image_flipped, angle_flipped


def generator(samples, path_img, batch_size=32, flip_images=True, add_left_images=True, add_right_images=True):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        correction = 0.2
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = path_img+batch_sample[0].split('/')[-1]
                center_image = plt.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                if add_left_images:
                    left_img_name = path_img+batch_sample[1].split('/')[-1]
                    left_image = plt.imread(left_img_name)
                    left_angle = center_angle + correction
                    images.append(left_image)
                    angles.append(left_angle)
                
                if add_right_images:
                    right_img_name = path_img+batch_sample[2].split('/')[-1]
                    right_image = plt.imread(right_img_name)
                    right_angle = center_angle - correction
                    images.append(right_image)
                    angles.append(right_angle)
                
                # add image augmentation by flipping *each* image
                if flip_images:
                    if center_angle!=0: # no need to flip image if steering wheel angle is 0!
                        image_flipped, angle_flipped = flip_image(center_image, center_angle)
                        images.append(image_flipped)
                        angles.append(angle_flipped)
                        

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            

if __name__ == '__main__':
    path_label = "data/driving_log.csv"
    path_img = "data/IMG/"
    train_samples, validation_samples = get_train_val_samples()
    # compile and train the model using the generator function
    train_generator = generator(train_samples, path_img=path_img, batch_size=batch_size)
    validation_generator = generator(validation_samples, path_img=path_img, batch_size=batch_size)

    model = build_model()
    model.compile(loss=keras.losses.mse,
                  optimizer=keras.optimizers.Adam())
    model.fit_generator(train_generator, steps_per_epoch=np.ceil(len(train_samples)/batch_size), validation_data=validation_generator, 
                        validation_steps=np.ceil(len(validation_samples)/batch_size), epochs=5, verbose=1)

    model.save("model.h5")
    print("model saved")



