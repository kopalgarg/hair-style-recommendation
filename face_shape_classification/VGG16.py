import numpy as np
import pandas as pd
import sys
import cv2
import dlib
import numpy as np
import numpy as np
from PIL import Image
import shutil
from google.colab.patches import cv2_imshow
from google.colab import files
import mediapipe as mp
from imutils import face_utils
import eos
from imageio import imread
from io import BytesIO
import IPython.display
import requests
from bs4 import BeautifulSoup
import time
from PIL import Image, ImageDraw
import face_recognition
from os.path import basename
import math
import pathlib
from pathlib import Path
import os
import random
import matplotlib.pyplot as plt
from skimage.draw import circle
import glob
import h5py
from PIL import ImageFile
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from tensorflow.keras.utils import to_categorical
from keras import backend as K
import faceBlendCommon as fbc
from tensorflow.keras import layers
import tensorflow as tf

vgg16 = '/content/rcmalli_vggface_tf_notop_vgg16.h5'

# Run preprocessing on all images
data_training = []

for r, d, f in os.walk(train_images):
    for file in f:
        if ".jpg" in file:
            data_training.append((os.path.join(r, file).split("/")[-2], os.path.join(r, file)))

df_training = pd.DataFrame(data_training, columns=['class','file_path'])


data_testing = []

for r, d, f in os.walk(test_images):
    for file in f:
        if ".jpg" in file:
            data_testing.append((os.path.join(r, file).split("/")[-2], os.path.join(r, file)))

df_testing = pd.DataFrame(data_testing, columns=['class','file_path'])

for i in range(len(df_training)):
    transform(df_training['file_path'][i])

for i in range(len(df_testing)):
    transform(df_testing['file_path'][i])

data_training = []

for r, d, f in os.walk(train_images):
    for file in f:
        if "_NEW_cropped.jpg" in file:
            data_training.append((os.path.join(r, file).split("/")[-2], os.path.join(r, file)))

df_training = pd.DataFrame(data_training, columns=['class','file_path'])


data_testing = []

for r, d, f in os.walk(test_images):
    for file in f:
        if "_NEW_cropped.jpg" in file:
            data_testing.append((os.path.join(r, file).split("/")[-2], os.path.join(r, file)))

df_testing = pd.DataFrame(data_testing, columns=['class','file_path'])

TrainDatagen = ImageDataGenerator(
        preprocessing_function= preprocess_input,
        horizontal_flip = True)

TestDatagen = ImageDataGenerator(
    preprocessing_function= preprocess_input)

train_data = TrainDatagen.flow_from_dataframe(
    df_training, x_col='file_path',
    target_size = (224,224),
    batch_size =batch_size,
    class_mode = 'categorical')

test_data = TestDatagen.flow_from_dataframe(
    df_testing, x_col='file_path',
    target_size = (224,224),
    batch_size = batch_size,
    class_mode = 'categorical'
)

# Define model architecture

# Loading VGG16 as base model
base_model = VGG16(input_shape=(224, 224, 3),  # same as our input
                   include_top=False,  # exclude the last layer
                   weights=vgg16)  # use VGGFace Weights
for layer in base_model.layers:
  layer.trainable = False

model_t1 = Sequential()

# Build and compile model
x = layers.Flatten()(base_model.output)

x = layers.Dense(64, activation='relu')(x)  # add 1 fully connected layer, try with 512 first 
x = layers.Dropout(0.5)(x)
x = layers.Dense(5, activation='softmax')(x)  # add final layer

model_t1 = tf.keras.models.Model(base_model.input, x)

model_t1.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])

model_t1.summary()

# Train the model 

ImageFile.LOAD_TRUNCATED_IMAGES = True
history = model_t1.fit(
    train_data,              
    steps_per_epoch = train_data.samples//batch_size,
    validation_data = test_data,
    validation_steps = test_data.samples//batch_size,
    epochs = 50,
    callbacks=[es,chkpt])

model_t1.save('face_shape_classifier.h5')
model_t1.save_weights('face_shape_classifier_weights.h5')

model_t1.predict(np.asarray(crop_image).reshape(-1, np.asarray(crop_image).shape[0], np.asarray(crop_image).shape[1], np.asarray(crop_image).shape[2]))

