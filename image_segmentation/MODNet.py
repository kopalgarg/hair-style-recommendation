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

# MODNet
import os

# clone the repository
%cd /content
if not os.path.exists('MODNet'):
  !git clone https://github.com/ZHKKKe/MODNet
%cd MODNet/

# dowload the pre-trained ckpt for image matting
pretrained_ckpt = 'pretrained/modnet_photographic_portrait_matting.ckpt'
if not os.path.exists(pretrained_ckpt):
  !gdown --id 1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz \
          -O pretrained/modnet_photographic_portrait_matting.ckpt

input_folder = '/content/input'
if os.path.exists(input_folder):
  shutil.rmtree(input_folder)
os.makedirs(input_folder)

output_folder = '/content/output'
if os.path.exists(output_folder):
  shutil.rmtree(output_folder)
os.makedirs(output_folder)

shutil.move('/content/test.jpg', os.path.join(input_folder, 'test.jpg'))

## Remove Background
!python -m demo.image_matting.colab.inference \
        --input-path /content/input/ \
        --output-path /content/output/ \
        --ckpt-path ./pretrained/modnet_photographic_portrait_matting.ckpt

# https://livecodestream.dev/post/remove-the-background-from-images-using-ai-and-python/
image_name = 'test.jpg'
matte_name = image_name.split('.')[0] + '.png'
image = Image.open(os.path.join(input_folder, image_name))
matte = Image.open(os.path.join(output_folder, matte_name))

# calculate display resolution
w, h = image.width, image.height
rw, rh = 800, int(h * 800 / (3 * w))
  
# obtain predicted foreground
image = np.asarray(image)
if len(image.shape) == 2:
  image = image[:, :, None]
if image.shape[2] == 1:
  image = np.repeat(image, 3, axis=2)
elif image.shape[2] == 4:
  image = image[:, :, 0:3]
matte = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2) / 255
foreground = image * matte + np.full(image.shape, 255) * (1 - matte)
  
# combine image, foreground, and alpha into one line
combined = np.concatenate((image, foreground, matte * 255), axis=1)
combined = Image.fromarray(np.uint8(combined)).resize((rw, rh))

foreground = Image.fromarray(np.uint8(foreground)).resize((w, h))

foreground.save("/content/output/testNoBackground.jpg")
