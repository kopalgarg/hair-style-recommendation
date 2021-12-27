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

image_select = '/content/output/testNoBackground.jpg'
image = face_recognition.load_image_file(image_select)
face_landmarks_list = face_recognition.face_landmarks(image)
facial_features = ['chin','left_eyebrow','right_eyebrow','nose_bridge','nose_tip','left_eye','right_eye','top_lip','bottom_lip']
pts = []
for face_landmarks in face_landmarks_list:
  for facial_feature in facial_features:
      for point in  face_landmarks[facial_feature]:
        for pix in point:
          pts.append(pix)  
  eyes = []
  lex = pts[72]
  ley = pts[73]
  rex = pts[90]
  rey = pts[91]
  eyes.append(pts[72:74])
  eyes.append(pts[90:92])
  crop_image = crop_face(image, eye_left=(lex, ley), eye_right=(rex, rey), offset_pct=(0.34,0.34), dest_sz=(224,224))

crop_image.save(str(image_select)+"_NEW_cropped.jpg")
nn = str(image_select)+"_NEW_cropped.jpg"
pts = []
face = 0
image = face_recognition.load_image_file(nn)
face_landmarks_list = face_recognition.face_landmarks(image)
facial_features = ['chin','left_eyebrow','right_eyebrow','nose_bridge','nose_tip','left_eye','right_eye','top_lip','bottom_lip']

pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)

for face_landmarks in face_landmarks_list:
    # trace out each facial feature in the image with points
    for facial_feature in face_landmarks.keys():
      d.line(face_landmarks[facial_feature], width=5)
      d.point(face_landmarks[facial_feature], fill = (255,255,255))
    for facial_feature in facial_features:
      for  point in  face_landmarks[facial_feature]:
        for pix in point:
          pts.append(pix)

face_width = np.sqrt(np.square(pts[0] - pts[32]) + np.square( pts[1]  -  pts[33]))
face_height = np.sqrt(np.square(pts[16] - pts[56]) + np.square(pts[17] -  pts[57] )) * 2
height_to_width = face_height/face_width
jaw_width = np.sqrt(np.square(pts[12]-pts[20]) + np.square(pts[13]-pts[21]))
jaw_width_to_face_width = jaw_width/face_width
eye_brow_arch = (abs(pts[43]-pts[39])+abs(pts[53]-pts[49]))/2
lip_width = abs(pts[139]-pts[127])
eye_height = (abs(pts[75]-pts[81])+abs(pts[87]-pts[95]))/2
eye_width = (abs(pts[72]-pts[78])+abs(pts[84]-pts[90]))/2
eye_height_to_eye_width = eye_height/eye_width
nose_length = abs(pts[57]-pts[67])
nose_width = abs(pts[70]-pts[62])
