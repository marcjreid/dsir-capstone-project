import streamlit as st
from pokemontcgsdk import Card
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
import shutil 
import random
import tensorflow as tf
import cv2
import urllib.request
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout, BatchNormalization
from tensorflow.keras.applications import MobileNetV2, VGG16, InceptionV3, Xception
from tensorflow.keras.callbacks import EarlyStopping

path = './Data/PokemonData/Poke_train'
classes = os.listdir(path)
image_model = load_model('./poke_cnn_model.h5')
tcg_model = load_model('./poke_tcg_model.h5')
train_path = './Data/PokemonData/Poke_train'
test_path = './Data/PokemonData/Poke_test'
batch_size = 16
image_shape = (256,256,3)

image_gen = ImageDataGenerator(rotation_range=20,
                               rescale = 1./255,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.1,
                               zoom_range=0.1,
                               horizontal_flip=True,
                               fill_mode='nearest')


train_image_gen = image_gen.flow_from_directory(train_path,
                                                target_size=image_shape[:2],
                                                color_mode='rgb',
                                                batch_size=batch_size,
                                                class_mode='categorical')

def model_predict(img_path):
    img = Image.open(img_path).convert('RGB')
    pred_class = np.argmax(image_model.predict((np.array(img.resize((256,256)))/255).reshape(-1, 256, 256, 3)), axis=-1)
    poke_class = {value:key for key,value in train_image_gen.class_indices.items()}
    st.image(img)
    st.write(poke_class[pred_class[0]])

def model_predict_url(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    pred_class = np.argmax(image_model.predict((np.array(img.resize((256,256)))/255).reshape(-1, 256, 256, 3)), axis=-1)
    poke_class = {value:key for key,value in train_image_gen.class_indices.items()}
    st.image(img)
    st.write(poke_class[pred_class[0]])

def tcg_model_predict_url(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    pred_class = np.argmax(tcg_model.predict((np.array(img.resize((256,256)))/255).reshape(-1, 256, 256, 3)), axis=-1)
    poke_class = {value:key for key,value in train_image_gen.class_indices.items()}


    st.image(border_image(image_url))
    st.write(poke_class[pred_class[0]])

def border_image(image_url):
    response = requests.get(image_url)
    bytes_im = BytesIO(response.content)
    frame = cv2.cvtColor(np.array(Image.open(bytes_im)), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (5,5))
    gray = cv2.bilateralFilter(gray, 11, 17, 17) #blur. very CPU intensive.
    cv2.imshow("Gray map", gray)

    edges = cv2.Canny(gray, 30, 120)


    #find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    # use RETR_EXTERNAL since we know the largest (external) contour will be the card edge.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    dilated = cv2.dilate(edges, kernel)
    cnts, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:1]
    screenCnt = None

# loop over our contours
    for c in cnts:
    # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.3 * peri, True)

        cv2.drawContours(frame, [cnts[0]], -1, (0, 255, 0), 2)

        # if our approximated contour has four points, then
        # we can assume that we have found our card
        if len(approx) == 4:
            screenCnt = approx;
        break
    return(frame)
    


st.title("This app is useful for identifying Pokémon cards.")
st.header("Pokémon image classifier")

st. subheader("Saved Image")

pkmn = st.selectbox("Pokemon", tuple(classes))
p_image = st.selectbox("Image", tuple(os.listdir(path + "/" + pkmn)))

model_predict(f"{path}/{pkmn}/{p_image}")


st.subheader("New Image")
img_url = st.text_input("Enter URL of Pokemon image", "")

model_predict_url(img_url)

st.header("Pokémon card classifier")
img_url = st.text_input("Enter URL of Pokemon card image", "")
tcg_model_predict_url(img_url)