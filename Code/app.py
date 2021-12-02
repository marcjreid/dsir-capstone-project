import streamlit as st
from pokemontcgsdk import Card
from simple_image_download import simple_image_download as simp
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
import urllib.request
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout, BatchNormalization
from tensorflow.keras.applications import MobileNetV2, VGG16, InceptionV3, Xception
from tensorflow.keras.callbacks import EarlyStopping

path = '../Data/PokemonData/Poke_train'
classes = os.listdir(path)

st.title("This app is useful for identifying Pokémon cards.")
st.header("Pokémon image classifier")

st. subheader("Saved Images")
option = st.selectbox("Select Pokemon Image", tuple(classes))

st.subheader("New Images")
img_url = st.text_input("Enter URL of Pokemon image", "")

