import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.express as px
from PIL import Image
import os
import cv2 
#from google.colab.patches import cv2_imshow
import dlib
from skimage import io
import matplotlib.pyplot as plt
from pathlib import Path
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

import numpy as np
from google.colab import files
from keras.preprocessing import image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow import keras

def welcome():
    st.title('오늘 몇 칼로리?')
    st.subheader('오늘의 식사를 이미지 파일로 업로드 해 주세요.')
    
    st.image('급식93.jpg',use_column_width=True)


def photo():
    st.title('식사를 보여주세요')
    uploaded_file = st.file_uploader("이미지파일선택",type = ["jpg","png","jpeg"])
    
    try:
      if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='선택된 이미지.', use_column_width=True)
        st.write("")
        st.write("어떤 종류의 밥일까요?")

        #이미지 크기가 위와 동일한 64로 가지고 와야함
            # img=image.load_img(path, target_size=(64, 64))

            # plt.imshow(image)
            # plt.show()


            # x=image.img_to_array(target_size=(64, 64))
            # x=np.expand_dims(x, axis=0)
            # images = np.vstack([x])

        model2= keras.models.load_model("keras_model.h5")
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        image = image

            #resize the image to a 224x224 with the same strategy as in TM2:
            #resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

            #turn the image into a numpy array
        image_array = np.asarray(image)
            # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            # Load the image into the array
        data[0] = normalized_image_array

            # run the inference
        prediction = model2.predict(data)
        st.write(prediction)

        label_ = 0
        result1 = "어떤 종류의 밥일까요?"

        label_ = np.argmax(prediction[0])


        if label_ == 0:
              result1 = "백미입니다. (130ckal / 100g)"
        if label_ == 1:
              result1 =  "현미입니다. (110.9ckal / 100g)"
        if label_ == 2:
              result1 = "흑미입니다. (330ckal / 100g)"
            
        st.write("오늘 내가 먹은 밥은?: "+ result1)
      
    except:
        st.error('밥 사진을 다시 올려주세요.')


selected_box = st.sidebar.selectbox('아래 탭을 눌러 사진을 업로드 해 주세요',('안녕하세요','사진업로드'))
    
if selected_box == '안녕하세요':
    welcome()
    st.sidebar.write("칼로리, 대신 계산해드립니다.")
if selected_box == '사진업로드':
    photo()
    st.sidebar.write("뭐 드셨는지만 보여주세요! ")