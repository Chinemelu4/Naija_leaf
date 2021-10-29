import numpy as np
import streamlit as st
from PIL import Image 
import requests
from skimage.transform import resize
from io import BytesIO
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input



st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Nigerian leaves classifier')
st.text('Upload your leaf image here')


model=load_model("best_model.h5",path=(https://github.com/Chinemelu4/Naija_leaf/blob/main/best_model.h5))

classes=['afang', 'bitterleaf', 'oha', 'pumpkin', 'waterleaf']

uploaded_file=st.file_uploader("choose an image....", type=["jpg","png"])
if uploaded_file is not None:
  img= Image.open(uploaded_file)
  st.image(img,caption='Uploaded Image') 

  if st.button('PREDICT'):
    classes=['Afang leaf', 'Bitterleaf', 'Oha leaf', 'Pumpkin leaf', 'Waterleaf']
    img=img.resize((256,256))

    i= img_to_array(img)

    i=preprocess_input(i)

    input_arr=np.array([i])
    
     
    y_out=np.argmax(model.predict(input_arr))
    y_out=classes[y_out]
    st.write(f' With 85% certainty, I can confirm that this is ',y_out)
    st.write("We would love to taste whatever you are cooking")
 
