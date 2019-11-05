import streamlit as st
from PIL import Image
import pickle
import numpy as np
import os

st.title('Computer vision model')
st.text('Classification of such images: buildings, forest, glacier, mountain, sea, street')
directory = 'C:/users/admin/desktop/web-app/img/'
if st.checkbox('Show current directory'):
	st.info(directory)

lst =  [i for i in os.listdir(directory) if i.endswith('.jpg')]

selection = st.selectbox("Select image (size: 150x150; format: *.jpg)", lst)

link = directory + selection
img = Image.open(link)
st.image(img)

dictionary = {0:'buildings', 1:'forest', 2:'glacier', 3:'mountain', 4:'sea', 5:'street'}

def pred_func():
	arr = np.asarray(img, dtype='uint8').reshape(1,150,150,3)/255.0
	predict_model = pickle.load(open('CV_model.sav','rb'))
	predict = predict_model.predict(arr).argmax(axis = 1)[0]
	return dictionary[predict]

if st.button('Predict'):
	st.success("It's {}".format(pred_func()))
