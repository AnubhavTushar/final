import cv2
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt

content_image,style_image=None,None
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)
st.title("Deep Learning (Neural Style Transfer) Web App")
uploaded_file = st.sidebar.file_uploader("Upload Content Image:",type=['png','jpeg','jpg'])
if uploaded_file is not None:
    st.image(uploaded_file, width=300, caption='Content Image')
    content_image = plt.imread("D:\\Tushar Project\\" + uploaded_file.name)

uploaded_file2 = st.sidebar.file_uploader("Upload Style Image:",type=['png','jpeg','jpg'])
if uploaded_file2 is not None:
    st.image(uploaded_file2, width=300, caption='Style Image')
    style_image = plt.imread("D:\\Tushar Project\\" + uploaded_file2.name)

if st.sidebar.button('Create Candidate Image'):
   content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
   style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
   style_image = tf.image.resize(style_image, (256, 256))
   outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
   stylized_image = outputs[0]
   stylized_image_reshape = tf.reshape(stylized_image,shape=(stylized_image.shape[1:]))
   st.image(stylized_image_reshape.numpy(),width=300,caption='Candidate Image')
