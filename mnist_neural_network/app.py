import streamlit as st
import numpy as np
import json
import requests
import matplotlib.pyplot as plt

URI = 'http://127.0.0.1:5000/' #currently my model server

st.set_option('deprecation.showPyplotGlobalUse', False)
#st.get_option('theme.backgroundColor')

st.title('MNIST – ML prediction')
st.subheader('Visualising hidden layers')
st.sidebar.markdown("<h2 style='text-align: center;'>Image input</h2>",
                    unsafe_allow_html=True)  # make sidebar fixed later

if st.button('Get random image'):
    response = requests.post(URI, data={})
    #st.markdown(response.data)
    response = json.loads(response.text)
    preds = response.get('prediction')
    image = response.get('image')
    image = np.reshape(image, (28, 28))

    col1, col2, col3 = st.sidebar.columns([2,5,1])

    with col1:
        st.write("")

    with col2:
        st.image(image, width=150)

    with col3:
        st.write("")
    #st.sidebar.image(image, width=200)

    for layer, pred in enumerate(preds):
        numbers = np.squeeze(np.array(pred)) # removes additional dimensionality of the data

        plt.figure(figsize=(32, 4))
        if layer == 2: # my final layer
            row = 1
            col = 10
        else:
            row = 2
            col = 16

        for i, number in enumerate(numbers):
            plt.subplot(row, col, i + 1)
            plt.imshow(number * np.ones((8, 8, 3)).astype('float32'))
            plt.xticks([])
            plt.yticks([])

            if layer == 2:
                plt.xlabel(str(i), fontsize=40)

        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.tight_layout()

        st.text(f'Layer {layer + 1} (colour range: black == 0, white == 1')
        st.pyplot()
