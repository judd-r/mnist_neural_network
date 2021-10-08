import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import requests
import matplotlib.pyplot as plt

URI = 'http://127.0.0.1:5000/' #currently my model server

st.set_option('deprecation.showPyplotGlobalUse', False)
#st.get_option('theme.backgroundColor')

st.title('MNIST dataset – ML prediction')
st.subheader('Visualising hidden layers in a Neural Network')

st.sidebar.write("")
st.sidebar.markdown("<h2 style='text-align: center;'>Image input</h2>",
                    unsafe_allow_html=True)  # make sidebar fixed later

st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgba(252, 203, 3, 0.77);
    color: rgba(7, 7, 7, 0.9);
}
</style>""",
            unsafe_allow_html=True)


def reversed_enumerate(a_list):
    return zip(range(len(a_list) - 1, -1, -1), a_list[::-1])

col1, col2 = st.columns(1, 1)



if st.button('Get random image'):
    response = requests.post(URI, data={}).json()
    preds = response.get('prediction')
    image = response.get('image')
    image = np.reshape(image, (28, 28))

    col1, col2, col3 = st.sidebar.columns([2,5,1])

    with col1:
        st.write("")

    with col2:
        st.image(image, width=150)
        st.write('Image from the MNIST database which contains 70,000 images'
                 ' of handwritten digits from 1 to 10.')
        st.write("")
        st.write("")

    with col3:
        st.write("")


    def layer_text(layer):
        if layer == len(preds) - 1: # reversing order below
            return "Prediction"
        else:
            return "Hidden layer"


    for layer, pred in reversed_enumerate(preds):
        numbers = np.squeeze(np.array(pred)) # removes additional dimensionality of the data

        plt.figure(figsize=(32, 4))
        if layer == preds[-1]: # final output layer
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

        st.text(f'{layer_text(layer)} – Layer {layer + 1}')
        st.pyplot()

if st.button('Draw a digit yourself'):

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width="1, 25, 3",
        background_color="#eee",
        update_streamlit=st.sidebar.checkbox("Update in realtime", True),
        height=150,
        key="full_app",
    )
