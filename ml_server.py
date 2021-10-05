import json
import tensorflow as tf
import numpy as np
from flask import Flask, request

app = Flask(__name__)

# simple model that outputs just the final layer
model = tf.keras.models.load_model('mnist_model.h5')

# model for visualisation, outputs all layers
model_all_layers = tf.keras.models.Model(
    model.inputs,
    [layer.output for layer in model.layers]
    )

# load MNIST dataset again, this time only X_test
_, (X_test, _) = tf.keras.datasets.mnist.load_data()
X_test = X_test / 255. # not reshaping it here because want to visualise

def get_prediction():
    """ Return outputs of predict method, all layers"""
    index = np.random.choice(len(X_test))
    image = X_test[index, :, :] # i.e. random image
    image_arr = np.reshape(image, (1, 784))

    return model_all_layers.predict(image_arr), image


@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'POST':
        preds, image = get_prediction()
        final_preds = [pred.tolist() for pred in preds]
        return json.dumps({
            'prediction': final_preds,
            'image': image.tolist()
        })

    return 'Welcome to the model server!'

if __name__ == '__main__':
    app.run()
