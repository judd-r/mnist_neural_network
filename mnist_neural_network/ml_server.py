import json
import tensorflow as tf
import numpy as np
from flask import Flask, request

app = Flask(__name__)

# 'normal' model that outputs just the final layer
model = tf.keras.models.load_model('mnist_model.h5') # ck path!

model_all_layers = tf.keras.models.Model(
    model.inputs,
    [layer.output for layer in model.layers], # currently: 3 layers
    )

# load dataset again, this time only X_test
_, (X_test, _) = tf.keras.datasets.mnist.load_data()
X_test = X_test / 255. # not reshaping it here bec. want to visualise?

def get_prediction():
    """ Return outputs (incl. y = [0-9] ?) of predict method on NN model
        showing output for all layers
    """
    index = np.random.choice(len(X_test)) # or X_test.shape[0]
    image = X_test[index, :, :] # gives random row
    image_arr = np.reshape(image, (1, 784))

    return model_all_layers.predict(image_arr)


@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'POST':
        predictions, image = get_prediction()
        final_preds = [pred.tolist() for pred in predictions]
        return json.dumps({
            'prediction': final_preds,
            'image': image.tolist()
        })
    return 'Welcome to the model server!'

if __name__ == '__main__':
    app.run()