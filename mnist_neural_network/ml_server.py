import json
import tensorflow as tf
import numpy as np
import random
from flask import Flask, request

app = Flask(__name__)

# 'normal' model that outputs just the final layer
model = tf.keras.models.load_model('mnist_model.h5') # ck path!

model_all_layers = tf.keras.models.Model(
    model.inputs,
    [layer.output for layer in model.layers], # currently: 3 layers
    )




@app.route('/')
def index():
    return 'Welcome to the model server!'

if __name__ == '__main__':
    app.run()
