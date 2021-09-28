import json
import tensorflow as tf
import numpy as np
import random
from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def index():
    return 'Welcome to the model server!'

if __name__ == '__main__':
    app.run()
