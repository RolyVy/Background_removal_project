import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, save_model, load_model
from data_generator import test_gen
mdl=tf.keras.models.load_model('models/final.04-0.5232.hdf5')

mdl.predict(test_gen())
