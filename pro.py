import numpy as np
import tensorflow as tf
import sys
import time
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,LSTM,Conv1D,Input,concatenate
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from processing import build_dataset
from utils import init_session
from processing import data_generator
from keras.layers.convolutional import ZeroPadding1D
new_model = tf.keras.models.load_model('file/CNN_LSTM4_1_model')
x="https://stackoverflow.com/questions/46086030/how-to-check-which-version-of-keras-is-installed"
from utils import GeneSeg
h=GeneSeg(x)
w=data_generator(x)
print(w)
new_model.summary(w)