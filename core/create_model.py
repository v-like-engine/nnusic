import tensorflow
import numpy as np
import pandas as pd
from collections import Counter
import random
import music21
from music21 import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adamax
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
from core.constants import MIDI, MODEL_LR
from core.parse_dataset import get_dataset
np.random.seed(42)


def create_model(input_shape, output_shape, lr=MODEL_LR):
    model = Sequential()
    model.add(LSTM(512, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(0.1))
    model.add(Dense(output_shape, activation='softmax'))
    # Compiling the model for training
    opt = Adamax(learning_rate=lr)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    return model
