import os
import tensorflow
import numpy as np
import pandas as pd
from collections import Counter
import random
import IPython
from IPython.display import Image, Audio
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
from core.constants import MIDI
np.random.seed(42)


def get_dataset(dataset='all'):
    ds_midi = []
    if dataset == 'all':
        for dir_ in MIDI:
            filepath = "./data/midi/" + dir_ + '/'
            # Getting midi files

            for i in os.listdir(filepath):
                if i.endswith(".mid"):
                    try:
                        tr = filepath + i
                        midi_ = converter.parse(tr)
                        ds_midi.append(midi_)
                    except:
                        pass
    return ds_midi
