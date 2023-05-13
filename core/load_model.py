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
from core.constants import MIDI
from core.create_model import create_model
from core.parse_dataset import get_dataset
from core.preprocess_dataset_midi import preprocess
np.random.seed(42)


def train_model(dataset_settings='all', save_model=False):
    """
        Starting function for music creation.
        Dataset to learn on could be specified. Default: all (every dataset)
        :param dataset_settings:  specified datasets to learn on
        :param save_model: True if trained model should be saved. Default: False
        :return: trained model, notes_corpus
        """
    notes_corpus = get_dataset(dataset_settings)
    X, y = preprocess(notes_corpus)
    X_train, X_seed, y_train, y_seed = train_test_split(X, y, test_size=0.2, random_state=42)
    model = create_model((X.shape[1], X.shape[2]), y.shape[1])
    history = model.fit(X_train, y_train, batch_size=256, epochs=200)
    return model, notes_corpus, X_seed
