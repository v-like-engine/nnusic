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
from core.constants import MIDI, CORPUS_DIVISION_LENGTH
from core.parse_dataset import get_dataset
from core.preprocess_dataset_midi import get_stream_from_snippet, get_symbol_mappings, get_symbol_corpus

np.random.seed(42)


class Synthesizer:
    def __init__(self, model, notes_corpus, seed):
        self.model = model
        self.symbol_corpus = get_symbol_corpus(notes_corpus)
        self.r_mapping = get_symbol_mappings(self.symbol_corpus)[1]
        self.seed = seed

    def compose(self, notes_count):
        seed = self.seed[np.random.randint(0, len(self.seed) - 1)]
        track = ""
        notes_composed=[]
        for i in range(notes_count):
            seed = seed.reshape(1, CORPUS_DIVISION_LENGTH, 1)
            prediction = self.model.predict(seed, verbose=0)[0]
            prediction = np.log(prediction) / 1.0  # diversity
            exp_preds = np.exp(prediction)
            prediction = exp_preds / np.sum(exp_preds)
            index = np.argmax(prediction)
            idx_n = index / float(len(self.symbol_corpus))
            notes_composed.append(index)
            track = [self.r_mapping[char] for char in notes_composed]
            seed = np.insert(seed[0], len(seed[0]), idx_n)
            seed = seed[1:]
        # convert to MIDI
        melody = get_stream_from_snippet(track)
        melody_midi = stream.Stream(melody)
        return track, melody_midi
