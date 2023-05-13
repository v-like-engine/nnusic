import os
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

np.random.seed(42)


def extract_notes(file):
    notes = []
    for j in file:
        if j:
            songs = instrument.partitionByInstrument(j)
            for part in songs.parts:
                pick = part.recurse()
                for element in pick:
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        notes.append(".".join(str(n) for n in element.normalOrder))
    return notes


def get_stream_from_snippet(snippet):
    melody = []
    offset = 0
    for i in snippet:
        # process a chord
        if "." in i or i.isdigit():
            chord_notes = i.split(".")
            notes = []
            for j in chord_notes:
                inst_note = int(j)
                note_snip = note.Note(inst_note)
                notes.append(note_snip)
                chord_snip = chord.Chord(notes)
                chord_snip.offset = offset
                melody.append(chord_snip)
        # process a note
        else:
            note_snip = note.Note(i)
            note_snip.offset = offset
            melody.append(note_snip)
        # increase offset each iteration so that notes do not stack
        offset += 1
    melody_midi = stream.Stream(melody)
    return melody_midi


def eliminate_not_frequent_notes(notes_corpus: list):
    count_num = Counter(notes_corpus)
    rare_note = []
    for index, (key_, value) in enumerate(count_num.items()):
        if value < 100:
            m = key_
            rare_note.append(m)
    for element in notes_corpus:
        if element in rare_note:
            notes_corpus.remove(element)
    return len(notes_corpus)


def get_symbol_mappings(symbol_corpus):
    # Building dictionary to access the vocabulary from indices and vice versa
    mapping = dict((c, i) for i, c in enumerate(symbol_corpus))
    reverse_mapping = dict((i, c) for i, c in enumerate(symbol_corpus))
    return mapping, reverse_mapping


def get_symbol_corpus(notes_corpus):
    return sorted(list(set(notes_corpus)))


def get_features_targets(notes_corpus: list, corp_len=None, symbol_corpus=None, mapping=None) -> (list, list):
    # Splitting the Corpus in equal length of strings and output target
    features_ = []
    targets_ = []
    if not corp_len:
        corp_len = len(notes_corpus)
    if not symbol_corpus:
        symbol_corpus = get_symbol_corpus(notes_corpus)
    if not mapping:
        mapping = get_symbol_mappings(symbol_corpus)[0]
    for i in range(0, corp_len - CORPUS_DIVISION_LENGTH, 1):
        feature = notes_corpus[i:i + CORPUS_DIVISION_LENGTH]
        target = notes_corpus[i + CORPUS_DIVISION_LENGTH]
        features_.append([mapping[j] for j in feature])
        targets_.append(mapping[target])
    X = (np.reshape(features_, (len(targets_), CORPUS_DIVISION_LENGTH, 1))) / float(len(symbol_corpus))
    # one hot encode the output variable
    y = tensorflow.keras.utils.to_categorical(targets_)
    return X, y


def preprocess(dataset):
    notes_corpus = extract_notes(dataset)
    corp_len = eliminate_not_frequent_notes(notes_corpus)
    return get_features_targets(notes_corpus, corp_len)
