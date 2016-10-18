# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import scipy.signal

import pydub


# noinspection PyProtectedMember
def load_audio(path, sr=44100):
    audio = pydub.AudioSegment.from_file(path).set_frame_rate(sr).set_channels(1)
    return (np.fromstring(audio._data, dtype="int16") + 0.5) / (0x7FFF + 0.5)  # convert to float


def to_percentage(number):
    return int(number * 1000) / 10.0


def to_one_hot(targets, class_count):
    one_hot_enc = np.zeros((len(targets), class_count))

    for r in range(len(targets)):
        one_hot_enc[r, targets[r]] = 1

    return one_hot_enc


def generate_delta(spec):
    # ported librosa v0.3.1. implementation
    window = np.arange(4, -5, -1)
    padding = [(0, 0), (5, 5)]
    delta = np.pad(spec, padding, mode='edge')
    delta = scipy.signal.lfilter(window, 1, delta, axis=-1)
    idx = [Ellipsis, slice(5, -5, None)]
    return delta[idx]
