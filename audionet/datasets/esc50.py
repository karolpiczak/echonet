# -*- coding: utf-8 -*-
"""Dataset wrappers for the ESC dataset.

Work in progress...

"""

import os

import librosa
import numpy as np
import pandas as pd
import scipy.signal
from tqdm import tqdm

from audionet.datasets.dataset import _load_audio, to_one_hot, Dataset


class ESC(Dataset):
    """

    """
    def __init__(self, data_dir, work_dir, train_folds, validation_folds, test_folds,
                 bands=200, esc10=False):
        super().__init__(data_dir, work_dir)

        self.meta = pd.read_csv(data_dir + 'esc50.csv')

        self.train_folds = train_folds
        self.validation_folds = validation_folds
        self.test_folds = test_folds

        self.class_count = 50

        self.bands = bands
        self.segment_length = 400

        self.esc10 = esc10
        if self.esc10:
            self.class_count = 10
            self.meta = self.meta[self.meta['esc10']]
            self.categories = pd.unique(self.meta.sort_values('target')['category'])
            self.meta['target'] = self.to_targets(self.meta['category'])
        else:
            self.categories = pd.unique(self.meta.sort_values('target')['category'])

        self._generate_spectrograms()

    def _generate_spectrograms(self):
        for row in tqdm(self.meta.itertuples(), total=len(self.meta)):
            specfile = self.work_dir + row.filename + '.mel{}.spec.npy'.format(self.bands)

            if os.path.exists(specfile):
                continue

            audio = _load_audio(self.data_dir + 'audio/' + row.filename)
            spec = self._compute_spectrogram(audio)
            np.save(specfile, spec, allow_pickle=False)

    def _compute_spectrogram(self, audio):
        spec = librosa.feature.melspectrogram(audio, sr=44100, n_fft=4410, hop_length=441,
                                              n_mels=self.bands, fmax=16000)
        mel_freq = librosa.core.mel_frequencies(n_mels=self.bands, fmax=16000)
        spec = librosa.core.perceptual_weighting(spec, mel_freq, ref_power=np.max)
        return spec[:, 0:500]

    def _extract_segment(self, filename):
        spec = np.load(self.work_dir + filename + '.mel{}.spec.npy'.format(self.bands))
        offset = np.random.randint(0, np.shape(spec)[1] - self.segment_length + 1)
        return spec[:, offset:offset + self.segment_length]

    def _get_data_batch(self, looper, batch_size):
        while True:
            X, y = [], []

            for i in range(batch_size):
                row = next(looper)
                X.append(self._extract_segment(row.filename))
                y.append(row.target)

            X = np.stack(X)
            X = X[:, np.newaxis, :, :]
            y = to_one_hot(np.array(y), self.class_count)

            if self.data_mean is None:
                self.data_mean = np.mean(X)
                self.data_std = np.std(X)

            X -= self.data_mean
            X /= self.data_std

            yield X, y

    def to_categories(self, targets):
        return self.categories[targets]

    def to_targets(self, categories):
        return [np.argmax(self.categories == name) for name in categories]

    def get_train_batch(self, batch_size):
        looper = super()._loop_dataset(self.meta[self.meta['fold'].isin(self.train_folds)])
        return self._get_data_batch(looper, batch_size)

    def get_train_size(self):
        return len(self.meta[self.meta['fold'].isin(self.train_folds)])

    def get_validation_batch(self, batch_size):
        looper = super()._loop_dataset(self.meta[self.meta['fold'].isin(self.validation_folds)])
        return self._get_data_batch(looper, batch_size)

    def get_validation_size(self):
        return len(self.meta[self.meta['fold'].isin(self.validation_folds)])

    def get_test_all(self):
        looper = super()._loop_dataset(self.meta[self.meta['fold'].isin(self.test_folds)], False)
        return self._get_data_batch(looper, self.get_test_size())

    def get_test_size(self):
        return len(self.meta[self.meta['fold'].isin(self.test_folds)])

    def get_input_shape(self):
        return (1, self.bands, self.segment_length)

    def score(self, targets, predictions, ensembling_mode=None):
        if ensembling_mode == 'average':
            predictions = np.average(predictions, axis=0)

        target_classes = np.argmax(targets, axis=1)
        predicted_classes = np.argmax(predictions, axis=1)
        return np.sum(target_classes == predicted_classes) / len(target_classes)


class OriginalESC(Dataset):
    """

    """
    def __init__(self, data_dir, work_dir, train_folds, validation_folds, test_folds, esc10=False):
        super().__init__(data_dir, work_dir)

        self.meta = pd.read_csv(data_dir + 'esc50.csv')

        self.train_folds = train_folds
        self.validation_folds = validation_folds
        self.test_folds = test_folds

        self.class_count = 50

        self.bands = 60
        self.segment_length = 101

        self.esc10 = esc10
        if self.esc10:
            self.class_count = 10
            self.meta = self.meta[self.meta['esc10']]
            self.categories = pd.unique(self.meta.sort_values('target')['category'])
            self.meta['target'] = self.to_targets(self.meta['category'])
        else:
            self.categories = pd.unique(self.meta.sort_values('target')['category'])

        self._generate_spectrograms()

    def _generate_spectrograms(self):
        for row in tqdm(self.meta.itertuples(), total=len(self.meta)):
            specfile = self.work_dir + row.filename + '.orig.spec.npy'.format(self.bands)

            if os.path.exists(specfile):
                continue

            audio = _load_audio(self.data_dir + 'audio/' + row.filename, 22050)
            audio *= 1.0 / np.max(np.abs(audio))

            spec = self._compute_spectrogram(audio)
            np.save(specfile, spec, allow_pickle=False)

    def _compute_spectrogram(self, audio):
        spec = librosa.feature.melspectrogram(audio, sr=22050, n_fft=1024, hop_length=512,
                                              n_mels=self.bands)
        spec = librosa.logamplitude(spec)
        return spec

    def _extract_segment(self, filename):
        spec = np.load(self.work_dir + filename + '.orig.spec.npy')
        offset = np.random.randint(0, np.shape(spec)[1] - self.segment_length + 1)
        spec = spec[:, offset:offset + self.segment_length]
        delta = self._generate_delta(spec)
        return np.stack([spec, delta])

    def _generate_delta(self, spec):
        # ported librosa v0.3.1. implementation
        window = np.arange(4, -5, -1)
        padding = [(0, 0), (5, 5)]
        delta = np.pad(spec, padding, mode='edge')
        delta = scipy.signal.lfilter(window, 1, delta, axis=-1)
        idx = [Ellipsis, slice(5, -5, None)]
        return delta[idx]

    def _get_data_batch(self, looper, batch_size):
        while True:
            X, y = [], []

            for i in range(batch_size):
                row = next(looper)
                X.append(self._extract_segment(row.filename))
                y.append(row.target)

            X = np.stack(X)
            y = to_one_hot(np.array(y), self.class_count)

            if self.data_mean is None:
                self.data_mean = np.mean(X)
                self.data_std = np.std(X)

            X -= self.data_mean
            X /= self.data_std

            yield X, y

    def to_categories(self, targets):
        return self.categories[targets]

    def to_targets(self, categories):
        return [np.argmax(self.categories == name) for name in categories]

    def get_train_batch(self, batch_size):
        looper = super()._loop_dataset(self.meta[self.meta['fold'].isin(self.train_folds)])
        return self._get_data_batch(looper, batch_size)

    def get_train_size(self):
        return len(self.meta[self.meta['fold'].isin(self.train_folds)])

    def get_validation_batch(self, batch_size):
        looper = super()._loop_dataset(self.meta[self.meta['fold'].isin(self.validation_folds)])
        return self._get_data_batch(looper, batch_size)

    def get_validation_size(self):
        return len(self.meta[self.meta['fold'].isin(self.validation_folds)])

    def get_test_all(self):
        looper = super()._loop_dataset(self.meta[self.meta['fold'].isin(self.test_folds)], False)
        return self._get_data_batch(looper, self.get_test_size())

    def get_test_size(self):
        return len(self.meta[self.meta['fold'].isin(self.test_folds)])

    def get_input_shape(self):
        return (2, self.bands, self.segment_length)

    def score(self, targets, predictions, ensembling_mode=None):
        if ensembling_mode == 'average':
            predictions = np.average(predictions, axis=0)

        target_classes = np.argmax(targets, axis=1)
        predicted_classes = np.argmax(predictions, axis=1)
        return np.sum(target_classes == predicted_classes) / len(target_classes)
