# -*- coding: utf-8 -*-
"""Dataset wrappers for the ESC dataset.

Work in progress...

"""


import os

import librosa
import numpy as np
import pandas as pd
import scipy.signal
import skimage as skim
import skimage.measure
from tqdm import tqdm

from echonet.datasets.dataset import Dataset
from echonet.utils.generics import generate_delta, load_audio, to_one_hot


class ESC(Dataset):
    """

    """
    def __init__(self, data_dir, work_dir, train_folds, validation_folds, test_folds, esc10=False,
                 downsample=True):
        super().__init__(data_dir, work_dir)

        self.meta = pd.read_csv(data_dir + 'esc50.csv')

        self.train_folds = train_folds
        self.validation_folds = validation_folds
        self.test_folds = test_folds

        self.class_count = 50

        self.DOWNSAMPLE = downsample
        self.SEGMENT_LENGTH = 300
        self.BANDS = 180
        self.WITH_DELTA = False
        self.FMAX = 16000
        self.FFT = 2205
        self.HOP = 441

        self.esc10 = esc10
        if self.esc10:
            self.class_count = 10
            self.meta = self.meta[self.meta['esc10']]
            self.categories = pd.unique(self.meta.sort_values('target')['category'])
            self.meta['target'] = self.to_targets(self.meta['category'])
        else:
            self.categories = pd.unique(self.meta.sort_values('target')['category'])

        self.train_meta = self.meta[self.meta['fold'].isin(self.train_folds)]
        self.validation_data.meta = self.meta[self.meta['fold'].isin(self.validation_folds)]
        self.test_data.meta = self.meta[self.meta['fold'].isin(self.test_folds)]

        self._validation_size = len(self.validation_data.meta)
        self._test_size = len(self.test_data.meta)

        self._generate_spectrograms()

        if self.DOWNSAMPLE:
            self.SEGMENT_LENGTH //= 2
            self.BANDS //= 3

        self._populate(self.validation_data)
        self._populate(self.test_data)

    def _generate_spectrograms(self):
        for row in tqdm(self.meta.itertuples(), total=len(self.meta)):
            specfile = self.work_dir + row.filename + '.mel.spec.npy'

            if os.path.exists(specfile):
                continue

            audio = load_audio(self.data_dir + 'audio/' + row.filename, 44100)
            # audio *= 1.0 / np.max(np.abs(audio))

            spec = librosa.feature.melspectrogram(audio, sr=44100, n_fft=self.FFT, fmax=self.FMAX,
                                                  hop_length=self.HOP, n_mels=self.BANDS)
            # spec = librosa.logamplitude(spec)
            freqs = librosa.core.mel_frequencies(n_mels=self.BANDS, fmax=self.FMAX)
            spec = librosa.core.perceptual_weighting(spec, freqs, ref_power=np.max)

            reduced_spec = skim.measure.block_reduce(spec, block_size=(3, 2), func=np.mean)
            np.save(specfile, spec.astype('float16'), allow_pickle=False)
            np.save(specfile[:-4] + '.ds.npy', reduced_spec.astype('float16'), allow_pickle=False)

    def _populate(self, data):
        X, y, meta = [], [], []

        for row in data.meta.itertuples():
            segments = self._extract_all_segments(row.filename)
            X.extend(segments)
            y.extend(np.repeat(row.target, len(segments)))
            values = dict(zip(row._fields[1:], row[1:]))
            columns = row._fields[1:]
            rows = [pd.DataFrame(values, columns=columns, index=[0]) for _ in range(len(segments))]
            meta.extend(rows)

        X = np.stack(X)
        y = to_one_hot(np.array(y), self.class_count)
        meta = pd.concat(meta, ignore_index=True)

        if self.data_mean is None:
            self.data_mean = np.mean(X)
            self.data_std = np.std(X)

        X -= self.data_mean
        X /= self.data_std

        data.X = X
        data.y = y
        data.meta = meta

    def _extract_all_segments(self, filename):
        if self.DOWNSAMPLE:
            spec = np.load(self.work_dir + filename + '.mel.spec.ds.npy').astype('float32')
        else:
            spec = np.load(self.work_dir + filename + '.mel.spec.npy').astype('float32')

        segments = []
        hop_length = self.SEGMENT_LENGTH // 2
        offset = 0

        while offset < np.shape(spec)[1] - self.SEGMENT_LENGTH:
            segment = spec[:, offset:offset + self.SEGMENT_LENGTH]
            if self.WITH_DELTA:
                delta = generate_delta(segment)
            offset += hop_length
            if self.WITH_DELTA:
                segments.append(np.stack([segment, delta]))
            else:
                segments.append(np.stack([segment]))
        return segments

    @property
    def input_shape(self):
            return 1 + self.WITH_DELTA, self.BANDS, self.SEGMENT_LENGTH

    @property
    def train_size(self):
        return len(self.train_meta)

    @property
    def validation_size(self):
        return self._validation_size

    @property
    def validation_segments(self):
        return len(self.validation_data.meta)

    @property
    def test_size(self):
        return self._test_size

    @property
    def test_segments(self):
        return len(self.test_data.meta)

    def to_categories(self, targets):
        return self.categories[targets]

    def to_targets(self, categories):
        return [np.argmax(self.categories == name) for name in categories]

    def test(self, model):
        return self._score(model, self.test_data)

    def validate(self, model):
        return self._score(model, self.validation_data)

    def iterbatches(self, batch_size):
        itrain = super()._iterrows(self.train_meta)

        while True:
            X, y = [], []

            for i in range(batch_size):
                row = next(itrain)
                X.append(self._extract_segment(row.filename))
                y.append(row.target)

            X = np.stack(X)
            y = to_one_hot(np.array(y), self.class_count)

            X -= self.data_mean
            X /= self.data_std

            yield X, y

    def _extract_segment(self, filename):
        if self.DOWNSAMPLE:
            spec = np.load(self.work_dir + filename + '.mel.spec.ds.npy').astype('float32')
        else:
            spec = np.load(self.work_dir + filename + '.mel.spec.npy').astype('float32')

        offset = self.RandomState.randint(0, np.shape(spec)[1] - self.SEGMENT_LENGTH + 1)
        spec = spec[:, offset:offset + self.SEGMENT_LENGTH]
        if self.WITH_DELTA:
            delta = generate_delta(spec)
            return np.stack([spec, delta])
        else:
            return np.stack([spec])

    def _score(self, model, data):
        predictions = pd.DataFrame(model.predict(data.X))
        results = pd.concat([data.meta[['filename', 'target']], predictions], axis=1)
        results = results.groupby('filename').aggregate('mean').reset_index()
        results['predicted'] = np.argmax(results.iloc[:, 2:].values, axis=1)
        return np.sum(results['predicted'] == results['target']) / len(results)
