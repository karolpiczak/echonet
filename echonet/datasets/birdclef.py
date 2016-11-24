# -*- coding: utf-8 -*-
"""

"""

from collections import OrderedDict
import os
import subprocess

import librosa
import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.model_selection
import skimage as skim
import skimage.measure
import skimage.morphology
import skimage.restoration
from tqdm import tqdm
import xml.etree.ElementTree

from echonet.datasets.dataset import Dataset
from echonet.utils.generics import generate_delta, load_audio, to_one_hot

from IPython.core.debugger import Tracer


class BirdCLEF2016(Dataset):
    """

    """
    def __init__(self, data_dir, work_dir, downsample=True):
        super().__init__(data_dir, work_dir)

        self.DOWNSAMPLE = downsample
        self.SEGMENT_LENGTH = 500
        self.BANDS = 180
        self.WITH_DELTA = False
        self.FMAX = 16000
        self.FFT = 2205
        self.HOP = 441

        self._resample_recordings()
        self._parse_recordings()
        self._generate_spectrograms()

        if self.DOWNSAMPLE:
            self.SEGMENT_LENGTH //= 2
            self.BANDS //= 3

        self.class_count = len(self.encoder.classes_)

        self._split_dataset()

        self.train_meta = self.meta[self.meta['fold'] == 'train']
        self.validation_data.meta = self.meta[self.meta['fold'] == 'validation']
        self.test_data.meta = self.meta[self.meta['fold'] == 'test']

        self._train_size = len(self.recordings[self.recordings['fold'] == 'train'])
        self._validation_size = len(self.recordings[self.recordings['fold'] == 'validation'])
        self._test_size = len(self.recordings[self.recordings['fold'] == 'test'])

        self._populate(self.validation_data)
        self._populate(self.test_data)

    def _resample_recordings(self):
        src_dir = self.data_dir + 'TrainingSet/wav/'

        for recording in tqdm(sorted(os.listdir(src_dir))):
            if os.path.isfile(src_dir + recording):
                wav_in = src_dir + recording
                wav_out = self.work_dir + recording

                if not os.path.isfile(wav_out):
                    subprocess.call(['sox', '-S', wav_in, '-r', '44100', '-b', '16', wav_out])

    def _parse_recordings(self):
        if os.path.isfile(self.work_dir + 'BirdCLEF2016.csv'):
            self.recordings = pd.read_csv(self.work_dir + 'BirdCLEF2016.csv')
            self.encoder = sk.preprocessing.LabelEncoder()
            self.encoder.fit(self.recordings['birdclass'].values)
        else:
            self.recordings = []

            src_dir = self.data_dir + 'TrainingSet/xml/'
            for recording in tqdm(sorted(os.listdir(src_dir))):
                root = xml.etree.ElementTree.parse(src_dir + recording).getroot()

                data = {
                    'filename': recording[:-4] + '.wav',
                    'birdclass': root.find('ClassId').text,
                    'species': root.find('Species').text,
                    'genus': root.find('Genus').text,
                    'family': root.find('Family').text,
                    'background': root.find('BackgroundSpecies').text
                }

                if data['background'] is None:
                    data['background'] = ''

                columns = ['filename', 'birdclass', 'species', 'genus', 'family', 'background']

                row = pd.DataFrame(data, columns=columns, index=[0])
                self.recordings.append(row)

            self.recordings = pd.concat(self.recordings, ignore_index=True)

            self.encoder = sk.preprocessing.LabelEncoder()
            self.encoder.fit(self.recordings['birdclass'].values)
            self.recordings['target'] = self.encoder.transform(self.recordings['birdclass'].values)

            self.recordings.to_csv(self.work_dir + 'BirdCLEF2016.csv', index=False)

    def _generate_spectrograms(self):
        if os.path.isfile(self.work_dir + 'BirdCLEF2016-clips.csv'):
            self.meta = pd.read_csv(self.work_dir + 'BirdCLEF2016-clips.csv')
        else:
            self.meta = []
            for row in tqdm(self.recordings.itertuples(), total=len(self.recordings)):
                self.meta.extend(self._split_recording(row))
            self.meta = pd.concat(self.meta, ignore_index=True)
            self.meta.to_csv(self.work_dir + 'BirdCLEF2016-clips.csv', index=False)

    def _split_recording(self, row):
        audio = load_audio(self.work_dir + row.filename, 44100)

        spec = librosa.feature.melspectrogram(audio, sr=44100, n_fft=self.FFT, fmax=self.FMAX,
                                              hop_length=self.HOP, n_mels=self.BANDS)
        freqs = librosa.core.mel_frequencies(n_mels=self.BANDS, fmax=self.FMAX)
        spec = librosa.core.perceptual_weighting(spec, freqs, ref_power=np.max)
        spec = self._enhance_spectrogram(spec)
        mask = skim.morphology.dilation(spec, selem=np.ones((3, 40))) > 0
        mask[:10, :] = False

        clip_list = []
        counter = 0
        current = []
        window_size = 25
        w = 0

        while w * window_size < np.shape(spec)[1]:
            window = slice(w * window_size, (w + 1) * window_size)

            if np.any(mask[:, window]):
                current.append(spec[:, window])
            elif len(current):
                clip_list.append(self._save(np.concatenate(current, axis=1), row, counter))
                counter += 1
                current = []
            w += 1

        if len(current):
            clip_list.append(self._save(np.concatenate(current, axis=1), row, counter))

        return clip_list

    def _enhance_spectrogram(self, spec):
        spec = (spec + 60.0) / 15.0  # quasi-normalization
        np.clip(spec, 0, 5, out=spec)
        spec = (spec ** 2 - 6.0) / 6.0

        spec = skim.restoration.denoise_tv_chambolle(spec, weight=0.1)

        spec = ((spec - np.min(spec)) / np.max(spec - np.min(spec)) - 0.5) * 2.0
        spec += 0.5
        spec[spec > 0] *= 2
        spec = ((spec - np.min(spec)) / np.max(spec - np.min(spec)) - 0.5) * 2.0

        return spec

    def _save(self, clip, row, counter):
        reduced_clip = skim.measure.block_reduce(clip, block_size=(3, 2), func=np.mean)
        np.save(self.work_dir + row.filename + '.spec{}.npy'.format(counter),
                clip.astype('float16'), allow_pickle=False)
        np.save(self.work_dir + row.filename + '.spec{}.ds.npy'.format(counter),
                reduced_clip.astype('float16'), allow_pickle=False)

        data = OrderedDict([
            ('filename', row.filename + '.spec{}.npy'.format(counter)),
            ('target', row.target),
            ('recording', row.filename),
            ('birdclass', row.birdclass),
            ('species', row.species),
            ('genus', row.genus),
            ('family', row.family),
            ('background', '' if pd.isnull(row.background) else row.background)
        ])

        return pd.DataFrame(data, columns=data.keys(), index=[0])

    def _split_dataset(self):
        """Splits the dataset into training/validation/testing folds

        Stratified split with shuffling:
        - 75% of recordings go to training
        - 12.5% validation
        - 12.5% testing

        """
        splitter = sklearn.model_selection.StratifiedShuffleSplit
        quarter = splitter(n_splits=1, test_size=0.25, random_state=20161013)
        half = splitter(n_splits=1, test_size=0.5, random_state=20161013)

        train_split = quarter.split(self.recordings['filename'], self.recordings['target'])
        train_idx, holdout_idx = list(train_split)[0]

        holdout_split = half.split(self.recordings.loc[holdout_idx, 'filename'],
                                   self.recordings.loc[holdout_idx, 'target'])
        validation_idx, test_idx = list(holdout_split)[0]

        self.recordings.loc[train_idx, 'fold'] = 'train'
        self.recordings.loc[holdout_idx[validation_idx], 'fold'] = 'validation'
        self.recordings.loc[holdout_idx[test_idx], 'fold'] = 'test'

        right = self.recordings[['filename', 'fold']].rename(columns={'filename': 'recording'})
        self.meta = pd.merge(self.meta, right, on='recording')

    @property
    def input_shape(self):
        return 1 + self.WITH_DELTA, self.BANDS, self.SEGMENT_LENGTH

    @property
    def train_size(self):
        return self._train_size

    @property
    def train_segments(self):
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
        return self.encoder.classes_[targets]

    def to_targets(self, categories):
        return self.encoder.transform(categories)

    def test(self, model):
        return self._score(model, self.test_data)

    def validate(self, model):
        return self._score(model, self.validation_data)

    def _populate(self, data):
        X, y, meta = [], [], []

        for row in tqdm(data.meta.itertuples(), total=len(data.meta)):
            values = dict(zip(row._fields[1:], row[1:]))
            columns = row._fields[1:]
            rows = []

            for _ in range(2):  # multiply segment variants for prediction
                X.append(self._extract_segment(row.filename))
                y.append(row.target)
                rows.append(pd.DataFrame(values, columns=columns, index=[0]))

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
            spec = np.load(self.work_dir + filename[:-4] + '.ds.npy').astype('float32')
        else:
            spec = np.load(self.work_dir + filename).astype('float32')
        spec = spec[:, :-1]  # trim border artifacts

        if np.shape(spec)[1] >= self.SEGMENT_LENGTH:
            offset = self.RandomState.randint(0, np.shape(spec)[1] - self.SEGMENT_LENGTH + 1)
            spec = spec[:, offset:offset + self.SEGMENT_LENGTH]
        else:
            offset = self.RandomState.randint(0, self.SEGMENT_LENGTH - np.shape(spec)[1] + 1)
            overlay = np.zeros((self.BANDS, self.SEGMENT_LENGTH)) - 1.0
            overlay[:, offset:offset + np.shape(spec)[1]] = spec
            spec = overlay

        if self.WITH_DELTA:
            delta = generate_delta(spec)
            return np.stack([spec, delta])
        else:
            return np.stack([spec])

    def _score(self, model, data):
        predictions = pd.DataFrame(model.predict(data.X))
        results = pd.concat([data.meta[['recording', 'target']], predictions], axis=1)
        results = results.groupby('recording').aggregate('mean').reset_index()
        results['predicted'] = np.argmax(results.iloc[:, 2:].values, axis=1)
        return np.sum(results['predicted'] == results['target']) / len(results)
