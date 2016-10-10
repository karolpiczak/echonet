# -*- coding: utf-8 -*-
"""

"""

import logging as log
import sys
import time

import seaborn as sb
import keras

log.basicConfig(stream=sys.stderr, level=log.DEBUG)


class AudioNet:
    """

    """
    def __init__(self, name, layers, optimizer=None, loss='categorical_crossentropy'):
        self.name = name
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.early_stopping = 0
        self.dataset = None

        if self.optimizer is None:
            self.optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)

        self.net = keras.models.Sequential()

        for layer in self.layers:
            self.net.add(layer)

        self.net.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        log.info('Keras model compilation finished.')

    def __str__(self):
        description = 'Model layers / shapes / parameters:\n'
        total_params = 0

        for layer in self.net.layers:
            layer_params = layer.count_params()
            description += '- {}'.format(layer.name).ljust(30)
            description += '{}'.format(layer.input_shape).ljust(30)
            description += '{0:,}'.format(layer_params).rjust(12)
            description += '\n'
            total_params += layer_params

        description += 'Total:'.ljust(30)
        description += '{0:,}'.format(total_params).rjust(42)

        return description

    def fit(self, dataset, batch_size=100, epoch_size=None, epochs=1000, early_stopping=0):
        self.dataset = dataset
        self.early_stopping = early_stopping

        if epoch_size is None:
            epoch_size = self.dataset.get_train_size()

        self.net.fit_generator(generator=self.dataset.get_train_batch(batch_size),
                               samples_per_epoch=epoch_size,
                               nb_epoch=epochs,
                               callbacks=[AudioNet.Monitor(self)],
                               verbose=0)

    class Monitor(keras.callbacks.Callback):
        """

        """
        def __init__(self, audionet):
            assert isinstance(audionet, AudioNet)
            self.audionet = audionet
            self.loss = []
            self.train_score = []
            self.validation_score = []
            self.test_score = []
            self.start_time = None

            validation_size = self.audionet.dataset.get_validation_size()
            self.validation_generator = self.audionet.dataset.get_validation_batch(validation_size)
            self.test_generator = self.audionet.dataset.get_test_all()

            #self.f = plt.figure(figsize=(12, 12))
            self.cmap = sb.diverging_palette(220, 10, as_cmap=True)

        def on_train_begin(self, logs={}):
            self.start_time = time.time()
            with open(self.audionet.name + '.log', 'w') as model_log:
                model_log.write(self.audionet.name + '\n')

        def on_batch_end(self, batch, logs={}):
            pass # TODO: progress update

        def on_epoch_end(self, epoch, logs={}):
            self.loss.append(logs.get('loss'))
            self.train_score.append(logs.get('acc'))

            epoch_summary = '\n'
            epoch_summary += 'Epoch {}\n'.format(epoch)

            epoch_summary += '- Train time:       {0:.2f}\n'.format(time.time() - self.start_time)
            self.start_time = time.time()

            validation_data = next(self.validation_generator)
            validation_pred = self.audionet.net.predict((validation_data[0]))
            validation_score = self.audionet.dataset.score(validation_data[1], validation_pred)
            self.validation_score.append(validation_score)

            epoch_summary += '- Validation time:  {0:.2f}\n'.format(time.time() - self.start_time)
            self.start_time = time.time()

            test_pred = []
            for i in range(3):
                test_data = next(self.test_generator)
                test_pred.append(self.audionet.net.predict(test_data[0]))
            test_score = self.audionet.dataset.score(test_data[1], test_pred, 'average')
            self.test_score.append(test_score)

            epoch_summary += '- Testing time:     {0:.2f}\n'.format(time.time() - self.start_time)
            self.start_time = time.time()

            time_elapsed = time.time() - self.start_time
            self.start_time = time.time()

            epoch_summary += '# Train score:      {0:.3f}\n'.format(self.train_score[-1])
            epoch_summary += '# Validation score: {0:.3f}\n'.format(self.validation_score[-1])
            epoch_summary += '# Test score:       {0:.3f}\n'.format(self.test_score[-1])

            print(epoch_summary)
            with open(self.audionet.name + '.log', 'a') as model_log:
                model_log.write(epoch_summary)

            # TODO: proper UI handling
            # TODO: stop training = model.stop_training is True
            # TODO: save_weights
            # TODO: save_state - https://github.com/fchollet/keras/issues/454
