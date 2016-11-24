# -*- coding: utf-8 -*-
"""

"""

import logging as log
import sys
import time

import numpy as np
import seaborn as sb
import keras

from echonet.utils.generics import to_percentage

log.basicConfig(stream=sys.stderr, level=log.WARN)


class EchoNet:
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

        self.net.fit_generator(generator=self.dataset.iterbatches(batch_size),
                               samples_per_epoch=epoch_size,
                               nb_epoch=epochs,
                               callbacks=[EchoNet.Monitor(self)],
                               verbose=0)

    class Monitor(keras.callbacks.Callback):
        """

        """
        def __init__(self, echonet):
            assert isinstance(echonet, EchoNet)
            self.echonet = echonet
            self.loss = []
            self.train_score = []
            self.validation_score = []
            self.test_score = []
            self.start_time = None

            #self.f = plt.figure(figsize=(12, 12))
            self.cmap = sb.diverging_palette(220, 10, as_cmap=True)

        def on_train_begin(self, logs={}):
            self.start_time = time.time()

            with open(self.echonet.name + '.log', 'w') as f:
                f.write(self.echonet.name + '\n')

        def on_batch_end(self, batch, logs={}):
            pass # TODO: progress update

        def on_epoch_end(self, epoch, logs={}):
            self.loss.append(logs.get('loss'))
            self.train_score.append(logs.get('acc'))
            time_in_training = time.time() - self.start_time

            validation_score = self.echonet.dataset.validate(self.echonet.net)
            self.validation_score.append(validation_score)

            test_score = self.echonet.dataset.test(self.echonet.net)
            self.test_score.append(test_score)

            time_elapsed = time.time() - self.start_time
            self.start_time = time.time()

            epoch_summary = 'Epoch: {:>3} - {:>5.1f} s / {:>5.1f} s | '.format(epoch,
                    time_in_training, time_elapsed)
            epoch_summary += 'Train: {:>6.2f} % | '.format(to_percentage(self.train_score[-1]))
            epoch_summary += 'Validation: {:>6.2f} % | '.format(to_percentage(self.validation_score[-1]))
            epoch_summary += 'Test: {:>6.2f} %'.format(to_percentage(self.test_score[-1]))

            print(epoch_summary)
            sys.stdout.flush()
            with open(self.echonet.name + '.log', 'a') as f:
                f.write(epoch_summary + '\n')

            # TODO: proper UI handling
            # TODO: stop training = model.stop_training is True
            # TODO: save_weights
            # TODO: save_state - https://github.com/fchollet/keras/issues/454
