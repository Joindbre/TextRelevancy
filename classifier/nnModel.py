#  Copyright (c) 2020-present Joindbre.com
#  Authors: Patricia Maraver.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the GNU Affero General Public License version 3
#  as published by the Free Software Foundation.
#  You should have received a copy of the Server Side Public License
#  along with this program. If not, see https://www.gnu.org/licenses/agpl-3.0.en.html
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero General Public License for more details.
#
#   If use of the software under the AGPL v3.0 does not fit you or your company,
#   commercial licenses are also available with Joindbre (https://www.joindbre.com/).
#   Feel free to contact us for more details at this address: sales@joindbre.com

import os

from sklearn.model_selection import KFold

from logger import set_up_logging
from keras import models, layers
from tensorflow import optimizers
import numpy as np

model_folder = '/resources/classifier/model/'

class NNModel(object):
    log = set_up_logging()

    def predict(self, version, test):
        # load model
        self.log.debug('Predicting class')
        file = os.path.abspath(os.getcwd()) + model_folder + version + '/model.h5'
        self.log.debug("Reading model from file %s", file)
        model2predict = models.load_model(file)
        # predict class
        result = model2predict.predict(test)
        return result


    def train_model_kfold(self, folder, x_samples, y_samples):

        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=10, shuffle=True)
        accuracy = []
        loss = []
        # K-fold Cross Validation model evaluation
        n_fold = 1
        for train, val in kfold.split(x_samples, y_samples):

            self.log.debug('#Training articles: %s ', len(train))
            self.log.debug('#Validation articles: %s ', len(val))

            # create the NN
            model = models.Sequential()
            model.add(layers.Dense(64,
                                   activation='relu', input_shape=(10000,)))
            model.add(layers.Dense(64,
                                   activation='relu', input_shape=(10000,)))
            model.add(layers.Dense(1, activation='sigmoid'))

            # compile the model
            model.compile(optimizer=optimizers.SGD(lr=0.01),
                          loss='mean_squared_error',
                          metrics=['acc'])
            # self.log.debug(model.summary())

            self.log.debug('Training for fold %s', n_fold)

            result = model.fit(x_samples[train],
                                y_samples[train],
                                epochs=30,
                                batch_size=8)

            scores = model.evaluate(x_samples[val],
                                    y_samples[val],
                                    verbose=0)
            accuracy.append(scores[1] * 100)
            loss.append(scores[0])

            # Increase fold number
            n_fold = n_fold + 1
        # res = {'trainAccuracy': result.history['acc'],
        #         'trainLoss': result.history['loss'],
        #         'epochs': len(result.history['acc']),
        #         'trainSamples': len(x_train)}
        self.log.debug('Accuracy %s', accuracy)
        self.log.debug('Loss %s', loss)

        # return res

    def train_model(self, folder, x_samples, y_samples):

        len_samples = int(len(x_samples) * .90);
        len_val_samples = int(len(x_samples) * .10);

        x_train = x_samples[:len_samples]
        y_train = y_samples[:len_samples]
        x_val = x_samples[len_samples:len_samples + len_val_samples]
        y_val = y_samples[len_samples:len_samples + len_val_samples]

        self.log.debug('#Training articles: ' + str(len(x_train)))
        self.log.debug('#Validation articles: ' + str(len(x_val)))

        # create the neural network
        model = models.Sequential()
        model.add(layers.Dense(64,
                               activation='relu', input_shape=(10000,)))
        model.add(layers.Dense(64,
                               activation='relu', input_shape=(10000,)))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizers.SGD(lr=0.01),
                      loss='mean_squared_error',
                      metrics=['acc'])
        self.log.debug(model.summary())

        # history = model.fit(x_train,
        #                     y_train,
        #                     epochs=30,
        #                     batch_size=8,
        #                     validation_data=(x_val, y_val))

        complete_data = np.concatenate((x_train, x_val))
        complete_labels = np.concatenate((y_train, y_val))
        result = model.fit(complete_data,
                            complete_labels,
                            epochs=30,
                            batch_size=8)

        file = folder + '/model.h5'
        self.log.debug('Writing model to file %s', file)
        model.save(file)
        res = {'trainAccuracy': result.history['acc'],
                'trainLoss': result.history['loss'],
                'epochs': len(result.history['acc']),
                'trainSamples': len(x_train)}
        self.log.debug(res)
        return res

    def test_model(self, folder, x_test, y_test):
        file = folder + '/model.h5'
        model = models.load_model(file)
        result = model.evaluate(x_test, y_test)
        y_pred = model.predict(x_test)
        result_list = []
        for i in range(len(y_pred)):
            result_list.append([int(y_test[i]), float(y_pred[i][0])])
        res = {'testAccuracy': result[1],
                'testLoss': result[0],
                'testSamples': len(x_test),
                'classList': result_list
               }
        self.log.debug(res)
        return res
