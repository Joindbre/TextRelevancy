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
import numpy as np

from classifier.vectorizeData import VectorizeData
from classifier.nnModel import NNModel
from classifier.data import Data
from logger import set_up_logging
from sklearn.utils import shuffle
model_file = '/resources/classifier/model/'


class TrainService(object):
    vectorizeData = VectorizeData()
    nnModel = NNModel()
    data = Data()
    log = set_up_logging()


    def train(self, version):
        positive = self.data.loadNPZ(version, 'Positive_Relevant')
        negative = self.data.loadNPZ(version, 'Negative_Relevant')
        negative = shuffle(negative, random_state=63)

        self.log.debug('#Positive relevant: %s ', len(positive))
        self.log.debug('#Negative relevant: %s ', len(negative))
        # Balance data #
        # negative_2test = negative[len(positive):len(positive) + len(negative)]
        negative = negative[:len(positive)]

        relevant = positive + negative
        labels = np.concatenate((np.ones(len(positive)), np.zeros(len(negative))))

        #######Shuffle #######
        relevant, labels = shuffle(relevant, labels, random_state=42)
        self.log.debug('#Total samples:  %s' % len(relevant))

        ######Split test data
        len_samples = int(len(relevant) * .90);
        len_test_samples = int(len(relevant) * .10);

        x_samples = relevant[:len_samples]
        y_samples = labels[:len_samples]
        x_test = relevant[len_samples:len_samples + len_test_samples]
        y_test = labels[len_samples:len_samples + len_test_samples]
        self.log.debug('#Train & Validation samples length:  %s' % len(x_samples))
        self.log.debug('#Train & Validation labels length:  %s' % len(y_samples))

        ######Tokenize data
        folder = os.path.abspath(os.getcwd()) + model_file + version
        if not os.path.exists(folder):
            os.makedirs(folder)
        data = self.vectorizeData.encoding_vocabulary_index(version, x_samples)  #create_index
        labels = np.asarray(y_samples)
        self.log.debug('Shape of data tensor: ' + str(data.shape))
        self.log.debug('Shape of label tensor: ' + str(labels.shape))
        train = self.nnModel.train_model_kfold(folder, data, labels)
        #train = self.nnModel.train_model(folder, data, labels)

        self.log.debug('#Test samples length:  %s', len(x_test))

        ############## TEST
        test_data = self.vectorizeData.encoding_from_file(version, x_test)
        y_test = np.asarray(y_test)
        test = self.nnModel.test_model(folder, test_data, y_test)

        return dict(list(train.items()) + list(test.items()))

    #
    def extract_relevant(self, version):
        self.data.save_relevant_paragrahps('Positive', version)
        self.data.save_relevant_paragrahps('Negative', version)
