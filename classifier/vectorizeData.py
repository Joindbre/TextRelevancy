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
import pickle

from logger import set_up_logging
from keras.preprocessing.text import Tokenizer
mode = 'binary'
tokenizer_file ='/resources/classifier/model/'

class VectorizeData(object):
    log = set_up_logging()

    def encoding_from_file(self, version, texts):
        self.log.debug('Encoding text')
        file = os.path.abspath(os.getcwd()) + tokenizer_file + version + '/tokenizer.pickle'
        # Encode
        self.log.debug('Reading tokenizer from file %s', file)
        with open(file, 'rb') as handle:
            tokenizer = pickle.load(handle)
            handle.close()
        # results = tokenizer.texts_to_sequences(texts)  # vector of indexes
        results = tokenizer.texts_to_matrix(texts, mode=mode)

        return results

    def encoding_vocabulary_index(self, version, texts):
        # Creates a tokenizer, configured to only take into account the # most common words
        tokenizer = Tokenizer(10000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n•\'')

        # Builds the word index
        tokenizer.fit_on_texts(texts)  # word -> index dictionary
        word_index = tokenizer.word_index
        self.log.debug('Found %s unique tokens.', len(word_index))
        file = os.path.abspath(os.getcwd()) + tokenizer_file + version + '/tokenizer.pickle'
        # Encode
        self.log.debug('Writing tokenizer to file %s', file)
        # Save the tokenizer for future predicts

        with open(file, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close

        results = tokenizer.texts_to_matrix(texts, mode=mode)
        return results
