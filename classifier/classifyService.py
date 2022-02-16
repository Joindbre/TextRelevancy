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

from classifier.vectorizeData import VectorizeData
from classifier.nnModel import NNModel
from logger import set_up_logging
from nlp.searchService import SearchService
words_folder='/resources/classifier/keywords/*'

class ClassifyService(object):
    searchService = SearchService()
    vectorizeData = VectorizeData()
    nnModel = NNModel()
    log = set_up_logging()

    def classify(self, version, text):
        folder = os.path.abspath(os.getcwd()) + words_folder
        paragraph_list, found_word_list = self.searchService.searchRelevant(text, folder)
        self.log.debug('Number of relevant paragraphs %s', len(paragraph_list))
        if len(paragraph_list) == 0:
            return {'relevant': False};
        else:
            joined_paragraph_list = ' '.join(str(x) for x in paragraph_list)
            # Vectorize
            texts_vectorized = self.vectorizeData.encoding_from_file(version, [joined_paragraph_list])
            # Predict
            result = self.nnModel.predict(version, texts_vectorized);
            self.log.debug('Class prediction result : %s', result[0][0])
            return {'relevant': True,
                    'result': float(result[0][0])};
