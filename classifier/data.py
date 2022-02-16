
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
import json
from nlp.searchService import SearchService
from logger import set_up_logging
words_folder='/resources/classifier/keywords/*'

data_folder='/resources/data/'

class Data(object):
    searchService = SearchService()
    log = set_up_logging()

    def save_relevant_paragrahps(self, collection, version):

        relevant_paragraph_list = []
        j = 1
        complete_words_folder = os.path.abspath(os.getcwd()) + words_folder

        folder = os.path.abspath(os.getcwd()) + data_folder + version + "/"
        self.log.debug('Reading all files from folder %s  ', folder)
        while True:
            file_path = folder + collection + '_' + str(j) + '.json'
            if os.path.isfile(file_path):
                file = folder + collection + '_Relevant_' + str(j) + '.npz'
                if not os.path.isfile(file):
                    with open(os.path.join(file_path), 'r') as f:
                        self.log.debug('Loading data from file: ' + file_path)
                        text_list = json.loads(f.read())
                        self.log.debug('Samples length: %s', len(text_list))
                    i = 0
                    for text in text_list:
                        try:
                            paragraph_list, found_word_list = self.searchService.searchRelevant(text, complete_words_folder)
                            paragraph_text = ' '.join(paragraph_list)
                            relevant_paragraph_list.append(paragraph_text)
                            self.log.debug('sample %s relevant paragraphs %s out of %s', i, len(paragraph_list), len(text))
                        except Exception as e:
                            self.log.error('Error reading sample %s %s', i, text[0])
                            self.log.error(e)
                        i = i+1
                    relevant_paragraph_array = np.asarray(relevant_paragraph_list)
                    np.savez(folder + collection + '_Relevant_' + str(j) + '.npz', text=relevant_paragraph_array)
                    relevant_paragraph_list = []
                j = j + 1
            else:
                break


    def loadNPZ(self, version, file):
        try:

            i = 1
            samples = []
            folder = os.path.abspath(os.getcwd()) + data_folder + version + "/"
            self.log.debug('Reading all files from folder %s  ', folder)
            while True:
                file_path = folder + file + '_' + str(i) + '.npz'
                if os.path.isfile(file_path):
                    self.log.debug('Loading data from file: ' + file_path)
                    with np.load(file_path, allow_pickle=True) as f:
                        text = f['text']
                        samples.extend(text)
                        i = i + 1
                else:
                    break
            self.log.debug(samples[0])
            return samples;
        except Exception as e:
            self.log.error('Error reading file')
            self.log.error(e)

def listToString(list):
    str1 = " "
    return (str1.join(list))