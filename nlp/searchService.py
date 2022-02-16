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

import glob
import os
import numpy
import spacy

from logger import set_up_logging

class SearchService(object):
    nlp = spacy.load("en_core_web_sm")
    nlp.Defaults.stop_words.add("-")
    nlp.Defaults.stop_words.add("-pron-")  # pronoms
    log = set_up_logging()

    def searchRelevant(self, text, folder):
        self.log.debug('Extracting relevant paragraphs and terms from text from folder %s', folder)
        file_list = glob.glob(folder)

        word_dict = {}
        for file in file_list:
            self.log.debug('Reading terms from %s', file)
            word_np_list = numpy.loadtxt(file, comments="#", delimiter="\n", unpack=False, dtype='str')
            word_dict[os.path.basename(file)] = list(word_np_list)

        relevant_text, found_word_list = self.tokenize(text, word_dict)
        self.log.debug('Terms found in text %s', found_word_list)
        return relevant_text, found_word_list

    def tokenize(self, text, word_dict):
        found_word_dict = dict()
        save_paragraph = False
        relevant_text = []
        for key, value_list in word_dict.items():
            found_word_dict[key]=set()
        for paragraph in text:
            try:
                paragraph_nlp = []
                doc = self.nlp(paragraph)  # Tokenize: Split the paragraph into words
                for token in doc:
                    if not self.nlp.vocab[token.lemma_].is_stop:
                        paragraph_nlp.append(token.lemma_.lower())
                paragraph_text = ' '.join(paragraph_nlp)

                for key, value_list in word_dict.items():
                    for value in value_list:
                        if ' ' + value.lower() + ' ' in paragraph_text or ' ' + value + ' ' in paragraph:
                            save_paragraph = True
                            found_word_dict[key].add(value)
                if save_paragraph:
                    relevant_text.append(paragraph)
                    save_paragraph = False
            except Exception as e:
                self.log.error('Error extracting relevant data from following paragraph %s', paragraph)
        for key, value_list in word_dict.items():
            found_word_dict[key] = list(found_word_dict[key])
        return relevant_text, found_word_dict