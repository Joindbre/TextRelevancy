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

import json
import os

from nlp.searchService import SearchService

import falcon
from logger import set_up_logging

words_folder = '/resources/nlp/keywords/*'


class SearchController(object):
    log = set_up_logging()
    searchService = SearchService()
    """
    @apiVersion 1.0.0
    @api {post} /nlp Search keywords
    @apiDescription Returns the keywords found on the text.
            From a configurable set of keywords stored in files returns the ones found on the provided paragraphs of text. 
            It extracts the token for each word and compares the lemma. The result is a json data format 
            object composed by each of the file names and the list of the terms found.
            Configuration files are stored in the folder /resources/nlp/keywords/ 

    @apiName nlp
    @apiGroup NLP

    @apiBody {String[]} text The list of paragraphs of a text.

    @apiSuccess {String[]} fileName The result is an object key: value[]. Where the key is each of the file names 
                            and the value is the list of terms found on the text.
    @apiSuccessExample {json} Success-Response:
                    {
                        "tracingSystem": [],
                        "cellType": [],
                        "term": ["morphological", "trace"],
                        "keyword": ["tree", "dendrite"]
                    }
    @apiError (5xx) InternalServerError Any of the data is incorrect
    """

    def on_post(self, req, resp):
        try:
            self.log.debug('Getting words from text')
            data = req.media
            folder = os.path.abspath(os.getcwd()) + words_folder
            relevant_text, found_word_list = self.searchService.searchRelevant(data['text'], folder)
            resp.body = json.dumps(found_word_list)

            resp.status = falcon.HTTP_200
        except Exception as e:
            self.log.exception('Error %s', e)
            resp.status = falcon.HTTP_500
