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

from classifier.classifyService import ClassifyService

import falcon
from logger import set_up_logging


class ClassifyController(object):
    classificationService = ClassifyService()
    log = set_up_logging()

    """
    @apiVersion 1.0.0
    @api {post} /classifier/classify/:version Classify
    @apiDescription Returns a json format data with the likelihood of relevancy of the given text.
            First, it searches if there are relevant paragraphs on the text, a paragraph is relevant if 
            it contains any pre-defined keywords. If there are relevant paragraphs, the text is vectorized 
            using the tokenizer file created in the previous step and sent to the classifier to predict 
            the relevancy likelihood using the model file created in the training step. 
            Otherwise, it is considered irrelevant for the classification task or off-topic.
            For example, a text describing economics will not be relevant for a field like neuroscience.
            
    @apiName classify
    @apiGroup Classifier

    @apiParam {String} version Version that is used for classifying
    @apiBody {String[]} text The list of paragraphs of a text to classify.

    @apiSuccess {Boolean} relevant Relevant is True if there is any paragraph that contains any keyword relevant 
     for the given classifier task otherwise is False
    @apiSuccess {Number} result Result is the probability value (value between 0-1) of belonging to a class. 
    The closer to 0 the more likely the text is irrelevant, the closer to 1 the more likely the text is relevant.
    @apiSuccessExample {json} Success-Response:
                    { 
                        "relevant": TRUE,
                        "result": 0.988296
                    }
     @apiSuccessExample {json} Success-Response:
                    { 
                        "relevant": TRUE,
                        "result": 0.012
                    }
     @apiSuccessExample {json} Success-Response:
                    { "relevant": FALSE}
    @apiError (5xx) InternalServerError Any of the data is incorrect
    """
    def on_post(self, req, resp):
        try:
            self.log.debug('Calling classify')
            data = req.media
            version = req.params['version']
            result = self.classificationService.classify(version, data['text'])
            resp.body = json.dumps(result)

            resp.status = falcon.HTTP_200
        except Exception as e:
            self.log.exception('Error %s', e)
            resp.status = falcon.HTTP_500