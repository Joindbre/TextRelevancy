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

from classifier.trainService import TrainService

import falcon
from logger import set_up_logging


class TrainController(object):
    trainService = TrainService()
    log = set_up_logging()

    """
        @apiVersion 1.0.0
        @api {put} /classifier/train/:version Pre-process
        @apiDescription Extracts the relevant paragraphs from all the texts stored at  
        Saves the resulted relevant paragraphs that are used for training the neural network in files, this step reduces
         the noise and improves the performance cutting down the input text by selecting only paragraphs relevant for the
          task
           
    Text instances containing the full text paragraphs are read from /resources/data/{version}/{Positive|Negative}_{i}.json
    Files with relevant paragraphs are created at /resources/data/{version}/{Positive|Negative}_Relevant_{i}.npz 
        @apiGroup Classifier
        @apiParam {String} version Version that is used for classifying

        @apiSuccess (200) OK
        @apiError (5xx) InternalServerError Any of the data is incorrect

       """

    def on_put(self, req, resp):
        try:
            version = req.params['version']
            self.log.debug('Extracting relevant text for version %s', version)
            self.trainService.extract_relevant(version)
            resp.status = falcon.HTTP_200
        except Exception as e:
            self.log.exception('Error %s', e)
            resp.status = falcon.HTTP_500
    # Train
    """
        @apiVersion 1.0.0
        @api {post} /classifier/train/:version Train
        @apiDescription Reads all the text instances that should be stored in files, vectorizes the texts and trains the classifier.
                 Two new files are created as a result of the classification: the tokenizer containing the vocabulary and 
                 the classification model. Both files will be used in the classification step to vectorize new data 
                 and predict the relevancy. The output returns data in json format containing the accuracy and loss 
                 values for the train and test sets, the epochs, the number of train and test samples, 
                 the version used when classifying (several versions can be maintained and used), and the classification 
                 result for each of the test samples [true value, classification value]

        Text instances used to train are read from /resources/data/{version}/{Positive|Negative}_Relevant_{i}.npz 
        Two files will be created as the result of classification:
            - The vectorization tokenizer /resources/classifier/model/{version}/tokenizer.pickle
            - The classification model /resources/classifier/model/{version}/model.h5 
        
        @apiName train
        @apiGroup Classifier

        @apiParam {String} version Version that is used for classifying
        @apiSuccess {Number[]} trainAccuracy Train Accuracy values for each epoch 
        @apiSuccess {Number[]} testLoss Test Loss values for each epoch 
        @apiSuccess {Number} epochs Number of epochs        
        @apiSuccess {Number} trainSamples Number of Samples used for training        
        @apiSuccess {Number} testAccuracy Test Accuracy        
        @apiSuccess {Number} testLoss Test Loss      
        @apiSuccess {Number} testSamples Number of Samples used for testing    
        @apiSuccess {Number[][]} classList Test samples result: labeled value and classifier result for each of the test samples
        @apiSuccess {String} version Version that is used for classifying    

        @apiSuccessExample {json} Success-Response:
            {
                "trainAccuracy": [
                            0.8099652528762817,
                            0.9126303791999817,
                            ...
                            0.9974507689476013
                            ],
                "trainLoss": [
                            0.15503212809562683,
                            0.07761093974113464,
                            ...
                            0.002978024771437049
                            ],
                "epochs": 30,
                "trainSamples": 3884,
                "testAccuracy": 0.9144049882888794,
                "testLoss": 0.07054183632135391,
                "testSamples": 479,
                "classList": [
                    [
                        1,
                        0.7641094923019409
                    ],
                    [
                        0,
                        0.00008273577259387821
                    ],
                    ...
                    [
                        0,
                        0.12123090028762817
                    ]
                ],
                    "version": "v3"
                }
                    
        @apiError (5xx) InternalServerError Any of the data is incorrect
       """

    def on_post(self, req, resp):
        try:
            version = req.params['version']
            self.log.debug('Training classifier for version %s', version)

            result = self.trainService.train(version)
            self.log.debug(result)
            result.update({'version': version})
            resp.body = json.dumps(result)
            resp.status = falcon.HTTP_200
        except Exception as e:
            self.log.exception('Error %s', e)
            resp.status = falcon.HTTP_500
