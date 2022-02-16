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

import falcon

from classifier.classifyController import ClassifyController
from classifier.trainController import TrainController
from nlp.searchController import SearchController

api = application = falcon.API()

classify = ClassifyController()
train = TrainController()
nlp = SearchController()

api.add_route('/classifier/classify', classify)
api.add_route('/classifier/train', train)
api.add_route('/nlp', nlp)


#Activate environment  source venv/bin/activate
#Launch ->  gunicorn -b 0.0.0.0:8191 --reload app --timeout 12000
