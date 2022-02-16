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

import logging
import logging.handlers
from datetime import datetime
import os
import sys

# Logging Levels
# https://docs.python.org/3/library/logging.html#logging-levels
# CRITICAL  50
# ERROR 40
# WARNING   30
# INFO  20
# DEBUG 10
# NOTSET    0


def set_up_logging():
    file_path = sys.modules[__name__].__file__
    project_path = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
    log_location = project_path + '/logs/'
    if not os.path.exists(log_location):
        os.makedirs(log_location)

    current_time = datetime.now()
    current_date = current_time.strftime("%Y-%m-%d")
    file_name = 'Classifier-' + current_date + '.log'
    file_location = log_location + file_name
    with open(file_location, 'a+'):
        pass

    logger = logging.getLogger(__name__)
    format = '%(asctime)s - %(levelname)s :- %(message)s'
    # To store in file
    logging.basicConfig(format=format, filemode='a+', filename=file_location, level=logging.DEBUG)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    # To print only
    # logging.basicConfig(format=format, level=logging.DEBUG)
    return logger