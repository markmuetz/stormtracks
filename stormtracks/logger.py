import os
import logging

from stormtracks.load_settings import settings, pyro_settings


class Logger(object):
    def __init__(self, name, filename):
        if not os.path.exists(settings.LOGGING_DIR):
            os.makedirs(settings.LOGGING_DIR)

        logging_filename = os.join(settings.LOGGING_DIR, '{0}.log'.format(filename))

        logging.basicConfig(filename=logging_filename,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            level=logging.DEBUG)

        self.__logger = logging.getLogger(name)

    def get(self):
        return self.__logger
