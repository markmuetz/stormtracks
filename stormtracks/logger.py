import os
import sys
import logging

from stormtracks.load_settings import settings, pyro_settings


class Logger(object):
    def __init__(self, name, filename, level_str='DEBUG'):
        if not os.path.exists(settings.LOGGING_DIR):
            os.makedirs(settings.LOGGING_DIR)

        level = getattr(logging, level_str)

        # N.B. .log gets added on automatically.
        logging_filename = os.path.join(settings.LOGGING_DIR, '{0}'.format(filename))

        logging.basicConfig(filename=logging_filename,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            level=level)

        self.__logger = logging.getLogger(name)

        console_print = logging.StreamHandler(sys.stdout)
        console_print.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_print.setFormatter(formatter)
        self.__logger.addHandler(console_print)

    def get(self):
        return self.__logger
