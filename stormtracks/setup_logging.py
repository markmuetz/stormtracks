import os
import logging

from stormtracks.load_settings import settings


def get_logger(name):
    '''Gets a logger specified by name. Sets up root logger ('st') if nec.'''
    # Get root stormtracks logger and check if it's already been setup.
    root_logger = logging.getLogger('st')

    if getattr(root_logger, 'is_setup', False):
        # Stops log being setup for a 2nd time during ipython reload(...)
	root_logger.debug('Root logger already setup')
    else:
	if not os.path.exists(settings.LOGGING_DIR):
	    os.makedirs(settings.LOGGING_DIR)

        console_level = getattr(settings, 'CONSOLE_LOG_LEVEL', 'INFO').upper()
        file_level = getattr(settings, 'FILE_LOG_LEVEL', 'DEBUG').upper()

	formatter = logging.Formatter('%(asctime)s:%(name)-12s:%(levelname)-8s: %(message)s')

	logging_filename = os.path.join(settings.LOGGING_DIR, 'stormtracks.log')
	fileHandler = logging.FileHandler(logging_filename, mode='a')
	fileHandler.setFormatter(formatter)
        fileHandler.setLevel(file_level)

	streamHandler = logging.StreamHandler()
	streamHandler.setFormatter(formatter)
	streamHandler.setLevel(console_level)

	root_logger.setLevel(min(console_level, file_level))

	root_logger.addHandler(fileHandler)
	root_logger.addHandler(streamHandler)

	root_logger.debug('Created root logger: {0}'.format('stormtracks.log'))

        root_logger.is_setup = True

    if name == 'st':
	return root_logger
    else:
	logger = logging.getLogger(name)
	return logger
