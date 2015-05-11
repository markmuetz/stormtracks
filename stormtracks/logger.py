import os
import logging

from stormtracks.load_settings import settings


def setup_logging(name, filename=None, level_str='INFO', console_level_str='WARNING'):
    log = logging.getLogger(name)
    level = getattr(logging, level_str)
    console_level = getattr(logging, console_level_str)

    if getattr(log, 'is_setup', False):
        # Stops log being setup for a 2nd time during ipython reload(...)
        log.debug('Already setup')
        return log
    else:
        log.is_setup = True

    if name == 'status':
        formatter = logging.Formatter('%(message)s')
    else:
        formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

    if not os.path.exists(settings.LOGGING_DIR):
        os.makedirs(settings.LOGGING_DIR)

    logging_filename = os.path.join(settings.LOGGING_DIR, '{0}'.format(filename))
    fileHandler = logging.FileHandler(logging_filename, mode='a')
    fileHandler.setFormatter(formatter)

    streamFormatter = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(streamFormatter)
    streamHandler.setLevel(console_level)

    log.setLevel(level)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)

    log.debug('Created logger {0}: {1}'.format(name, filename))

    return log


def get_logger(name, console_logging=True, console_level_str='WARNING'):
    return logging.getLogger(name)

# TODO: remove
# def setup_logging(name, filename=None, level_str='DEBUG', console_level_str='WARNING'):
#     if not os.path.exists(settings.LOGGING_DIR):
#         os.makedirs(settings.LOGGING_DIR)
#
#     level = getattr(logging, level_str)
#
#     # N.B. .log gets added on automatically.
#     logging_filename = os.path.join(settings.LOGGING_DIR, '{0}'.format(filename))
#
#     if name == 'status':
#         logging.basicConfig(filename=logging_filename,
#                             format='%(message)s',
#                             datefmt='%m-%d %H:%M',
#                             level=level)
#     else:
#         logging.basicConfig(filename=logging_filename,
#                             format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
#                             datefmt='%m-%d %H:%M.%S',
#                             level=level)
#
#
#     return get_logger(name, console_level_str)


# def get_logger(name, console_logging=True, console_level_str='WARNING'):
#     logger = logging.getLogger(name)
#
#     if console_logging:
#         console_level = getattr(logging, console_level_str)
#         console_print = logging.StreamHandler(sys.stdout)
#         console_print.setLevel(console_level)
#         formatter = logging.Formatter('%(message)s')
#         console_print.setFormatter(formatter)
#         logger.addHandler(console_print)
#
#     return logger
