'''Looks for a python file `stormtracks_settings.py` in the current dir'''
import os
import sys

# appends user's current dir to the path.
sys.path.append('.')

# Used to use a special dotdir:
# settings_dir = os.path.join(os.path.expandvars('$HOME'), '.stormtracks')

# Looks for a modified or a default settings file in '.'.
try:
    import stormtracks_settings as settings
except ImportError, ie:
    print('Could not find settings file:\n{0}'.format(
        os.path.join('.', 'stormtracks_settings.py')))
