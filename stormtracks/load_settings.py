import os
import sys

# appends user's stormtracks dir to the path.
settings_dir = os.path.join(os.path.expandvars('$HOME'), '.stormtracks')
sys.path.append(settings_dir)

# Looks for a modified or a default settings file in ~/.stormtracks/
try:
    try:
        # First look for a user modified settings file.
        import stormtracks_settings as settings
    except ImportError, ie:
        # Else load default.
        import default_stormtracks_settings as settings
except ImportError, ie:
    print('Could not find settings file:\n{0}'.format(
        os.path.join(settings_dir, 'default_stormtracks_settings.py')))

try:
    try:
        import stormtracks_pyro_settings as pyro_settings
    except ImportError, ie:
        import default_stormtracks_pyro_settings as pyro_settings
except ImportError, ie:
    print('Could not find settings file:\n{0}'.format(
        os.path.join(settings_dir, 'default_stormtracks_pyro_settings.py')))
