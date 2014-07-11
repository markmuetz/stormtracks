import Pyro4

from stormtracks.load_settings.pyro_settings import is_ucl

if is_ucl:
    Pyro4.naming.main(['-n', 'madrid'])
else:
    Pyro4.naming.main(['-n', '192.168.0.15'])
