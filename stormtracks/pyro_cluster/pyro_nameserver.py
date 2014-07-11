import Pyro4

from stormtracks.load_settings import pyro_settings

Pyro4.naming.main(['-n', pyro_settings.nameserver])
