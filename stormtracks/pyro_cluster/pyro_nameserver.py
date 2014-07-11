#!/usr/bin/python
import Pyro4

from stormtracks.load_settings import pyro_settings

def main():
    Pyro4.naming.main(['-n', pyro_settings.nameserver])

if __name__ == '__main__':
    main()
