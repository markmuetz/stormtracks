#!/usr/bin/python
import Pyro4

from stormtracks.load_settings import pyro_settings


def main():
    '''Launches a Pyro4 nameserver'''
    Pyro4.naming.main(['-n', pyro_settings.nameserver])


if __name__ == '__main__':
    main()
