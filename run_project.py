#!/usr/bin/python
from __future__ import print_function

from argparse import ArgumentParser

from get_data import get_data
from setup_data import setup_data

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-s', '--settings', help='Settings file')
    parser.add_argument('-r', '--remove', help='Remove', action='store_true')
    args = parser.parse_args()

    if not args.settings:
	import settings.default as settings
    else:
	raise Exception('not impl')

    if settings.GET_DATA:
	get_data(settings)
    if settings.SETUP_DATA:
	setup_data(settings)


