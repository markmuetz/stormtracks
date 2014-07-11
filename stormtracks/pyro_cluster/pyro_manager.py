#!/usr/bin/python
import socket
import time

import Pyro4

from stormtracks.load_settings import pyro_settings

def main():
    print('Calling from {0}'.format(socket.gethostname()))
    year = 2005
    asyncs = []

    for i, server_name in enumerate(pyro_settings.worker_servers):
	ensemble_member = i + 20
	worker = Pyro4.Proxy('PYRONAME:stormtracks.worker_{0}'.format(server_name))          # get a Pyro proxy to the greeting object
	async_worker = Pyro4.async(worker)
	print('Requesting matches from year {0} ensemble {1}'.format(year, ensemble_member))
	async = async_worker.do_work(year, ensemble_member)
	async.server_name = server_name

	asyncs.append(async)

    finished = False
    sleep_count = 0
    while not finished:
	print('Sleeping {0}'.format(sleep_count))
	sleep_count += 1
	time.sleep(1)
	finished = True
	for async in asyncs:
	    if async.ready:
		print('{0:8s}: {1}'.format(async.server_name, async.value))
	    else:
		print('{0:8s}: Not ready'.format(async.server_name))
		finished = False

if __name__ == '__main__':
    main()
