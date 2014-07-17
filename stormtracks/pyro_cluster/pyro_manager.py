#!/usr/bin/python
import socket
import time
import copy

import Pyro4

from stormtracks.load_settings import pyro_settings
from stormtracks.pyro_task import PyroTaskSchedule


def main():
    print('Calling from {0}'.format(socket.gethostname()))
    year = 2005
    schedule = PyroTaskSchedule(year, year)
    asyncs = []

    workers = {}
    free_workers = copy.copy(pyro_settings.worker_servers)

    for server_name in free_workers:
        print('Adding server {0}'.format(server_name))

        worker_proxy = Pyro4.Proxy('PYRONAME:stormtracks.worker_{0}'.format(server_name))
        async_worker_proxy = Pyro4.async(worker_proxy)
        workers[server_name] = (worker_proxy, async_worker_proxy)

    sleep_count = 0
    orig_len_free_workers = len(free_workers)
    all_tasks_complete = False
    task = schedule.get_next_outstanding()

    while not all_tasks_complete:
        while free_workers:
            server_name = free_workers.pop()
            print('Requesting work from {0} year {1} ensemble {2}'.format(
                server_naem, task.year, task.ensemble_member))

            worker_proxy, async_worker_proxy = workers[server_name]

            async_response = async_worker_proxy.do_work(task)
            async_response.server_name = server_name
            asyncs.append(async_response)

            task.status = 'working'

            task = schedule.get_next_outstanding()

        print('Sleeping {0}'.format(sleep_count))
        sleep_count += 1
        time.sleep(1)

        for async_response in asyncs:
            if async_response.ready:
                response_task = async_response.value
                print('{0:8s}: {1}'.format(async_response.server_name, response_task.status))
                if response_task.status == 'complete':
                    free_workers.append(async_response.server_name)
                    schedule.update_task_status(response_task)
                    asyncs.remove(async_response)
                    schedule.print_years([year])
                elif response_task.status == 'failure':
                    task = response_task
                    task.status = 'outstanding'
                else:
                    raise Exception(task.status)

            else:
                print('{0:8s}: Not ready'.format(async_response.server_name))

        if not task and len(free_workers) == orig_len_free_workers:
            all_tasks_complete = True


if __name__ == '__main__':
    main()
