#!/usr/bin/python
from __future__ import print_function
import socket
import time
import copy

import Pyro4

from stormtracks.load_settings import pyro_settings
from stormtracks.pyro_cluster.pyro_task import PyroTaskSchedule


def main():
    start = time.time()

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
        if task:
            while free_workers:
                server_name = free_workers.pop()
                print('Requesting work from {0} year {1} ensemble {2}'.format(
                    server_name, task.year, task.ensemble_member))

                worker_proxy, async_worker_proxy = workers[server_name]

                async_response = async_worker_proxy.do_work(task.year,
                                                            task.ensemble_member,
                                                            task.task)
                async_response.server_name = server_name
                async_response.task = task
                asyncs.append(async_response)

                task.status = 'working'

                task = schedule.get_next_outstanding()

                if not task:
                    print('All tasks now being worked on')

        print('Sleep {0:4d}: '.format(sleep_count), end='')
        schedule.print_years([year])

        sleep_count += 1
        time.sleep(1)

        for async_response in asyncs:
            if async_response.ready:
                response = async_response.value

                print('{0:8s}: {1}'.format(async_response.server_name, response['status']))
                if response['status'] == 'complete':
                    # schedule.update_task_status(response_task)
                    async_response.task.status = response['status']
                    asyncs.remove(async_response)
                    schedule.print_years([year])

                    free_workers.append(async_response.server_name)
                elif response['status'] == 'failure':
                    print(response['exception'])
                    task = async_response.task
                    task.status = 'outstanding'

                    free_workers.append(async_response.server_name)
                else:
                    raise Exception(response['status'])

            else:
                # print('{0:8s}: Not ready'.format(async_response.server_name))
                pass

        if not task and len(free_workers) == orig_len_free_workers:
            all_tasks_complete = True

    end = time.time()
    tasks_completed = (1 + schedule.end_year - schedule.start_year) * schedule.num_ensemble_members

    print('Completed {0} tasks in {1}s'.format(tasks_completed, end - start))


if __name__ == '__main__':
    main()
