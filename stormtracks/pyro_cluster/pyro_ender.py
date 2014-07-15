#!/usr/bin/python
import subprocess

from stormtracks.load_settings import pyro_settings


def main():
    for computer in pyro_settings.worker_servers:
        cmd = pyro_settings.ssh_kill_cmd_tpl.format(computer)
        print('Executing command:{0}'.format(cmd))
        return_code = subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    main()
