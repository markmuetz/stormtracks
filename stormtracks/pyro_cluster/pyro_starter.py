import subprocess

from stormtracks.load_settings import pyro_settings

for computer in pyro_settings.worker_servers:
    cmd = pyro_settings.ssh_start_cmd_tpl.format(computer)
    print('Executing command:{0}'.format(cmd))
    return_code = subprocess.call(cmd, shell=True)  
