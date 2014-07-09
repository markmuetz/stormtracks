import subprocess

from pyro_server_list import worker_servers

for computer in worker_servers[:10]:
#for computer in ['warsaw']:
    return_code = subprocess.call('ssh {0} "bash DATA/stormtracks/kill_pyro.sh"'.format(computer), shell=True)  

