import subprocess

from pyro_server_list import worker_servers

for computer in worker_servers[:10]:
#for computer in ['warsaw']:
    return_code = subprocess.call('ssh {0} "cd /home/ucfamue/DATA/stormtracks/ && /opt/anaconda/bin/python pyro_worker.py &" &'.format(computer), shell=True)  


