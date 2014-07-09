import subprocess

from pyro_settings import worker_servers, is_ucl

for computer in worker_servers[:10]:
    if is_ucl:
	return_code = subprocess.call('ssh {0} "cd /home/ucfamue/DATA/stormtracks/pyro_cluster && /opt/anaconda/bin/python pyro_worker.py &" &'.format(computer), shell=True)  
    else:
	return_code = subprocess.call('ssh {0} "cd /home/markmuetz/Projects/stormtracks/pyro_cluster && python pyro_worker.py &" &'.format(computer), shell=True)  


