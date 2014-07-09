import subprocess

from pyro_settings import worker_servers, is_ucl

for computer in worker_servers[:10]:
    if is_ucl:
	return_code = subprocess.call('ssh {0} "bash DATA/stormtracks/bin/kill_pyro.sh"'.format(computer), shell=True)  
    else:
	return_code = subprocess.call('ssh {0} "bash Projects/stormtracks/bin/kill_pyro.sh"'.format(computer), shell=True)  

