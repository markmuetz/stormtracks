import subprocess

def setup_data(settings):
    subprocess.call('compile_cyclone.sh', shell=True)
