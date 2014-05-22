import subprocess
import os

from jinja2 import Template, FileSystemLoader

def run():
    os.chdir('bin')
    subprocess.call('./cyclone', shell=True)

if __name__ == "__main__":
    run()


