import time
import sys
import os

# args = sys.argv
path = '/nas-homes/krlabmember/hayakawa/binary/20210115'

while True:
    os.chmod(path, 0o755)
    time.sleep(10)