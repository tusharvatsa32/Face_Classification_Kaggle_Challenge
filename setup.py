import subprocess
import os,stat,sys
import shutil
from shutil import copyfile
from sys import exit


subprocess.check_call([sys.executable,'-m','pip','install','kaggle'])
subprocess.check_call([sys.executable,'-m','pip','install','natsort'])
subprocess.check_call([sys.executable,'-m','pip','install','pyyaml'])
subprocess.check_call([sys.executable,'-m','pip','install','torchvision'])


os.makedirs('~/Desktop/kaggle_competitions',exist_ok=True)
os.makedirs('./content',exist_ok=True)
os.makedirs('./content/.kaggle',exist_ok=True)


import json

token={"username":"richiet","key":"f356f91914539a3ee386ebdd612fc0de"}
with open('./content/.kaggle/kaggle.json','w') as file:
  json.dump(token,file)

path="./content/.kaggle/kaggle.json"
os.chmod(path,stat.S_IRWXU )

subprocess.call(['kaggle','config','set','-n','path','-v','./content'])
subprocess.check_call([sys.executable,'-m','pip','uninstall','-y','kaggle'])
subprocess.check_call([sys.executable,'-m','pip','install','--upgrade','pip'])
subprocess.check_call([sys.executable,'-m','pip','install','kaggle==1.5.6'])

subprocess.check_call([sys.executable,'-m','kaggle','competitions','download','-c','11785-spring2021-hw2p2s1-face-classification'])





