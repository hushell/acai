# ACAI (1807.07543) Pytorch Implementation based on [kylemcdonald's gist](https://gist.github.com/kylemcdonald/e8ca989584b3b0e6526c0a737ed412f0)
Python3.6 + Pytorch 1.0

Setup:
``` Bash
virtualenv -p /usr/bin/python3.6 ./py36+pytorch1
source ./py36+pytorch1/bin/activate
pip install https://download.pytorch.org/whl/cu80/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
pip install -r requirement.txt
```

To visualize and remote visualize : 
``` Bash
tensorboard --logdir=./log
ssh -L LOCAL_PORT:IP:REMOTE_PORT USER_NAME@SERVER_IP
```

