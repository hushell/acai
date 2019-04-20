# cross_interpolation
Python3.6 + Pytorch 1.0 + ipython + ujson + tqdm + argparse

``` Bash
virtualenv -p /usr/bin/python3.6 ../py36+pytorch1
source ../py36+pytorch1/bin/activate
bash requirement.sh
```

To use tensorboardX, need to install tensorflow first : 
``` Bash
pip3 install --upgrade tensorflow-gpu==1.4.1 # cuda 8.0
```

Install tensorboardX
``` Bash
pip3 install tensorboardX
```

To visualize and remote visualize : 
``` Bash
tensorboard --logdir=./log
ssh -L LOCAL_PORT:IP:REMOTE_PORT USER_NAME@SERVER_IP
```

