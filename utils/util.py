import os
import json
from pathlib import Path
from collections import OrderedDict

import tensorflow as tf


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def check_gpu(devices, memory_limit=True): 
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = devices

    gpus = tf.config.list_physical_devices('GPU')
    gpu_n = len(gpus)
    print(gpu_n, " GPU found!")
    
    if gpu_n == 0:
        return 1

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, memory_limit)

    return gpu_n
