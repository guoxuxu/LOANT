import os
import argparse
import torch
import numpy
import random
from datetime import datetime


def format_time():
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m-%d-%H:%M:%S")
    return date_time


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seed(seed):
    # torch.backends.cudnn.deterministic = True  ## this one is controversial
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

