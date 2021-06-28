import os
import socket
from datetime import datetime
import subprocess as sp


def format_time():
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m-%d-%H:%M:%S")
    return date_time


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_dir(path):
    if not os.path.exists(path):
        return True
    else:
        return False


def nvidia_free_mem(cuda):
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    free_mem = memory_free_values[cuda]
    return free_mem


def nvidia_gpu_mem(init_free_gpu, cuda):
    memory_free_values = nvidia_free_mem(cuda)
    mem_occupied = init_free_gpu - memory_free_values
    return mem_occupied


class FileLogger(object):
    """
    A file logger that opens the file periodically and write to it.
    """

    def __init__(self, path:str, filename: str, seed:int, header: list):
        self.filename = os.path.join(path, filename)
        # remove the old file
        if os.path.exists(self.filename):
            os.remove(self.filename)
        # or create the file
        if not os.path.exists(self.filename):
            if not os.path.exists(path):
                os.makedirs(path)
        if len(header) != 0:
            header = ', '.join(header)
            with open(self.filename, 'w') as out:
                print('Info: seed=' + str(seed) + ', Env=' + socket.gethostname() +', Time=' + format_time(), file=out)
                print('Header: ' + header, file=out)

    def log(self, message):
        with open(self.filename, 'a') as out:
            print(message, file=out)

