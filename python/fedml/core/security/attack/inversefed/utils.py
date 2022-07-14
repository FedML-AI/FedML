"""Various utilities."""

import os
import csv

import torch
import random
import numpy as np

import socket
import datetime


def system_startup(args=None, defs=None):
    """Print useful system information."""
    # Choose GPU device and print status information:
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    setup = dict(device=device, dtype=torch.float)  # non_blocking=NON_BLOCKING
    print('Currently evaluating -------------------------------:')
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')
    if args is not None:
        print(args)
    if defs is not None:
        print(repr(defs))
    if torch.cuda.is_available():
        print(f'GPU : {torch.cuda.get_device_name(device=device)}')
    return setup

def save_to_table(out_dir, name, dryrun, **kwargs):
    """Save keys to .csv files. Function adapted from Micah."""
    # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, f'table_{name}.csv')
    fieldnames = list(kwargs.keys())

    # Read or write header
    try:
        with open(fname, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            header = [line for line in reader][0]
    except Exception as e:
        print('Creating a new .csv table...')
        with open(fname, 'w') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writeheader()
    if not dryrun:
        # Add row for this experiment
        with open(fname, 'a') as f:
            writer = csv.DictWriter(f, delimiter='\t', fieldnames=fieldnames)
            writer.writerow(kwargs)
        print('\nResults saved to ' + fname + '.')
    else:
        print(f'Would save results to {fname}.')
        print(f'Would save these keys: {fieldnames}.')

def set_random_seed(seed=233):
    """233 = 144 + 89 is my favorite number."""
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)

def set_deterministic():
    """Switch pytorch into a deterministic computation mode."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
