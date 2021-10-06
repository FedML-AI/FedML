import logging
from time import time
def log_communication_tick(sender, receiver):
    logging.info('--Benchmark tick from {} to {} at {}'.format(sender,receiver,time()))

def log_communication_tock(sender, receiver):
    logging.info('--Benchmark tock from {} to {} at {}'.format(sender,receiver,time()))