import logging
from time import time
def log_communication_tick(sender, receiver, timestamp=None):
    logging.info('--Benchmark tick from {} to {} at {}'.format(sender,receiver,timestamp or time()))

def log_communication_tock(sender, receiver, timestamp=None):
    logging.info('--Benchmark tock from {} to {} at {}'.format(sender,receiver,timestamp or time()))