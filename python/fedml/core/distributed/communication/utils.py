import logging
from time import time


def log_communication_tick(sender, receiver, timestamp=None):
    logging.info(
        "--Benchmark tick from {} to {} at {}".format(
            sender, receiver, timestamp or time()
        )
    )


def log_communication_tock(sender, receiver, timestamp=None):
    logging.info(
        "--Benchmark tock from {} to {} at {}".format(
            sender, receiver, timestamp or time()
        )
    )


def log_round_start(client_idx, round_number, timestamp=None):
    logging.info(
        "--Benchmark start round {} for {} at {}".format(
            round_number, client_idx, timestamp or time()
        )
    )


def log_round_end(client_idx, round_number, timestamp=None):
    logging.info(
        "--Benchmark end round {} for {} at {}".format(
            round_number, client_idx, timestamp or time()
        )
    )
