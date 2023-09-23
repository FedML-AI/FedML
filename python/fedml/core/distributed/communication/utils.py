import logging
from time import time


def log_communication_tick(sender, receiver, timestamp=None):
    """
    Log a benchmark tick event from sender to receiver.

    Args:
        sender (str): Sender's identifier.
        receiver (str): Receiver's identifier.
        timestamp (float): Timestamp for the event (default is current time).
    """
    logging.info(
        "--Benchmark tick from {} to {} at {}".format(
            sender, receiver, timestamp or time()
        )
    )


def log_communication_tock(sender, receiver, timestamp=None):
    """
    Log a benchmark tock event from sender to receiver.

    Args:
        sender (str): Sender's identifier.
        receiver (str): Receiver's identifier.
        timestamp (float): Timestamp for the event (default is current time).
    """
    logging.info(
        "--Benchmark tock from {} to {} at {}".format(
            sender, receiver, timestamp or time()
        )
    )


def log_round_start(client_idx, round_number, timestamp=None):
    """
    Log the start of a benchmark round for a client.

    Args:
        client_idx (int): Client's index or identifier.
        round_number (int): Round number.
        timestamp (float): Timestamp for the event (default is current time).
    """
    logging.info(
        "--Benchmark start round {} for {} at {}".format(
            round_number, client_idx, timestamp or time()
        )
    )


def log_round_end(client_idx, round_number, timestamp=None):
    """
    Log the end of a benchmark round for a client.

    Args:
        client_idx (int): Client's index or identifier.
        round_number (int): Round number.
        timestamp (float): Timestamp for the event (default is current time).
    """
    logging.info(
        "--Benchmark end round {} for {} at {}".format(
            round_number, client_idx, timestamp or time()
        )
    )
