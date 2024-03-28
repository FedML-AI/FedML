import argparse
import logging
import os
import shutil

import pytest

LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs')

from fedml.core.mlops import MLOpsRuntimeLog


@pytest.fixture
def setup():
    print("\nSetup Logs Directory")
    if os.path.exists(LOGS_DIR):
        shutil.rmtree(LOGS_DIR)
    os.makedirs(LOGS_DIR, exist_ok=True)
    yield
    shutil.rmtree(LOGS_DIR)
    print("\nTeardown Logs Directory")


@pytest.fixture
def args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = parser.parse_args([])
    setattr(args, "log_file_dir", LOGS_DIR)
    setattr(args, "run_id", "10")
    setattr(args, "edge_id", "11")
    setattr(args, "role", "client")
    setattr(args, "config_version", "local")
    setattr(args, "using_mlops", True)
    return args


def test_logs_rollover(setup, args):
    print(hasattr(args, "using_mlops"))
    MLOpsRuntimeLog.get_instance(args).init_logs()
    logging.info("Test Log Message")
    print("Log File Path: ", MLOpsRuntimeLog.get_instance(args).origin_log_file_path)
