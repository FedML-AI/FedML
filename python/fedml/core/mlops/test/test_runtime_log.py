import argparse
import logging
import os
import shutil
import time

import pytest

LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs')

from fedml.core.mlops import MLOpsRuntimeLog, MLOpsRuntimeLogDaemon


@pytest.fixture
def mock_requests_post(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200

    # Mock requests.post() method
    mocker.patch('requests.post', return_value=mock_response)


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
    setattr(args, "log_server_url", "http://localhost:8080")
    return args


def test_logs_rollover(setup, args, mock_requests_post):
    MLOpsRuntimeLog.get_instance(args).init_logs()
    MLOpsRuntimeLogDaemon.get_instance(args).start_log_processor(args.run_id, args.edge_id)
    # timeout variable can be omitted, if you use specific value in the while condition
    timeout = 300  # [seconds]

    timeout_start = time.time()
    index = 0

    while time.time() < timeout_start + timeout:
        logging.info(f"Test Log Message: {index}")
        time.sleep(1)
        index += 1

    print("Log File Path: ", MLOpsRuntimeLog.get_instance(args).origin_log_file_path)
