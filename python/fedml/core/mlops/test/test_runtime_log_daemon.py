import argparse
import logging
import os

import time

import pytest

LOGS_DIR = os.path.join(os.path.dirname(__file__), 'logs')
from fedml.core.mlops import MLOpsRuntimeLogDaemon


@pytest.fixture
def mock_requests_post(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200

    # Mock requests.post() method
    mocker.patch('requests.post', return_value=mock_response)


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


def test_logs_daemon_upload(args, mock_requests_post):
    MLOpsRuntimeLogDaemon.get_instance(args).start_log_processor(args.run_id, args.edge_id)
    logging.info("Test Log Message")
    while True:
        time.sleep(1)
    # print("Log File Path: ", MLOpsRuntimeLog.get_instance(args).origin_log_file_path)
