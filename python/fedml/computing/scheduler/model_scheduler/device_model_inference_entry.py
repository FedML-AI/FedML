import argparse
import os
import time

from fedml.computing.scheduler.model_scheduler.device_client_constants import ClientConstants
from fedml.computing.scheduler.model_scheduler.device_server_constants import ServerConstants
from fedml.computing.scheduler.comm_utils.sys_utils import get_python_program


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--redis_addr", "-rp", type=str, default="local", help="redis address")
    parser.add_argument("--redis_port", "-ra", type=int, default=6379, help="redis port")
    parser.add_argument("--redis_password", "-rpw", type=str, default="fedml_default")
    parser.add_argument("--end_point_id", "-ep", type=str, help="end point id")
    parser.add_argument("--end_point_name", "-epn", type=str, help="end point name")
    parser.add_argument("--model_id", "-mi", type=str, help="model id")
    parser.add_argument("--model_name", "-mn", type=str, help="model name")
    parser.add_argument("--model_version", "-mv", type=str, help="model version")
    parser.add_argument("--infer_url", "-iu", type=str, help="inference url")
    parser.add_argument("--config_version", "-cv", type=str, help="config version")
    parser.add_argument("--infer_port", "-ip", type=int,
                        default=ServerConstants.MODEL_INFERENCE_DEFAULT_PORT, help="inference port")
    args = parser.parse_args()

    # create directories
    if not os.path.exists(ClientConstants.get_model_dir()):
        os.makedirs(ClientConstants.get_model_dir(), exist_ok=True)
    if not os.path.exists(ClientConstants.get_model_package_dir()):
        os.makedirs(ClientConstants.get_model_package_dir(), exist_ok=True)
    if not os.path.exists(ClientConstants.get_model_serving_dir()):
        os.makedirs(ClientConstants.get_model_serving_dir(), exist_ok=True)

    # start unified inference server
    running_model_name = ClientConstants.get_running_model_name(args.end_point_name,
                                                                args.model_name, args.model_version,
                                                                args.end_point_id, args.model_id)
    python_program = get_python_program()
    process = ServerConstants.exec_console_with_script(
        "REDIS_ADDR=\"{}\" REDIS_PORT=\"{}\" REDIS_PASSWORD=\"{}\" "
        "END_POINT_NAME=\"{}\" "
        "MODEL_NAME=\"{}\" MODEL_VERSION=\"{}\" MODEL_INFER_URL=\"{}\" VERSION=\"{}\" "
        "{} -m uvicorn fedml.computing.scheduler.model_scheduler.device_model_inference:api --host 0.0.0.0 --port {} "
        "--log-level critical".format(
            args.redis_addr, args.redis_port, args.redis_password,
            args.end_point_name,
            args.model_name, args.model_version, args.infer_url, args.config_version,
            python_program, str(args.infer_port)),
        should_capture_stdout=False,
        should_capture_stderr=False
    )

    while True:
        time.sleep(3)

