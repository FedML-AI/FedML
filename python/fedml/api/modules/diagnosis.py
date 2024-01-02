import os

import click

from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.computing.scheduler.slave.client_diagnosis import ClientDiagnosis


def diagnose(open, s3, mqtt, mqtt_daemon, mqtt_s3_backend_server, mqtt_s3_backend_client,
             mqtt_s3_backend_run_id):
    check_open = open
    check_s3 = s3
    check_mqtt = mqtt
    check_mqtt_daemon = mqtt_daemon

    check_mqtt_s3_backend_server = mqtt_s3_backend_server
    check_mqtt_s3_backend_client = mqtt_s3_backend_client
    run_id = mqtt_s3_backend_run_id

    if open is None and s3 is None and mqtt is None:
        check_open = True
        check_s3 = True
        check_mqtt = True

    if mqtt_daemon is None:
        check_mqtt_daemon = False

    if mqtt_s3_backend_server is None:
        check_mqtt_s3_backend_server = False

    if mqtt_s3_backend_client is None:
        check_mqtt_s3_backend_client = False

    if check_open:
        is_open_connected = ClientDiagnosis.check_open_connection()
        if is_open_connected:
            click.echo("The connection to https://open.fedml.ai is OK.")
        else:
            click.echo("You can not connect to https://open.fedml.ai.")

    if check_s3:
        is_s3_connected = ClientDiagnosis.check_s3_connection()
        if is_s3_connected:
            click.echo("The connection to S3 Object Storage is OK.")
        else:
            click.echo("You can not connect to S3 Object Storage.")

    if check_mqtt:
        is_mqtt_connected = ClientDiagnosis.check_mqtt_connection()
        if is_mqtt_connected:
            click.echo("The connection to mqtt.fedml.ai (port:1883) is OK.")
        else:
            click.echo("You can not connect to mqtt.fedml.ai (port:1883).")

    if check_mqtt_daemon:
        ClientDiagnosis.check_mqtt_connection_with_daemon_mode()

    sys_utils.cleanup_all_fedml_client_diagnosis_processes()
    if check_mqtt_s3_backend_server:
        pip_source_dir = os.path.dirname(__file__)
        pip_source_dir = os.path.dirname(pip_source_dir)
        pip_source_dir = os.path.dirname(pip_source_dir)
        server_diagnosis_cmd = os.path.join(pip_source_dir, "computing", "scheduler", "slave", "client_diagnosis.py")
        backend_server_process = sys_utils.run_subprocess_open([
            sys_utils.get_python_program(),
            server_diagnosis_cmd,
            "-t",
            "server",
            "-r",
            run_id
        ]
        ).pid

    if check_mqtt_s3_backend_client:
        pip_source_dir = os.path.dirname(__file__)
        pip_source_dir = os.path.dirname(pip_source_dir)
        pip_source_dir = os.path.dirname(pip_source_dir)
        client_diagnosis_cmd = os.path.join(pip_source_dir, "computing", "scheduler", "slave", "client_diagnosis.py")
        backend_client_process = sys_utils.run_subprocess_open([
            sys_utils.get_python_program(),
            client_diagnosis_cmd,
            "-t",
            "client",
            "-r",
            run_id
        ]
        ).pid
