import json
import random

import numpy as np
import paho.mqtt.client as mqtt_client
import requests
import torch

from .stackoverflow_lr.data_loader import load_partition_data_federated_stackoverflow_lr
from .FederatedEMNIST.data_loader import load_partition_data_federated_emnist
from .ImageNet.data_loader import load_partition_data_ImageNet
from .Landmarks.data_loader import load_partition_data_landmarks
from .MNIST.data_loader import load_partition_data_mnist, download_mnist
from .cifar10.data_loader import load_partition_data_cifar10
from .cifar10.efficient_loader import efficient_load_partition_data_cifar10
from .cifar100.data_loader import load_partition_data_cifar100
from .cinic10.data_loader import load_partition_data_cinic10
from .edge_case_examples.data_loader import load_poisoned_dataset
from .fed_cifar100.data_loader import load_partition_data_federated_cifar100
from .fed_shakespeare.data_loader import load_partition_data_federated_shakespeare
from .file_operation import *
from .shakespeare.data_loader import load_partition_data_shakespeare
from .stackoverflow_nwp.data_loader import load_partition_data_federated_stackoverflow_nwp
from ..core.mlops import MLOpsConfigs


import boto3
from botocore.config import Config


def connect_mqtt(mqtt_config) -> mqtt_client:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Host!")
        else:
            print("Failed to connect, return code %d\n", rc)

    # generate client ID with pub prefix randomly
    client_id = f"python-mqtt-{random.randint(0, 1000)}"
    client = mqtt_client.Client(client_id, clean_session=False)
    client.username_pw_set(mqtt_config["MQTT_USER"], mqtt_config["MQTT_PWD"])
    client.connect(mqtt_config["BROKER_HOST"], mqtt_config["BROKER_PORT"])
    return client


def subscribe(s3_obj, BUCKET_NAME, client: mqtt_client, args):
    def on_message(client, userdata, msg):
        logging.info(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
        if msg.payload.decode():
            disconnect(client)
            make_dir(
                os.path.join(
                    args.data_cache_dir,
                    "run_Id_%s" % args.run_id,
                    "edgeNums_%s" % (args.client_num_in_total),
                    args.dataset,
                    "edgeId_%s" % args.client_id,
                )
            )
            # start download the file
            download_s3_file(
                s3_obj,
                BUCKET_NAME,
                json.loads(msg.payload.decode())["edge_id"],
                json.loads(msg.payload.decode())["dataset"],
                os.path.join(
                    args.data_cache_dir,
                    "run_Id_%s" % args.run_id,
                    "edgeNums_%s" % (args.client_num_in_total),
                    args.dataset,
                    "edgeId_%s" % args.client_id,
                ),
                os.path.join(
                    args.data_cache_dir,
                    "run_Id_%s" % args.run_id,
                    "edgeNums_%s" % (args.client_num_in_total),
                    args.dataset,
                    "edgeId_%s" % args.client_id,
                    "cifar-10-python.tar.gz",
                ),
            )

    topic = "data_svr/dataset/%s" % args.client_id
    client.subscribe(topic)
    client.on_message = on_message


def disconnect(client: mqtt_client):
    client.disconnect()
    logging.info(f"Received message, Mqtt stop listen.")


def setup_s3_service(s3_config):
    _config = Config(
        retries={
            'max_attempts': 4,
            'mode': 'standard'
        }
    )
    # s3 client
    s3 = boto3.client('s3', region_name=s3_config["CN_REGION_NAME"], aws_access_key_id=s3_config["CN_S3_AKI"],
                      aws_secret_access_key=s3_config["CN_S3_SAK"], config=_config)
    BUCKET_NAME = s3_config["BUCKET_NAME"]
    return s3, BUCKET_NAME


def data_server_preprocess(args):
    mqtt_config, s3_config = MLOpsConfigs.get_instance(args).fetch_configs()
    s3_obj, BUCKET_NAME = setup_s3_service(s3_config)

    args.private_local_data = ""
    if args.process_id == 0:
        pass
    else:
        client = connect_mqtt(mqtt_config)
        subscribe(s3_obj, BUCKET_NAME, client, args)
        if args.dataset == "cifar10":
            # Mlops Run
            # check mlops run_status
            private_local_dir, split_status, edgeids, dataset_s3_key = check_rundata(args)
            args.private_local_data = private_local_dir
            # MLOPS Run. User supply the local data dir
            if len(args.private_local_data) != 0:
                logging.info("User has set the private local data dir")
                disconnect(client)
            # MLOPS Run need to Split Data
            elif len(args.synthetic_data_url) != 0:
                if split_status == 0 or split_status == 3:
                    logging.info("Data Server Start Splitting Dataset")
                    split_edge_data(args, edgeids)
                elif split_status == 1:
                    logging.info("Data Server Is Splitting Dataset, Waiting For Mqtt Message")
                elif split_status == 2:
                    logging.info("Data Server Splitted Dataset Complete")
                    query_data_server(args, args.client_id, s3_obj, BUCKET_NAME)
                    disconnect(client)
            elif len(args.data_cache_dir) != 0:
                logging.info("No synthetic data url and private local data dir")
                return
        client.loop_forever()


def split_edge_data(args, edge_list=None):
    try:
        url = "http://127.0.0.1:5000/split_dataset"
        edge_list = json.loads(edge_list)
        json_params = {"runId": args.run_id, "edgeIds": edge_list, "dataset": args.dataset}
        response = requests.post(
            url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "keep-alive"}
        )
        result = response.json()["errno"]
        return result
    except requests.exceptions.SSLError as err:
        print(err)


def check_rundata(args):
    # local simulation run
    logging.info("Checking Run Data")
    # mlops run
    try:
        url = "http://127.0.0.1:5000/check_rundata"
        json_params = {
            "runId": args.run_id,
        }
        response = requests.post(
            url,
            json=json_params,
            verify=True,
            headers={"content-type": "application/json", "Connection": "keep-alive"},
        )
        return response.json()["private_local_dir"], response.json()["split_status"], response.json()["edgeids"], response.json()["dataset_s3_key"]
    except requests.exceptions.SSLError as err:
        print(err)


def query_data_server(args, edgeId, s3_obj, BUCKET_NAME):
    try:
        url = "http://127.0.0.1:5000/get_edge_dataset"
        json_params = {"runId": args.run_id, "edgeId": edgeId}
        response = requests.post(
            url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "keep-alive"}
        )
        if response.json()["errno"] == 0:
            if not check_is_download(
                os.path.join(
                    args.data_cache_dir,
                    "run_Id_%s" % args.run_id,
                    "edgeNums_%s" % (args.client_num_in_total),
                    args.dataset,
                    "edgeId_%s" % edgeId,
                    "cifar-10-batches-py",
                )
            ):
                make_dir(
                    os.path.join(
                        args.data_cache_dir,
                        "run_Id_%s" % args.run_id,
                        "edgeNums_%s" % (args.client_num_in_total),
                        args.dataset,
                        "edgeId_%s" % edgeId,
                    )
                )
                # start download the file
                download_s3_file(
                    s3_obj,
                    BUCKET_NAME,
                    edgeId,
                    response.json()["dataset_key"],
                    os.path.join(
                        args.data_cache_dir,
                        "run_Id_%s" % args.run_id,
                        "edgeNums_%s" % (args.client_num_in_total),
                        args.dataset,
                        "edgeId_%s" % edgeId,
                    ),
                    os.path.join(
                        args.data_cache_dir,
                        "run_Id_%s" % args.run_id,
                        "edgeNums_%s" % (args.client_num_in_total),
                        args.dataset,
                        "edgeId_%s" % edgeId,
                        "cifar-10-python.tar.gz",
                    ),
                )
            else:
                logging.info("Edge Data Already Exists. Start Training Now.")
        return response.json()
    except requests.exceptions.SSLError as err:
        print(err)
        return err


def load(args):
    return load_synthetic_data(args)


def combine_batches(batches):
    full_x = torch.from_numpy(np.asarray([])).float()
    full_y = torch.from_numpy(np.asarray([])).long()
    for (batched_x, batched_y) in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    return [(full_x, full_y)]


def load_synthetic_data(args):
    if args.training_type == "cross_silo" and args.dataset == "cifar10" and hasattr(args, 'synthetic_data_url') and args.synthetic_data_url.find("https") != -1:
        data_server_preprocess(args)
    dataset_name = args.dataset
    # check if the centralized training is enabled
    centralized = True if (args.client_num_in_total == 1 and args.training_type != "cross_silo") else False

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False

    if dataset_name == "mnist":
        download_mnist(args.data_cache_dir)
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_mnist(
            args,
            args.batch_size,
            train_path=os.path.join(args.data_cache_dir, "MNIST", "train"),
            test_path=os.path.join(args.data_cache_dir, "MNIST", "test"),
        )
        """
        For shallow NN or linear models, 
        we uniformly sample a fraction of clients each round (as the original FedAvg paper)
        """
        args.client_num_in_total = client_num

    elif dataset_name == "femnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_emnist(args.dataset, args.data_cache_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_shakespeare(args.batch_size)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_shakespeare(args.dataset, args.data_cache_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "fed_cifar100":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_cifar100(args.dataset, args.data_cache_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_lr":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_stackoverflow_lr(args.dataset, args.data_cache_dir)
        args.client_num_in_total = client_num
    elif dataset_name == "stackoverflow_nwp":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            client_num,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_federated_stackoverflow_nwp(args.dataset, args.data_cache_dir)
        args.client_num_in_total = client_num

    elif dataset_name == "ILSVRC2012":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_ImageNet(
            dataset=dataset_name,
            data_dir=args.data_cache_dir,
            partition_method=None,
            partition_alpha=None,
            client_number=args.client_num_in_total,
            batch_size=args.batch_size,
        )

    elif dataset_name == "gld23k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 233
        fed_train_map_file = os.path.join(args.data_cache_dir, "mini_gld_train_split.csv")
        fed_test_map_file = os.path.join(args.data_cache_dir, "mini_gld_test.csv")

        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_landmarks(
            dataset=dataset_name,
            data_dir=args.data_cache_dir,
            fed_train_map_file=fed_train_map_file,
            fed_test_map_file=fed_test_map_file,
            partition_method=None,
            partition_alpha=None,
            client_number=args.client_num_in_total,
            batch_size=args.batch_size,
        )

    elif dataset_name == "gld160k":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        args.client_num_in_total = 1262
        fed_train_map_file = os.path.join(args.data_cache_dir, "federated_train.csv")
        fed_test_map_file = os.path.join(args.data_cache_dir, "test.csv")

        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_landmarks(
            dataset=dataset_name,
            data_dir=args.data_cache_dir,
            fed_train_map_file=fed_train_map_file,
            fed_test_map_file=fed_test_map_file,
            partition_method=None,
            partition_alpha=None,
            client_number=args.client_num_in_total,
            batch_size=args.batch_size,
        )

    else:
        if dataset_name == "cifar10":
            if hasattr(args, "synthetic_data_url") or hasattr(args, "private_local_data"):
                if hasattr(args, "synthetic_data_url"):
                    args.private_local_data = ""
                else:
                    args.synthetic_data_url = ""
                if args.process_id != 0:
                    args.data_cache_dir = os.path.join(
                        args.data_cache_dir,
                        "run_Id_%s" % args.run_id,
                        "edgeNums_%s" % (args.client_num_in_total),
                        args.dataset,
                        "edgeId_%s" % args.client_id,
                    )
                (
                    train_data_num,
                    test_data_num,
                    train_data_global,
                    test_data_global,
                    train_data_local_num_dict,
                    train_data_local_dict,
                    test_data_local_dict,
                    class_num,
                ) = efficient_load_partition_data_cifar10(
                    args.dataset,
                    args.data_cache_dir,
                    args.partition_method,
                    args.partition_alpha,
                    args.client_num_in_total,
                    args.batch_size,
                    args.process_id,
                    args.synthetic_data_url,
                    args.private_local_data
                )

                if centralized:
                    train_data_local_num_dict = {
                        0: sum(user_train_data_num for user_train_data_num in train_data_local_num_dict.values())
                    }
                    train_data_local_dict = {
                        0: [batch for cid in sorted(train_data_local_dict.keys()) for batch in
                            train_data_local_dict[cid]]
                    }
                    test_data_local_dict = {
                        0: [batch for cid in sorted(test_data_local_dict.keys()) for batch in test_data_local_dict[cid]]
                    }
                    args.client_num_in_total = 1

                if full_batch:
                    train_data_global = combine_batches(train_data_global)
                    test_data_global = combine_batches(test_data_global)
                    train_data_local_dict = {
                        cid: combine_batches(train_data_local_dict[cid]) for cid in train_data_local_dict.keys()
                    }
                    test_data_local_dict = {
                        cid: combine_batches(test_data_local_dict[cid]) for cid in test_data_local_dict.keys()
                    }
                    args.batch_size = args_batch_size

                dataset = [
                    train_data_num,
                    test_data_num,
                    train_data_global,
                    test_data_global,
                    train_data_local_num_dict,
                    train_data_local_dict,
                    test_data_local_dict,
                    class_num,
                ]

                return dataset, class_num
            else:
                # data_loader = load_partition_data_cifar10
                data_loader = efficient_load_partition_data_cifar10

        elif dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100
        elif dataset_name == "cinic10":
            data_loader = load_partition_data_cinic10
        else:
            data_loader = load_partition_data_cifar10
        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = data_loader(
            args.dataset,
            args.data_cache_dir,
            args.partition_method,
            args.partition_alpha,
            args.client_num_in_total,
            args.batch_size,
        )

    if centralized:
        train_data_local_num_dict = {
            0: sum(user_train_data_num for user_train_data_num in train_data_local_num_dict.values())
        }
        train_data_local_dict = {
            0: [batch for cid in sorted(train_data_local_dict.keys()) for batch in train_data_local_dict[cid]]
        }
        test_data_local_dict = {
            0: [batch for cid in sorted(test_data_local_dict.keys()) for batch in test_data_local_dict[cid]]
        }
        args.client_num_in_total = 1

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {
            cid: combine_batches(train_data_local_dict[cid]) for cid in train_data_local_dict.keys()
        }
        test_data_local_dict = {cid: combine_batches(test_data_local_dict[cid]) for cid in test_data_local_dict.keys()}
        args.batch_size = args_batch_size

    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]

    return dataset, class_num


def load_poisoned_dataset_from_edge_case_examples(args):
    return load_poisoned_dataset(args=args)
