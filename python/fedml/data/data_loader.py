import gzip
import os
import zipfile

import numpy as np
import torch
from fedml.data.FederatedEMNIST.data_loader import load_partition_data_federated_emnist
from fedml.data.ImageNet.data_loader import load_partition_data_ImageNet
from fedml.data.Landmarks.data_loader import load_partition_data_landmarks
from fedml.data.MNIST.data_loader import load_partition_data_mnist
from fedml.data.cifar10.data_loader import load_partition_data_cifar10
# from fedml.data.cifar10.efficient_loader import efficient_load_partition_data_cifar10
from cifar10.efficient_loader import efficient_load_partition_data_cifar10
from fedml.data.cifar100.data_loader import load_partition_data_cifar100
from fedml.data.cinic10.data_loader import load_partition_data_cinic10
from fedml.data.fed_cifar100.data_loader import load_partition_data_federated_cifar100
from fedml.data.fed_shakespeare.data_loader import (
    load_partition_data_federated_shakespeare,
)
from fedml.data.shakespeare.data_loader import load_partition_data_shakespeare
from fedml.data.stackoverflow_lr.data_loader import (
    load_partition_data_federated_stackoverflow_lr,
)
from fedml.data.stackoverflow_nwp.data_loader import (
    load_partition_data_federated_stackoverflow_nwp,
)
from file_operation import *


# from .MNIST.data_loader import download_mnist
# from .edge_case_examples.data_loader import load_poisoned_dataset
import logging
import requests
from botocore.config import Config
import boto3
import tarfile
import shutil
import paho.mqtt.client as mqtt_client
import json
import random

broker = "mqtt.fedml.ai"
port = 1883
username = "admin"
password = "password"
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 1000)}'

_config = Config(
    retries={
        'max_attempts': 4,
        'mode': 'standard'
    }
)
CN_REGION_NAME = "us-east-1"
CN_S3_AKI = "AKIAY7HWPQWRHEZQDVGS"
CN_S3_SAK = "chnPTIfUYxLbGCChXqFCTdvcz3AGWqsX3zTeynnL"
BUCKET_NAME = "fedmls3"


# s3 client
s3 = boto3.client('s3', region_name=CN_REGION_NAME, aws_access_key_id=CN_S3_AKI,
                  aws_secret_access_key=CN_S3_SAK, config=_config)
# s3 resource
s3_resource = boto3.resource('s3', region_name=CN_REGION_NAME, config=_config,
                             aws_access_key_id=CN_S3_AKI, aws_secret_access_key=CN_S3_SAK)


def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Host!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id, clean_session=False)
    client.username_pw_set(username, password)
    client.connect(broker, port)
    return client


def subscribe(client: mqtt_client, args):
    def on_message(client, userdata, msg):
        logging.info(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
        if msg.payload.decode():
            disconnect(client)
            if args.run_id == '0':
                make_dir(os.path.join(args.data_cache_dir, 'run_Id_%s' % args.run_id, 'edgeNums_%s' % (args.client_num_in_total), args.dataset, 'edgeId_%s' % (args.process_id)))
                # start download the file
                download_s3_file(json.loads(msg.payload.decode())["edge_id"], json.loads(msg.payload.decode())["dataset"],
                                 os.path.join(args.data_cache_dir, 'run_Id_%s' % args.run_id, 'edgeNums_%s' % (args.client_num_in_total), args.dataset, 'edgeId_%s' % (args.process_id)),
                                 os.path.join(args.data_cache_dir, 'run_Id_%s' % args.run_id, 'edgeNums_%s' % (args.client_num_in_total), args.dataset, 'edgeId_%s' % (args.process_id),
                                              "cifar-10-python.tar.gz"))
            else:
                make_dir(os.path.join(args.data_cache_dir, 'run_Id_%s' % args.run_id,
                                      'edgeNums_%s' % (args.client_num_in_total), args.dataset,
                                      'edgeId_%s' % (15 + int(args.client_id_list[1]))))
                # start download the file
                download_s3_file(json.loads(msg.payload.decode())["edge_id"],
                                 json.loads(msg.payload.decode())["dataset"],
                                 os.path.join(args.data_cache_dir, 'run_Id_%s' % args.run_id,
                                              'edgeNums_%s' % (args.client_num_in_total), args.dataset,
                                              'edgeId_%s' % (15 + int(args.client_id_list[1]))),
                                 os.path.join(args.data_cache_dir, 'run_Id_%s' % args.run_id,
                                              'edgeNums_%s' % (args.client_num_in_total), args.dataset,
                                              'edgeId_%s' % (15 + int(args.client_id_list[1])),
                                              "cifar-10-python.tar.gz"))
    # topic = "data_svr/dataset/%s" % (15 + int(args.client_id_list[1]))
    topic = "data_svr/dataset/%s" % args.process_id
    client.subscribe(topic)
    client.on_message = on_message


def disconnect(client: mqtt_client):
    client.disconnect()
    logging.info(f'Received message, Mqtt stop listen.')


def data_server_preprocess(args):
    # args.run_id = 1378
    if args.process_id == 0:
        pass
    else:
        client = connect_mqtt()
        subscribe(client, args)
        if args.dataset == "cifar10":
            # Local Simulation Run
            if args.run_id == '0':
                # split_status = 0 (unsplit), 1(splitting), 2(split_finished), 3(split failed, Interruption occurs)
                split_status = check_rundata(args)
                if split_status == 0 or split_status == 3:
                    logging.info("Data Server Start Splitting Dataset")
                    split_edge_data(args)
                elif split_status == 1:
                    logging.info("Data Server Is Splitting Dataset, Waiting For Mqtt Message")
                elif split_status == 2:
                    logging.info("Data Server Splitted Dataset Complete")
                    query_data_server(args, int(args.client_id_list[1]))
                    disconnect(client)
                args.data_cache_dir = args.data_cache_dir = os.path.join(args.data_cache_dir, 'run_Id_%s' % args.run_id,
                                                                         'edgeNums_%s' % (args.client_num_in_total),
                                                                         args.dataset,
                                                                         'edgeId_%s' % (int(args.client_id_list[1])))
            # Mlops Run
            else:
                # check mlops run_status
                private_local_dir, split_status, edgeids = check_rundata(args)
                # MLOPS Run. User supply the local data dir
                if len(private_local_dir) != 0:
                    logging.info("User has set the private local data dir")
                    args.data_cache_dir = private_local_dir
                    disconnect(client)
                # MLOPS Run need to Split Data
                else:
                    if split_status == 0 or split_status == 3:
                        logging.info("Data Server Start Splitting Dataset")
                        split_edge_data(args, edgeids)
                    elif split_status == 1:
                        logging.info("Data Server Is Splitting Dataset, Waiting For Mqtt Message")
                    elif split_status == 2:
                        logging.info("Data Server Splitted Dataset Complete")
                        query_data_server(args, 15 + int(args.client_id_list[1]))
                        disconnect(client)
                args.data_cache_dir = args.data_cache_dir = os.path.join(args.data_cache_dir, 'run_Id_%s' % args.run_id,
                                                                         'edgeNums_%s' % (args.client_num_in_total),
                                                                         args.dataset,
                                                                         'edgeId_%s' % (15 + int(args.client_id_list[1])))
        client.loop_forever()


def split_edge_data(args, edge_list = None):
    try:
        url = "http://127.0.0.1:5000/split_dataset"
        if args.run_id == '0':
            edge_li = []
            for i in range(1, args.client_num_in_total + 1):
                edge_li.append(i)
            json_params = {
                "runId": args.run_id,
                "edgeIds": edge_li,
                "deviceId": args.device_id,
                "dataset": args.dataset
            }
        else:
            edge_list = json.loads(edge_list)
            json_params = {
                "runId": args.run_id,
                "edgeIds": edge_list,
                "dataset": args.dataset
            }
        response = requests.post(
            url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "keep-alive"}
        )
        result = response.json()["errno"]
        return result
    except requests.exceptions.SSLError as err:
        print(err)


def check_rundata(args):
    # args.run_id = 1378
    # local simulation run
    logging.info("Checking Run Data")
    edge_li = []
    if args.run_id == '0':
        for i in range(1, args.client_num_in_total+1):
            edge_li.append(i)
        try:
            url = "http://127.0.0.1:5000/check_rundata"
            json_params = {
                "runId": args.run_id,
                "deviceId": args.device_id,
                "edgeIds": edge_li,
                "dataset": args.dataset
            }
            response = requests.post(
                url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "keep-alive"}
            )
            return response.json()["split_status"]
        except requests.exceptions.SSLError as err:
            print(err)
    else:
        # mlops run
        try:
            url = "http://127.0.0.1:5000/check_rundata"
            json_params = {
                "runId": args.run_id,
            }
            response = requests.post(
                url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "keep-alive"}
            )
            return response.json()['private_local_dir'], response.json()['split_status'], response.json()['edgeids']
        except requests.exceptions.SSLError as err:
            print(err)



def query_data_server(args, edgeId):
    try:
        url = "http://127.0.0.1:5000/get_edge_dataset"
        json_params = {"runId": args.run_id,
                       "edgeId": edgeId}
        response = requests.post(
            url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "keep-alive"}
        )
        if response.json()['errno'] == 0:
            if not check_is_download(os.path.join(args.data_cache_dir, 'run_Id_%s' % args.run_id,
                                              'edgeNums_%s' % (args.client_num_in_total), args.dataset,
                                              'edgeId_%s' % edgeId,
                                              "cifar-10-batches-py")):
                make_dir(os.path.join(args.data_cache_dir, 'run_Id_%s' % args.run_id,
                                      'edgeNums_%s' % (args.client_num_in_total), args.dataset,
                                      'edgeId_%s' % edgeId))
                # start download the file
                download_s3_file(edgeId,
                                 response.json()['dataset_key'],
                                 os.path.join(args.data_cache_dir, 'run_Id_%s' % args.run_id,
                                              'edgeNums_%s' % (args.client_num_in_total), args.dataset,
                                              'edgeId_%s' % edgeId),
                                 os.path.join(args.data_cache_dir, 'run_Id_%s' % args.run_id,
                                              'edgeNums_%s' % (args.client_num_in_total), args.dataset,
                                              'edgeId_%s' % edgeId,
                                              "cifar-10-python.tar.gz"))
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
    data_server_preprocess(args)
    dataset_name = args.dataset
    # check if the centralized training is enabled
    centralized = (
        True
        if (args.client_num_in_total == 1 and args.training_type != "cross_silo")
        else False
    )

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
            train_path=args.data_cache_dir + "/MNIST/train",
            test_path=args.data_cache_dir + "/MNIST/test",
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
        ) = load_partition_data_federated_stackoverflow_lr(
            args.dataset, args.data_cache_dir
        )
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
        ) = load_partition_data_federated_stackoverflow_nwp(
            args.dataset, args.data_cache_dir
        )
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
        fed_train_map_file = os.path.join(
            args.data_cache_dir, "mini_gld_train_split.csv"
        )
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
            # data_loader = load_partition_data_cifar10
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
                args.process_id
            )

            if centralized:
                train_data_local_num_dict = {
                    0: sum(
                        user_train_data_num
                        for user_train_data_num in train_data_local_num_dict.values()
                    )
                }
                train_data_local_dict = {
                    0: [
                        batch
                        for cid in sorted(train_data_local_dict.keys())
                        for batch in train_data_local_dict[cid]
                    ]
                }
                test_data_local_dict = {
                    0: [
                        batch
                        for cid in sorted(test_data_local_dict.keys())
                        for batch in test_data_local_dict[cid]
                    ]
                }
                args.client_num_in_total = 1

            if full_batch:
                train_data_global = combine_batches(train_data_global)
                test_data_global = combine_batches(test_data_global)
                train_data_local_dict = {
                    cid: combine_batches(train_data_local_dict[cid])
                    for cid in train_data_local_dict.keys()
                }
                test_data_local_dict = {
                    cid: combine_batches(test_data_local_dict[cid])
                    for cid in test_data_local_dict.keys()
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
            0: sum(
                user_train_data_num
                for user_train_data_num in train_data_local_num_dict.values()
            )
        }
        train_data_local_dict = {
            0: [
                batch
                for cid in sorted(train_data_local_dict.keys())
                for batch in train_data_local_dict[cid]
            ]
        }
        test_data_local_dict = {
            0: [
                batch
                for cid in sorted(test_data_local_dict.keys())
                for batch in test_data_local_dict[cid]
            ]
        }
        args.client_num_in_total = 1

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {
            cid: combine_batches(train_data_local_dict[cid])
            for cid in train_data_local_dict.keys()
        }
        test_data_local_dict = {
            cid: combine_batches(test_data_local_dict[cid])
            for cid in test_data_local_dict.keys()
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


def load_poisoned_dataset_from_edge_case_examples(args):
    return load_poisoned_dataset(args=args)
