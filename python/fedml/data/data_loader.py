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
from fedml.data.cifar10.efficient_loader import efficient_load_partition_data_cifar10
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

from .MNIST.data_loader import download_mnist
from .edge_case_examples.data_loader import load_poisoned_dataset
import logging


from .MNIST.data_loader import download_mnist
from .edge_case_examples.data_loader import load_poisoned_dataset
import logging
import requests
from botocore.config import Config
import boto3
import tarfile
import shutil
import paho.mqtt.client as mqtt_client
import json
import time
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
            make_dir(os.path.join(args.data_cache_dir, 'run_Id_%s' % args.run_id, 'edgeNums_%s' % (args.client_num_in_total), args.dataset, 'edgeId_%s' % (args.process_id - 1)))
            # start download the file
            download_s3_file(json.loads(msg.payload.decode())["edge_id"]-1, json.loads(msg.payload.decode())["dataset"],
                             os.path.join(args.data_cache_dir, 'run_Id_%s' % args.run_id, 'edgeNums_%s' % (args.client_num_in_total), args.dataset, 'edgeId_%s' % (args.process_id - 1)),
                             os.path.join(args.data_cache_dir, 'run_Id_%s' % args.run_id, 'edgeNums_%s' % (args.client_num_in_total), args.dataset, 'edgeId_%s' % (args.process_id - 1),
                                          "cifar-10-python.tar.gz"))
    topic = "data_svr/dataset/%s" % (args.process_id)
    client.subscribe(topic)
    client.on_message = on_message


def disconnect(client: mqtt_client):
    client.disconnect()
    logging.info(f'Received message, Mqtt stop listen.')


def data_server_preprocess(args):
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
                    split_status = split_edge_data(args)
                elif split_status == 1:
                    logging.info("Data Server Is Splitting Dataset, Waiting For Mqtt Message")
                elif split_status == 2:
                    logging.info("Data Server Splitted Dataset Complete")
                    query_data_server(args, args.client_id_list)
                    disconnect(client)
            # Mlops Run
        client.loop_forever()


def split_edge_data(args):
    edge_li = []
    try:
        url = "http://127.0.0.1:51200/split_dataset"
        if args.run_id == '0':
            for i in range(1, args.client_num_in_total + 1):
                edge_li.append(i)
            json_params = {
                "runId": args.run_id,
                "edgeIds": edge_li,
                "deviceId": args.device_id,
                "dataset": args.dataset
            }
        else:
            json_params = {
                "runId": args.run_id,
                "edgeIds": edge_li,
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
    edge_li = []
    if args.run_id == '0':
        for i in range(1, args.client_num_in_total+1):
            edge_li.append(i)
    try:
        url = "http://127.0.0.1:51200/check_rundata"
        json_params = {
            "runId": args.run_id,
            "deviceId": args.device_id,
            "edgeIds": edge_li,
            "dataset": args.dataset
        }
        response = requests.post(
            url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "keep-alive"}
        )
        split_status = response.json()["split_status"]
        return split_status
    except requests.exceptions.SSLError as err:
        print(err)



def query_data_server(args, edgeId):
    try:
        url = "http://127.0.0.1:51200/get_edge_dataset"
        json_params = {"runId": args.run_id,
                       "edgeId": edgeId}
        response = requests.post(
            url, json=json_params, verify=True, headers={"content-type": "application/json", "Connection": "keep-alive"}
        )
        if response.json()['errno'] == 0:
            if not check_is_download(os.path.join(args.data_cache_dir, 'run_Id_%s' % args.run_id,
                                              'edgeNums_%s' % (args.client_num_in_total), args.dataset,
                                              'edgeId_%s' % (args.process_id - 1),
                                              "cifar-10-batches-py")):
                make_dir(os.path.join(args.data_cache_dir, 'run_Id_%s' % args.run_id,
                                      'edgeNums_%s' % (args.client_num_in_total), args.dataset,
                                      'edgeId_%s' % (args.process_id - 1)))
                # start download the file
                download_s3_file(args.process_id - 1,
                                 args.dataset,
                                 os.path.join(args.data_cache_dir, 'run_Id_%s' % args.run_id,
                                              'edgeNums_%s' % (args.client_num_in_total), args.dataset,
                                              'edgeId_%s' % (args.process_id - 1)),
                                 os.path.join(args.data_cache_dir, 'run_Id_%s' % args.run_id,
                                              'edgeNums_%s' % (args.client_num_in_total), args.dataset,
                                              'edgeId_%s' % (args.process_id - 1),
                                              "cifar-10-python.tar.gz"))
            else:
                logging.info("Edge Data Already Exists. Start Training Now.")
        return response.json()
    except requests.exceptions.SSLError as err:
        print(err)
        return err

def make_dir(file_path):
    """
    package tar.gz file
    :param file_path: target file path
    :param process_id: current start edge id
    :return: bool
    """
    try:
        os.makedirs(file_path)
        return True
    except Exception as e:
        logging.exception(e)
        return False

def download_s3_file(process_id, path_s3, root, path_local):
    """
    download file
    :param path_s3: s3 key
    :param path_local: local path
    :return:
    """
    retry = 0
    while retry < 3:
        # retry 3 times
        logging.info(f'Start downloading files. | path_s3: {path_s3} | path_local: {path_local}')
        try:
            with open(path_local, 'wb') as data:
                s3.download_fileobj(BUCKET_NAME, path_s3, data)
            file_size = os.path.getsize(path_local)
            logging.info(f'Downloading completed. | size: {round(file_size / 1048576, 2)} MB')
            file_extract(process_id, root, path_local)
            move_file(process_id, root)
            break
        except Exception as e:
            logging.error(f'Download zip failed. | Exception: {e}')
            retry += 1
    if retry >= 3:
        logging.error(f'Download zip failed after max retry.')


def check_is_download(path):
    if os.path.isdir(path):
        logging.info(f'Edge Data exist.')
        return True
    else:
        return False


def file_extract(process_id: int, root: str, file_path: str):
    if file_path.endswith('.zip'):
        return un_zip(file_path)
    elif file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
        return un_tar(process_id, root, file_path)
    else:
        return None


def un_zip(file_name):
    """unzip zip file"""
    dest_dir = file_name
    try:
        with zipfile.ZipFile(file_name) as zip_file:
            if not os.path.isdir(dest_dir):
                os.mkdir(file_name)
            zip_file.extractall(path=dest_dir)
        return dest_dir
    except Exception as e:
        shutil.rmtree(dest_dir)
        logging.exception(e)
        return None


def un_gz(file_name):
    """un_gz zip file"""
    # get file name without suffix
    f_name = file_name.replace(".gz", "")
    try:
        with gzip.GzipFile(file_name) as g_file:
            with open(f_name, "wb") as dest_file:
                dest_file.write(g_file.read())
        return f_name
    except Exception as e:
        shutil.rmtree(f_name)
        logging.exception(e)
        return None


def un_tar(process_id, root, file_name):
    """ untar zip file"""
    dest_dir = os.path.join(root, 'cifar-10-batches-py')
    try:
        with tarfile.open(file_name) as tar:
            if not os.path.isdir(dest_dir):
                os.mkdir(dest_dir)
            tar.extractall(path=dest_dir)
            logging.info("untar zip file finished")
        return dest_dir
    except Exception as e:
        shutil.rmtree(dest_dir)
        logging.exception(e)
        return None


def move_file(process_id, root):
    target_src = os.path.join(root, 'cifar-10-batches-py', 'device_%s' % process_id)
    dirs = os.listdir(target_src)
    for file_name in dirs:
        shutil.move(os.path.join(target_src, file_name), os.path.join(root, 'cifar-10-batches-py'))
    logging.info("Move file finished")
    return None

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
    # redirect the file path
    process_data_dir = (
        os.path.join(args.data_cache_dir, 'run_Id_%s' % args.run_id, 'edgeNums_%s' % (args.client_num_in_total),
                     args.dataset, 'edgeId_%s' % (args.process_id - 1))
        if args.process_id != 0
        else None
    )
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
            process_data_dir,
            args.process_id,
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
