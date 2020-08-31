import logging
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

import numpy as np
import torch

from fedml_mobile.server.executor.fedavg.FedAVGAggregator import FedAVGAggregator
from fedml_mobile.server.executor.fedavg.FedAvgServerManager import FedAVGServerManager


from fedml_api.model.deep_neural_networks.mobilenet import mobilenet
from fedml_api.model.deep_neural_networks.resnet import resnet56

from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_distributed_cifar100
from fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_distributed_cinic10
from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_distributed_cifar10

from fedml_core.distributed.communication import Observer

from flask import Flask, request, render_template, jsonify, send_from_directory
from fedml_mobile.server.executor.conf.conf import MODEL_FOLDER_PATH, RESOURCE_DIR_PATH

from fedml_core.distributed.communication.mqtt import MqttCommManager

app = Flask(__name__)

HOST = "81.71.1.31"
PORT = 1883
client = MqttCommManager(HOST, PORT, "TrainingExecutor")


class Obs(Observer):
    def receive_message(self, msg_type, msg_params) -> None:
        global __log
        __log.info("receive_message(%s,%s)" % (msg_type, msg_params))


client.add_observer(Obs())


ALLOWED_EXTENSIONS = {'txt', 'png', 'jpg', 'JPG', 'PNG'}


# file EXTENSIONS check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def hello_world():
    cur_time = datetime.datetime.now()
    str_cur_time = cur_time.strftime("%Y-%m-%d %H:%M:%S")
    return render_template('Home.html', time=str_cur_time)


# test page
@app.route('/test/upload')
def upload_test():
    return render_template('upload.html')


@app.route('/api/upload', methods=['POST'], strict_slashes=False)
def api_upload():
    # TODO: parse form get training info
    print(request.values['filename'])
    file_dir = MODEL_FOLDER_PATH
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    f = request.files['model_file']
    if f and allowed_file(f.filename):
        fname = f.filename
        # get file without extension
        name_without_ext = fname.rsplit('.', 1)[0]
        # get file extension
        ext = fname.rsplit('.', 1)[1]
        unix_time = int(time.time())
        # modify file name
        new_filename = '%s_%s.%s' % (name_without_ext, str(unix_time), ext)
        # save to the upload folder
        f.save(os.path.join(file_dir, new_filename))
        return jsonify({"errno": 0, "errmsg": "upload success!"})
    else:
        return jsonify({"errno": 1001, "errmsg": "upload fail!"})


@app.route("/download/<path:filename>")
def downloader(filename):
    return send_from_directory(RESOURCE_DIR_PATH, filename, as_attachment=True)



@app.route('/api/register', methods=['POST'])
def register_device():
    __log.info("register_device()")
    __log.info(request.args['device_id'])
    # TODO: save device_id
    client.send("hello", "Hello world!")
    client.send("temperature", "24.0")
    client.send("humidity", "65%")
    # return jsonify({"errno": 0, "executorId": client.client_id, "executorTopic": client.topic})
    training_task_json = {"dataset": 'mnist',
                          "model": "lr",
                          "round_num": 100,
                          "local_epoch_num": 10,
                          "local_lr": 0.03,
                          "batch_size": 10}
    return jsonify({"errno": 0,
                    "executorId": "executorId",
                    "executorTopic": "executorTopic",
                    "training_task": training_task_json})


def init_training_device(process_ID, fl_worker_num, gpu_num_per_machine):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    if process_ID == 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device
    process_gpu_dict = dict()
    for client_index in range(fl_worker_num):
        gpu_index = client_index % gpu_num_per_machine
        process_gpu_dict[client_index] = gpu_index

    logging.info(process_gpu_dict)
    device = torch.device("cuda:" + str(process_gpu_dict[process_ID - 1]) if torch.cuda.is_available() else "cpu")
    logging.info(device)
    return device


if __name__ == '__main__':
    worker_number = 0
    process_id = 0
    training_task_info = None
    [dataset, model, args] = training_task_info

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(worker_number)

    logging.info("process_id = %d, size = %d" % (process_id, worker_number))
    device = init_training_device(process_id, worker_number - 1, 4)

    # load data
    if args.dataset == "cifar10":
        data_loader = load_partition_data_distributed_cifar10
    elif args.dataset == "cifar100":
        data_loader = load_partition_data_distributed_cifar100
    elif args.dataset == "cinic10":
        data_loader = load_partition_data_distributed_cinic10
    else:
        data_loader = load_partition_data_distributed_cifar10
    train_data_num, train_data_global, \
    test_data_global, local_data_num, \
    train_data_local, test_data_local, class_num = data_loader(process_id, args.dataset, args.data_dir,
                                                               args.partition_method, args.partition_alpha,
                                                               args.client_number, args.batch_size)

    # create the model
    model = None
    if args.model == "resnet56":
        model = resnet56(class_num)
    elif args.model == "mobilenet":
        model = mobilenet(class_num=class_num)

    client_num = 4
    aggregator = FedAVGAggregator(train_data_global, test_data_global, train_data_num, client_num, device, model,
                                  args)

    server_manager = FedAVGServerManager(args, aggregator)
    server_manager.run()

    app.run(host='127.0.0.1', port=5000, debug=True)
