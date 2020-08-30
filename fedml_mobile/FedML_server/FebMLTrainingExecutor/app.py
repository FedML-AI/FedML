import datetime
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_core.distributed.communication import Observer

import time

from flask import Flask, request, render_template, jsonify, send_from_directory
from fedml_mobile.FedML_server.FebMLTrainingExecutor.log import __log
from fedml_mobile.FedML_server.FebMLTrainingExecutor.conf.conf import MODEL_FOLDER_PATH, MQTT_BROKER_HOST, \
    MQTT_BROKER_PORT, RESOURCE_DIR_PATH

from fedml_core.distributed.communication.mqtt import MqttClient


app = Flask(__name__)

__log.info(MQTT_BROKER_HOST)
__log.info(MQTT_BROKER_PORT)

HOST = "81.71.1.31"
client = MqttClient(HOST, MQTT_BROKER_PORT, "TrainingExecutor")


class Obs(Observer):
    def receive_message(self, msg_type, msg_params) -> None:
        global __log
        __log.info("receive_message(%s,%s)" % (msg_type, msg_params))
        print("receive_message(%s,%s)" % (msg_type, msg_params))


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


@app.route('/api/register', methods=['POST', ])
def register_device():
    __log.info("register_device()")
    __log.info(request)
    # print(request.json)
    # device_id = request.json['device_id']
    # TODO: save device_id
    client.send("hello", "Hello world!")
    client.send("temperature", "24.0")
    client.send("humidity", "65%")
    # return jsonify({"errno": 0, "executorId": client.client_id, "executorTopic": client.topic})
    return jsonify({"errno": 0, "executorId": "executorId", "executorTopic": "executorTopic"})


if __name__ == '__main__':
    app.run(host='192.168.3.104', port=5000, debug=True)
