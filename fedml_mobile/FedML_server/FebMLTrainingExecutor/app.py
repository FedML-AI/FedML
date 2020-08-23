import datetime
import os
import time

from flask import Flask, request, render_template, jsonify, send_from_directory

from conf import MODEL_FOLDER_PATH
from conf import RESOURCE_DIR_PATH
from conf import MQTT_BROKER_HOST
from conf import MQTT_BROKER_PORT
from mqtt_client import MqttClient

client = MqttClient(MQTT_BROKER_HOST, MQTT_BROKER_PORT)
app = Flask(__name__)

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
    # TODO: parse data_json get training info
    data_json = request.json
    file_dir = MODEL_FOLDER_PATH
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    f = request.files['model_file']
    if f and allowed_file(f.filename):
        fname = f.filename
        # get file extension
        ext = fname.rsplit('.', 1)[1]
        unix_time = int(time.time())
        # modify file name
        new_filename = str(unix_time) + '.' + ext
        # save to the upload folder
        f.save(os.path.join(file_dir, new_filename))
        return jsonify({"errno": 0, "errmsg": "upload success!"})
    else:
        return jsonify({"errno": 1001, "errmsg": "upload fail!"})


@app.route("/download/<path:filename>")
def downloader(filename):
    return send_from_directory(RESOURCE_DIR_PATH, filename, as_attachment=True)


@app.route('/api/deviceOnLine', methods=['POST', ])
def device_on_line():
    print(request.headers)
    print(type(request.json))
    print(request.json)
    device_id = request.json['deviceId']
    # TODO: save device_id
    client.send("hello", str(device_id))
    return jsonify({"errno": 0, "executorId": client.client_id, "executorTopic": client.topic})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
