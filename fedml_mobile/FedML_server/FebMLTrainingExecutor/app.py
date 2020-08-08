import datetime

from flask import Flask, request, render_template
from conf import MQTT_BROKER_HOST
from conf import MQTT_BROKER_PORT
from mqtt_client import MqttClient

client = MqttClient(MQTT_BROKER_HOST, MQTT_BROKER_PORT)
app = Flask(__name__)


@app.route('/')
def hello_world():
    cur_time = datetime.datetime.now()
    str_cur_time = cur_time.strftime("%Y-%m-%d %H:%M:%S")
    return render_template('Home.html', time=str_cur_time)


@app.route('/test', methods=['POST', ])
def test():
    print(request.headers)
    print(type(request.json))
    print(request.json)
    result = request.json['a'] + request.json['b']
    client.send("hello", str(result))
    return str(result)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
