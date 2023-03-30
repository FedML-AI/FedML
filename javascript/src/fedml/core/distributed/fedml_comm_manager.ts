import { MyMessage } from '../../cross_web/client/message_define'
import { BaseCommunicationManager } from './communication/base_com_manager'
// import { MqttS3MNNCommManager } from './distributed/communication/mqtt_s3_mnn/mqtt_s3_comm_manager';
import { MqttS3MultiClientsCommManager } from './communication/mqtt_s3_multi_clients_comm_manager'
// import { fetchConfig } from '/@/api/config/index'
import { Message } from './communication/message'
import axios from 'axios'
export class FedMLCommManager {
  args;
  comm;
  rank: number;
  size;
  backend;
  com_manager;
  message_handler_dict;

  constructor(args, comm = null, rank = 0, size = 0, backend = 'MPI') {
    this.args = args;
    this.size = size;
    this.rank = Number(rank);
    this.backend = backend;
    this.comm = comm;
    this.com_manager = null;
    this.message_handler_dict = {};
    this.init_manager();
  }

  register_comm_manager(comm_manager: BaseCommunicationManager) {
    this.com_manager = comm_manager;
  }

  run() {
    this.com_manager.handle_receive_message();
    console.log('start run');
  }

  get_sender_id() {
    return this.rank;
  }

  async send_message(message) {
    const model_url = await this.com_manager.send_message(message);
    return model_url;
  }

  register_message_receive_handler(msg_type, handler_callback_func) {
    this.message_handler_dict[msg_type] = handler_callback_func;
  }

  finish() {
    console.info('__finish');
    // this.com_manager.stop_receive_message();
  }

  // mqtt config and s3 config need to be hard coded
  private init_manager() {
    // const { mqtt_config, s3_config } = await this.get_training_mqtt_s3_config();
    const mqtt_config = {
      BROKER_HOST: 'mqtt.fedml.ai',
      MQTT_PWD: 'exfedml4321CCC',
      BROKER_PORT: 1883,
      MQTT_KEEPALIVE: 180,
      MQTT_USER: 'admin',
    };
    this.get_training_mqtt_s3_config();
    const s3_config = {
      CN_S3_SAK: 'fpU7ED2Xht1UGYAQrX9j/UPwAlXhn0cAcJZXnNi+',
      CN_REGION_NAME: 'us-west-1',
      CN_S3_AKI: 'AKIAUAWARWF4SW36VYXP',
      BUCKET_NAME: 'fedml',
    };
    this.com_manager = new MqttS3MultiClientsCommManager(
      mqtt_config,
      s3_config,
      this.args.run_id,
      this.rank,
      this.size,
      this.args,
    );
    console.log('add_observer ', this);
    this.com_manager.add_observer(this);
  }

  async get_training_mqtt_s3_config() {
    const params = {
      config_name: ['mqtt_config', 's3_config'],
    };
    const { mqtt_config, s3_config } = await this.fetchConfig(params);
    console.log('config!!!!!!!!!!! ', mqtt_config, ' ', s3_config);
    return {
      mqtt_config: mqtt_config,
      s3_config: s3_config,
    };
  }

  receive_message(msg_type, msg_params) {
    console.log('receive topic ', msg_type, 'msg_params: ', msg_params);
    // TODO: callback handler_callback_func = self.message_handler_dict[msg_type], handler_callback_func(msg_params)
    const handler_callback_func = this.message_handler_dict[msg_type];
    handler_callback_func(msg_params);
  }

  async send_client_status(receive_id, status = 'ONLINE') {
    console.log('send_client_status');
    const message = new Message(MyMessage.MSG_TYPE_C2S_CLIENT_STATUS, 1, receive_id);
    message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_STATUS, status);
    message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, 'mac');
    await this.send_message(message);
  }

  async fetchConfig(params) {
    axios.post('https://jsonplaceholder.typicode.com/posts', {
      title: 'foo',
      body: 'bar',
      userId: 1,
    }).then(response => {
      console.log(response.data);
    }).catch(error => {
      console.log(error);
    });
  }
}

export default FedMLCommManager;
