import { MyMessage } from '../../cross_web/client/message_define'
import { fetchConfig } from '../../../api/index'
import type { BaseCommunicationManager } from './communication/base_com_manager'
import { MqttS3MultiClientsCommManager } from './communication/mqtt_s3_multi_clients_comm_manager'
import { Message } from './communication/message'

export class FedMLCommManager {
  args
  comm: null
  rank: number
  size
  backend
  com_manager: BaseCommunicationManager | null
  message_handler_dict: Record<string, { (msg: any): any }>

  constructor(args: any, comm = null, rank = 0, size = 0, backend = 'MPI') {
    this.args = args
    this.size = size
    this.rank = Number(rank)
    this.backend = backend
    this.comm = comm
    this.com_manager = null
    this.message_handler_dict = {}
    this.init_manager().catch(console.warn)
  }

  register_comm_manager<T extends BaseCommunicationManager>(comm_manager: T) {
    this.com_manager = comm_manager
  }

  run() {
    if (!this.com_manager)
      return console.warn('The com_manager is not initialized~')
    this.com_manager.handle_receive_message()
    console.log('start run')
  }

  get_sender_id() {
    return this.rank
  }

  async send_message(message: Message) {
    if (!this.com_manager) {
      console.warn('The com_manager is not initialized~')
      return null
    }

    const model_url = await this.com_manager.send_message(message)
    return model_url
  }

  register_message_receive_handler(
    msg_type: string | number,
    handler_callback_func: {
      (msg: any): any
    },
  ) {
    this.message_handler_dict[String(msg_type)] = handler_callback_func
  }

  finish() {
    console.info('__finish')
    // this.com_manager.stop_receive_message();
  }

  // mqtt config and s3 config need to be hard coded
  async init_manager() {
    const data = await this.get_training_mqtt_s3_config()
    const mqtt_config = {
      BROKER_HOST: data.mqtt_config.BROKER_HOST,
      MQTT_PWD: data.mqtt_config.MQTT_PWD,
      BROKER_PORT: data.mqtt_config.BROKER_PORT,
      MQTT_KEEPALIVE: data.mqtt_config.MQTT_KEEPALIVE,
      MQTT_USER: data.mqtt_config.MQTT_USER,
    }
    const s3_config = {
      CN_S3_SAK: data.s3_config.CN_S3_SAK,
      CN_REGION_NAME: data.s3_config.CN_REGION_NAME,
      CN_S3_AKI: data.s3_config.CN_S3_AKI,
      BUCKET_NAME: data.s3_config.BUCKET_NAME,
    }
    this.com_manager = new MqttS3MultiClientsCommManager(
      mqtt_config,
      s3_config,
      this.args.run_id,
      this.rank,
      this.size,
      this.args,
    )
    console.log('add_observer ', this)
    this.com_manager.add_observer(this)
  }

  async get_training_mqtt_s3_config() {
    const params = {
      config_name: ['mqtt_config', 's3_config'],
    }
    const { mqtt_config, s3_config } = await fetchConfig(params)
    return {
      mqtt_config,
      s3_config,
    }
  }

  receive_message(msg_type: string | number, msg_params: any) {
    console.log('receive topic ', msg_type, 'msg_params: ', msg_params)
    // TODO: callback handler_callback_func = self.message_handler_dict[msg_type], handler_callback_func(msg_params)
    const handler_callback_func = this.message_handler_dict[msg_type]
    handler_callback_func(msg_params)
  }

  async send_client_status(receive_id: number | undefined, status = 'ONLINE') {
    console.log('send_client_status')
    const message = new Message(MyMessage.MSG_TYPE_C2S_CLIENT_STATUS, 1, receive_id)
    message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_STATUS, status)
    message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, 'mac')
    await this.send_message(message)
  }
}

export default FedMLCommManager
