import { FedMLCommManager } from '../../core/distributed/fedml_comm_manager';
import { Message } from '../../core/distributed/communication/message';
import { event } from '../../mlops/mlops_init';
import { MyMessage } from './message_define';
import { TrainerDistAdapter } from './fedml_trainer_dist_adapter';
// import { computed } from 'vue';
// import { useAppStore } from '/@/store/modules/app';

// const appStore = useAppStore();
// const currClient = computed(() => appStore.getClient);
export class ClientMasterManager extends FedMLCommManager {
  args: object;
  trainer_dist_adapter: TrainerDistAdapter;
  num_rounds: number;
  round_idx: number;
  rank;
  client_real_ids;
  client_real_id;
  has_sent_online_msg;
  message_handler_dict;
  is_inited;

  constructor(
    args,
    trainer_dist_adapter: TrainerDistAdapter,
    comm = null,
    rank = 0,
    size = 0,
    backend = 'MPI',
  ) {
    super(args, comm, rank, size, backend);

    this.trainer_dist_adapter = trainer_dist_adapter;
    this.args = args;
    this.num_rounds = args.comm_round;
    this.round_idx = 0;
    this.rank = rank;
    this.client_real_ids = args.client_id_list;
    this.message_handler_dict = {};
    this.client_real_id = this.client_real_ids[0];
    this.has_sent_online_msg = false;
    this.is_inited = false;
  }

  register_message_receive_handlers() {
    console.log('fedml_runner register_message_receive_handlers');
    super.register_message_receive_handler(
      MyMessage.MSG_TYPE_CONNECTION_IS_READY,
      this.handle_message_connection_ready.bind(this),
    );
    super.register_message_receive_handler(
      MyMessage.MSG_TYPE_S2C_CHECK_CLIENT_STATUS,
      this.handle_message_check_status.bind(this),
    );
    super.register_message_receive_handler(
      MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
      this.handle_message_init.bind(this),
    );
    super.register_message_receive_handler(
      MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
      this.handle_message_receive_model_from_server.bind(this),
    );
    super.register_message_receive_handler(
      MyMessage.MSG_TYPE_S2C_FINISH,
      this.handle_message_finish.bind(this),
    );
  }

  handle_message_connection_ready(msg) {
    console.log('handle_message_connection_ready ', msg);
    if (!this.has_sent_online_msg) {
      this.has_sent_online_msg = true;
      this.send_client_status(0);
    }
  }

  handle_message_check_status(msg) {
    console.log('handle_message_check_status ', msg);
    this.send_client_status(0);
  }

  async handle_message_init(msg) {
    if (this.is_inited) {
      return;
    }

    this.is_inited = true;
    console.log('handle_message_init: ', msg);
    const global_model_params = msg.msg_params[MyMessage.MSG_ARG_KEY_MODEL_PARAMS];
    const data_silo_index = msg.msg_params[MyMessage.MSG_ARG_KEY_CLIENT_INDEX];
    console.log('data_silo_index = ', data_silo_index);
    console.log('handle_message_init ', global_model_params);
    this.report_client_training_status(
      this.args.client_id_list[0],
      this.args.run_id,
      MyMessage.MSG_MLOPS_CLIENT_STATUS_TRAINING,
    );
    this.trainer_dist_adapter.update_model(global_model_params);
    this.round_idx = 0;
    await this.train();
    this.round_idx += 1;
  }

  async handle_message_receive_model_from_server(msg) {
    console.log('handle_message_receive_model_from_server.', msg);
    console.log('check the args-content: ', this.args);
    const model_params = msg.msg_params.model_params;
    console.log('handle_message_receive_model_from_server ', model_params);
    this.trainer_dist_adapter.update_model(model_params);

    if (this.round_idx < this.num_rounds) {
      await this.train();
      this.round_idx += 1;
    } else {
      this.send_client_status(0, MyMessage.MSG_MLOPS_CLIENT_STATUS_FINISHED);
      this.report_client_training_status(
        this.args.client_id_list[0],
        this.args.run_id,
        MyMessage.MSG_MLOPS_CLIENT_STATUS_FINISHED,
      );
      super.finish();
    }
  }

  handle_message_finish(msg) {
    console.log('====================cleanup ====================');
    console.log('handle_message_finish.', msg);
    this.cleanup();
  }

  cleanup() {
    this.send_client_status(0, MyMessage.MSG_MLOPS_SERVER_STATUS_FINISHED);
    this.report_client_training_status(
      this.args.client_id_list[0],
      this.args.run_id,
      MyMessage.MSG_MLOPS_SERVER_STATUS_FINISHED,
    );
    super.finish();
    // currClient.value.end();
  }

  report_client_training_status(edge_id, runId, status) {
    console.log('report_client_training_status ', runId);

    this.common_report_client_training_status(edge_id, runId, status);

    this.common_report_client_id_status(edge_id, runId, status);

    this.report_client_device_status_to_web_ui(edge_id, runId, status);
  }

  broadcast_client_training_status(edge_id, runId, status) {
    this.common_report_client_training_status(edge_id, runId, status);

    this.report_client_device_status_to_web_ui(edge_id, runId, status);
  }

  report_client_device_status_to_web_ui(edge_id, runId, status) {
    console.log('report_client_device_status_to_web_ui');
    const run_id = runId;
    const payload = {
      run_id: run_id,
      edge_id: edge_id,
      status: status,
      version: 'v1.0',
    };
    this.com_manager.mqtt_mgr.publish(
      'fl_client/mlops/status',
      JSON.stringify(payload),
      { qos: 2, retain: true },
      function (error) {
        if (error) {
          console.log(error);
        } else {
          console.log('report_client_device_status_to_web_ui Published');
        }
      },
    );
  }

  common_report_client_id_status(edge_id, runId, status) {
    console.log('report_client_id_status');
    const run_id = runId;
    const payload = {
      run_id: run_id,
      edge_id: edge_id,
      status: status,
    };
    this.com_manager.mqtt_mgr.publish(
      'fl_client/flclient_agent_' + edge_id + '/status',
      JSON.stringify(payload),
      { qos: 2, retain: true },
      function (error) {
        if (error) {
          console.log(error);
        } else {
          console.log('report_client_id_status Published');
        }
      },
    );
  }

  common_report_client_training_status(edge_id, runId, status) {
    const run_id = runId;
    const payload = {
      run_id: run_id,
      edge_id: edge_id,
      status: status,
    };
    this.com_manager.mqtt_mgr.publish(
      'fl_run/fl_client/mlops/status',
      JSON.stringify(payload),
      { qos: 2, retain: true },
      function (error) {
        if (error) {
          console.log(error);
        } else {
          console.log('common_report_client_training_status Published');
        }
      },
    );
  }

  async send_model_to_server(receive_id, weights) {
    event('comm_c2s', true, String(this.round_idx));
    const message = new Message(
      MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
      this.client_real_id,
      receive_id,
    );
    console.log('send_model_to_server');
    message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights);
    message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, 10);
    message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, 'web');
    const model_url = await this.send_message(message);
    console.log('check the model_url: ', model_url);
    await this.log_client_model_info(this.round_idx + 1, model_url);
  }

  async send_client_status(receive_id, status = 'ONLINE') {
    console.log('send_client_status ', status);
    const message = new Message(
      MyMessage.MSG_TYPE_C2S_CLIENT_STATUS,
      this.client_real_id,
      receive_id,
    );
    message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_STATUS, status);
    message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, 'web');
    await this.send_message(message);
  }

  async log_client_model_info(round_index, model_url) {
    if (model_url == 'undefined') {
      return;
    }
    console.log('model info ', model_url);
    const model_info = {
      run_id: this.args.run_id,
      edge_id: this.args.client_id_list[0],
      round_idx: round_index,
      client_model_s3_address: model_url,
    };
    const message_json = JSON.stringify(model_info);
    this.com_manager.mqtt_mgr.publish(
      'fl_server/mlops/client_model',
      message_json,
      { qos: 2, retain: true },
      function (error) {
        if (error) {
          console.log(error);
        } else {
          console.log('Client Model Published');
        }
      },
    );
  }

  async train() {
    event('train', true, String(this.round_idx));

    const weights = await this.trainer_dist_adapter.train(this.round_idx);

    event('train', false, String(this.round_idx));

    await this.send_model_to_server(0, weights);
  }

  run(): void {
    this.register_message_receive_handlers();
  }
}

export default ClientMasterManager;
