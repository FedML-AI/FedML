import { FedMLCommManager } from '../../core/distributed/fedml_comm_manager';
import { MyMessage } from './message_define';
import { Message } from '../../core/distributed/communication/message';
import { event } from '../../mlops/mlops_init';
import { TrainerDistAdapter } from './fedml_trainer_dist_adapter';

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
  }

  register_message_receive_handlers() {
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
    console.log('fedml_runner register_message_receive_handlers ', this);
  }

  handle_message_connection_ready(msg_params) {
    console.log('handle_message_connection_ready ', this);
    this.send_client_status(0);
    console.log('fedml_runner handle_message_connection_ready ', this);
  }

  handle_message_check_status(msg_params) {
    this.send_client_status(0);
  }

  handle_message_init(msg_params) {
    console.log('handle_message_init ', msg_params);
    // const msg_params = {
    //   type: 1,
    //   sender_id: 0,
    //   receiver_id: 1,
    //   msg_params: {
    //     msg_type: 1,
    //     sender: 0,
    //     receiver: 1,
    //   },
    //   model_params: {},
    //   client_idx: 642,
    //   client_os: 'PythonClient',
    // };
    // const global_model_params = msg_params[MyMessage.MSG_ARG_KEY_MODEL_PARAMS];
    // const data_silo_index = msg_params[MyMessage.MSG_ARG_KEY_CLIENT_INDEX];
    // console.info('data_silo_index = ', data_silo_index);
    // this.trainer_dist_adapter.update_model(global_model_params);
    // this.trainer_dist_adapter.update_dataset(data_silo_index);
    // this.round_idx = 0;
    this.train();
  }

  handle_message_receive_model_from_server(msg_params){
    console.log('handle_message_receive_model_from_server.');
    // const model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS);
    // const client_index = msg_params.get(MyMessage.MSG_ARG_KEY_CLIENT_INDEX);
    // this.trainer_dist_adapter.update_model(model_params);
    // this.trainer_dist_adapter.update_dataset(client_index);
    // this.round_idx += 1;
    // this.train();
  }

  handle_message_finish(msg_params) {
    console.log('====================cleanup ====================');
    this.cleanup();
  }

  cleanup() {
    super.finish();
  }

  send_model_to_server(receive_id, weights) {
    event('comm_c2s', true, String(this.round_idx));
    const message = new Message(
      MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
      this.client_real_id,
      receive_id,
    );
    message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights);
    message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, 10);
    this.send_message(message);
  }

  send_client_status(receive_id, status = 'ONLINE') {
    console.log('send_client_status');
    const message = new Message(
      MyMessage.MSG_TYPE_C2S_CLIENT_STATUS,
      this.client_real_id,
      receive_id,
    );
    message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_STATUS, status);
    message.add_params(MyMessage.MSG_ARG_KEY_CLIENT_OS, 'mac');
    this.send_message(message);
  }

  sync_process_group(round_idx, model_params = null, client_index = null, src = 0) {
    console.log('sending round number to pg');
    const round_number = [round_idx, model_params, client_index];
  }

  private train() {
    event('train', true, String(this.round_idx));

    const weights = this.trainer_dist_adapter.train(this.round_idx);

    event('train', false, String(this.round_idx));

    this.send_model_to_server(0, weights);
  }

  run(): void {
    this.register_message_receive_handlers();
  }
}

export default ClientMasterManager;
