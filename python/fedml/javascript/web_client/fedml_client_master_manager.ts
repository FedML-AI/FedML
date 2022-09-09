import FedMLCommManager;
import {MyMessage} from './message_define';

export class ClientMasterManager extends FedMLCommManager {
    args;
    trainer_dist_adapter;
    num_rounds;
    round_idx;
    rank;
    client_real_ids;
    client_real_id;
    has_sent_online_msg;

    constructor(args, trainer_dist_adapter, comm=null, rank=0, size=0, backend="MPI"){
        super(args, comm, rank, size, backend);

        this.trainer_dist_adapter = trainer_dist_adapter;
        this.args = args;
        this.num_rounds = args.comm_round;
        this.round_idx = 0;
        this.rank = rank;
        this.client_real_ids = JSON.parse(args.client_id_list);
        this.client_real_id = this.client_real_ids[0];
        this.has_sent_online_msg = false;
    }

    register_message_receive_handlers(){
        this.register_message_receive_handlers(
            MyMessage.MSG_TYPE_CONNECTION_IS_READY, this.handle_message_connection_ready
        )
    }
}