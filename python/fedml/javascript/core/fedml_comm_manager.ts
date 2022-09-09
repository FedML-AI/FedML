
export class FedMLCommManager extends Observer {
    args;
    comm;
    rank:number;
    size;
    backend;
    com_manager;
    message_handler_dict;

    constructor(args, comm=null, rank=0, size=0, backend='MPI'){
        super();
        this.args = args;
        this.size = size;
        this.rank = Number(rank);
        this.backend = backend;
        this.comm = comm;
        this.com_manager = null;
        this.message_handler_dict = {};
        this._init_manager();

    }

    register_comm_manager(comm_manager: BaseCommunicationManager) {
        this.com_manager = comm_manager;
    }

    _init_manager(){
        if(this.backend === 'MPI') {

        }
    }
}