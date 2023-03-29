import { pre_setup } from './mlops/mlops_init';
import { Arguments } from './arguments';

let _global_training_type = null;
let _global_comm_backend = null;

export function init() {
  let args: Object;
  if (args == null) {
    args = Arguments.load_arguments(_global_training_type, _global_comm_backend);
  }
  _global_training_type = args.training_type;
  _global_comm_backend = args.backend;

  pre_setup(args);

  args.scenario = 'horizontal';
  init_cross_silo_horizontal(args);
  return args;
}

export function init_cross_silo_horizontal(args) {
  args.n_proc_in_silo = 1;
  args.proc_rank_in_silo = 0;
  //   待处理
  args.rank = 1;
  args.process_id = 1;
  args.rank = 1;
  args.client_id_list = [1];
  return args;
}
