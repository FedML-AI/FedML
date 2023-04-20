import { pre_setup } from './mlops/mlops_init'
import { Arguments } from './arguments'
// import type { RunArgsJSON } from './arguments'

let _global_training_type: null | undefined = null
let _global_comm_backend: null | undefined = null

export function init(run_arguments: any, client_id: any) {
  console.log('check the run_args: ', run_arguments)
  let args: any
  if (!args) {
    args = Arguments.load_arguments(
      _global_training_type,
      _global_comm_backend,
    )
  }

  _global_training_type = args.training_type
  _global_comm_backend = args.backend

  pre_setup(args)

  args.scenario = 'horizontal'
  init_cross_silo_horizontal(args, client_id)
  args.run_id = 0
  args.server_id = 0
  if (run_arguments.runId > 0) {
    args.run_id = run_arguments.runId
    args.server_id = run_arguments.server_id
    args.comm_round = run_arguments.run_config.parameters.train_args.comm_round
    args.currentEdgeId = run_arguments.currentEdgeId
    args.edgeids = run_arguments.edgeids
    args.client_id_list = [client_id]
  }
  console.log('check the init server_id: ', args.server_id)
  console.log('check the comm_round: ', args.comm_round)
  console.log('chech the currentEdgeId: ', args.currentEdgeId)
  args.dataset = run_arguments.run_config.parameters.data_args.dataset
  // args.data_args.dataset = run_arguments.run_config.parameters.data_args.dataset;
  return args
}

export function init_cross_silo_horizontal(
  args: {
    n_proc_in_silo?: number
    proc_rank_in_silo?: number
    rank?: any
    process_id?: any
    client_id_list?: any[]
  },
  client_id: string | number,
) {
  args.n_proc_in_silo = 1
  args.proc_rank_in_silo = 0
  //   待处理
  args.rank = client_id
  args.process_id = client_id
  args.rank = client_id
  args.client_id_list = [client_id]
  return args
}
