import { ClientConstants } from '../../cli/edge_deployment/client_constants'

export class MLOpsStore {
  mlops_args: any
  mlops_project_id!: number
  mlops_run_id: any
  mlops_edge_id!: number
  mlops_log_metrics: any
  mlops_log_round_info: any
  mlops_log_client_training_status = ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING
  mlops_log_round_start_time: any
  mlops_log_metrics_lock: any
  mlops_log_mqtt_mgr: any
  mlops_log_mqtt_lock: any
  mlops_log_mqtt_is_connected = false
  mlops_log_agent_config: any
  mlops_metrics: any
  mlops_event!: { log_event_started: (arg0: any, arg1: any, arg2: any) => void; log_event_ended: (arg0: any, arg1: any, arg2: any) => void }
  mlops_bind_result = false
  server_agent_id: any
  current_parrot_process: any

  pre_setup(args: object) {
    this.mlops_args = args
  }

  init(args: object) {
    this.mlops_args = args
  }

  event(event_name: any, event_started = true, event_value: any, event_edge_id: null) {
    if (!this.mlops_enabled(this.mlops_args))
      return

    this.set_realtime_params()

    // set_log_mqtt_mgr();

    if (!this.mlops_bind_result)
      return

    if (event_started)
      this.mlops_event.log_event_started(event_name, event_value, event_edge_id)
    else
      this.mlops_event.log_event_ended(event_name, event_value, event_edge_id)
  }

  set_realtime_params() {
    this.mlops_bind_result = true
    if (this.mlops_args) {
      this.mlops_run_id = this.mlops_args.run_id
      if (this.mlops_args != null && this.mlops_args.client_id != null)
        this.mlops_edge_id = this.mlops_args.client_id
      else if (this.mlops_args.client_id_list != null)
        this.mlops_edge_id = this.mlops_args.client_id_list
      else
        this.mlops_edge_id = 0

      if (this.mlops_args.server_agent_id != null)
        this.server_agent_id = this.mlops_args.server_agent_id
      else
        this.server_agent_id = this.mlops_edge_id
    }
    return true
  }

  mlops_enabled(args: any): Boolean {
    return args && args.using_mlops != undefined && args.using_mlops != null
  }
}
