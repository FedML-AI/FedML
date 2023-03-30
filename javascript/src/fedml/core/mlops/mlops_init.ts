import { ClientConstants } from '../../cli/edge_deployment/client_constants';
export class MLOpsStore {
  mlops_args: Object;
  mlops_project_id: Number;
  mlops_run_id;
  mlops_edge_id;
  mlops_log_metrics: Object;
  mlops_log_round_info: Object;
  mlops_log_client_training_status = ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING;
  mlops_log_round_start_time;
  mlops_log_metrics_lock;
  mlops_log_mqtt_mgr;
  mlops_log_mqtt_lock;
  mlops_log_mqtt_is_connected = false;
  mlops_log_agent_config: Object;
  mlops_metrics;
  mlops_event;
  mlops_bind_result = false;
  server_agent_id;
  current_parrot_process;

  pre_setup(args) {
    this.mlops_args = args;
  }

  init(args) {
    this.mlops_args = args;
  }

  event(event_name, event_started = true, event_value, event_edge_id) {
    if (!this.mlops_enabled(this.mlops_args)) {
      return;
    }

    this.set_realtime_params();

    // set_log_mqtt_mgr();

    if (!this.mlops_bind_result) {
      return;
    }
    if (event_started) {
      this.mlops_event.log_event_started(event_name, event_value, event_edge_id);
    } else {
      this.mlops_event.log_event_ended(event_name, event_value, event_edge_id);
    }
  }

  set_realtime_params() {
    this.mlops_bind_result = true;
    if (this.mlops_args != null) {
      this.mlops_run_id = this.mlops_args.run_id;
      if (this.mlops_args != null && this.mlops_args.client_id != null) {
        this.mlops_edge_id = this.mlops_args.client_id;
      } else if (this.mlops_args.client_id_list != null) {
        this.mlops_edge_id = this.mlops_args.client_id_list;
      } else {
        this.mlops_edge_id = 0;
      }

      if (this.mlops_args.server_agent_id != null) {
        this.server_agent_id = this.mlops_args.server_agent_id;
      } else {
        this.server_agent_id = this.mlops_edge_id;
      }
    }
    return true;
  }

  mlops_enabled(args): Boolean {
    if (args.using_mlops != undefined && args.using_mlops != null) {
      return true;
    } else {
      return false;
    }
  }
}
