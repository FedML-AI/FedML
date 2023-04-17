import { MLOpsStore } from '../core/mlops/mlops_init'
const mlopsStore = new MLOpsStore()
export function pre_setup(args) {
  mlopsStore.pre_setup(args)
}
export function init(args) {
  mlopsStore.init(args)
}
export function event(event_name, event_started, event_value, event_edge_id = null) {
  mlopsStore.event(event_name, event_started, event_value, event_edge_id)
}
