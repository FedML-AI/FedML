import { TrainerDistAdapter } from './fedml_trainer_dist_adapter'
import { ClientMasterManager } from './fedml_client_master_manager'

export function init_client(
  args: any,
  device: any,
  comm: any,
  client_rank: any,
  client_num: any,
  model: any,
  trainData: any,
  trainDataLabel: any,
  testData: any,
  testDataLabel: any,
  model_trainer = null,
) {
  const backend = 'MQTT_S3'
  const trainer_dist_adapter = get_trainer_dist_adapter(
    args,
    device,
    client_rank,
    model,
    trainData,
    trainDataLabel,
    testData,
    testDataLabel,
    model_trainer,
  )
  const client_manager = get_client_manager_master(
    args,
    trainer_dist_adapter,
    comm,
    client_rank,
    client_num,
    backend,
  )
  client_manager.run()
}

export function get_trainer_dist_adapter(
  args: Object,
  device: string,
  client_rank: any,
  model: any,
  trainData: any,
  trainDataLabel: any,
  testData: any,
  testDataLabel: any,
  model_trainer: null | undefined,
) {
  return new TrainerDistAdapter(
    args,
    device,
    client_rank,
    model,
    trainData,
    trainDataLabel,
    testData,
    testDataLabel,
    model_trainer,
  )
}

export function get_client_manager_master(
  args: any,
  trainer_dist_adapter: TrainerDistAdapter,
  comm: null | undefined,
  client_rank: number | undefined,
  client_num: number | undefined,
  backend: string | undefined,
) {
  return new ClientMasterManager(
    args,
    trainer_dist_adapter,
    comm,
    client_rank,
    client_num,
    backend,
  )
}
