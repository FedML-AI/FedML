import { TrainerDistAdapter } from './fedml_trainer_dist_adapter';
import { ClientMasterManager } from './fedml_client_master_manager';

export function init_client(
  args,
  device,
  comm,
  client_rank,
  client_num,
  model,
  trainData,
  trainDataLabel,
  testData,
  testDataLabel,
  model_trainer = null,
) {
  const backend = 'MQTT_S3';
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
  );
  const client_manager = get_client_manager_master(
    args,
    trainer_dist_adapter,
    comm,
    client_rank,
    client_num,
    backend,
  );
  client_manager.run();
}

export function get_trainer_dist_adapter(
  args,
  device,
  client_rank,
  model,
  trainData,
  trainDataLabel,
  testData,
  testDataLabel,
  model_trainer,
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
  );
}

export function get_client_manager_master(
  args,
  trainer_dist_adapter,
  comm,
  client_rank,
  client_num,
  backend,
) {
  return new ClientMasterManager(
    args,
    trainer_dist_adapter,
    comm,
    client_rank,
    client_num,
    backend,
  );
}
