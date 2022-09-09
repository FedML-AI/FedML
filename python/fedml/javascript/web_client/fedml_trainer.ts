export class FedMLTrainer {
  trainer;
  client_index;
  train_data_local_num_dict;
  test_data_local_dict;
  all_train_data_num;
  train_local;
  local_sample_number;
  test_local;
  device;
  args;

  constructor(
    client_index,
    train_data_local_dict,
    train_data_local_num_dict,
    test_data_local_dict,
    train_data_num,
    device,
    args,
    model_trainer,
  ) {
    this.trainer = model_trainer;
    this.client_index = client_index;
    this.train_data_local_num_dict = train_data_local_num_dict;
    this.test_data_local_dict = test_data_local_dict;
    this.all_train_data_num = train_data_num;
    this.train_local = null;
    this.local_sample_number = null;
    this.test_local = null;

    this.device = device;
    this.args = args;
  }
}
