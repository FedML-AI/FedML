import { FedMLTrainer } from "./fedml_trainer";

export class TrainerDistAdapter {
  device;
  client_rank;
  model;
  train_data_num;
  train_data_local_num_dict;
  train_data_local_dict;
  test_data_local_dic;
  model_trainer;
  client_index;
  trainer;
  args;

  constructor(
    args,
    device,
    client_rank,
    model,
    train_data_num,
    train_data_local_num_dict,
    train_data_local_dict,
    test_data_local_dict,
    model_trainer,
  ) {
    this.model_trainer.model = model;
    const client_index = this.client_rank - 1;
    this.model_trainer.set_id(client_index)
    console.log("Initializing Trainer")
    const trainer = this.get_trainer(
      client_index,
      train_data_local_dict,
      train_data_local_num_dict,
      test_data_local_dict,
      train_data_num,
      device,
      args,
      this.model_trainer,
    );

    this.client_index = client_index;
    this.client_rank = client_rank;
    this.device = device;
    this.trainer = trainer;
    this.args = args;
  }

  /**
   * @description: get trainer
   */
  get_trainer(client_index, train_data_local_dict, train_data_local_num_dict, test_data_local_dict, train_data_num, device, args, model_trainer) {
    return new FedMLTrainer(
      client_index,
      train_data_local_dict,
      train_data_local_num_dict,
      test_data_local_dict,
      train_data_num,
      device,
      args,
      model_trainer,
    )
  }

  train(round_idx) {
    const { weights, local_sample_num } = this.trainer.train(round_idx)
    return {
      weights: weights, 
      local_sample_num: local_sample_num,
    };
  }

  update_model(model_params) {
    this.trainer.update_model(model_params)
  }

  update_dataset(client_index=null) {
    const _client_index = client_index || this.client_index;
  }
}
