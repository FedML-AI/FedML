export class FedMLTrainer {
  trainer;
  client_index;
  trainData: any;
  trainDataLabel: any;
  testData: any;
  testDataLabel: any;
  local_sample_number;
  train_local;
  test_local;
  device;
  args;

  constructor(
    client_index,
    trainData,
    trainDataLabel,
    testData,
    testDataLabel,
    device,
    args,
    model_trainer,
  ) {
    this.trainer = model_trainer;
    this.client_index = client_index;
    this.trainData = trainData;
    this.trainDataLabel = trainDataLabel;
    this.testData = testData;
    this.testDataLabel = testDataLabel;
    this.train_local = null;
    this.local_sample_number = null;
    this.test_local = null;
    this.device = device;
    this.args = args;
  }

  train(round_idx = null) {
    this.args.round_idx = round_idx;
    this.trainer.train();
    const weights = this.trainer.get_model_params();
    return weights;
  }

  update_model(weights) {
    this.trainer.set_model_params(weights);
  }
}
