export abstract class ClientTrainer {
  model;
  id;
  args;
  constructor(model, args) {
    this.model = model;
    this.id = 0;
    this.args = args;
  }

  set_id(trainer_id) {
    this.id = trainer_id;
  }

  abstract get_model_params();

  abstract set_model_params(model_parameters);

  abstract train(train_data, device, args);
}
