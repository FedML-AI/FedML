import { ClientTrainer } from '../../core/alg_frame/client_trainer';

export class ModelTrainerCLS implements ClientTrainer {
  model: any;
  id: any;
  args: any;
  trainData;
  trainDataLabel;
  testData;
  testDataLabel;
  constructor(model, trainData, trainDataLabel, testData, testDataLabel, args) {
    this.model = model;
    this.args = args;
    this.trainData = trainData;
    this.trainDataLabel = trainDataLabel;
    this.testData = testData;
    this.testDataLabel = testDataLabel;
  }
  set_id(trainer_id: any): void {
    this.id = trainer_id;
  }
  train() {
    const model = this.model;
    model.fit(this.trainData, this.trainDataLabel, {
      validationData: [this.testData, this.testDataLabel],
      batchSize: 200,
      epochs: 50,
    });
  }
  get_model_params() {
    try {
      const weights = this.model.getWeights();
      return weights;
    } catch (err) {}
  }
  set_model_params(model_parameters: any) {
    this.model.setWeights(model_parameters);
  }
}
