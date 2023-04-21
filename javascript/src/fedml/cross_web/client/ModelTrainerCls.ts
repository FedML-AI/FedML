import { ClientTrainer } from '../../core/alg_frame/client_trainer';
import * as tf from '@tensorflow/tfjs';
// import * as tfvis from '@tensorflow/tfjs-vis';

export class ModelTrainerCLS implements ClientTrainer {
  model: any;
  id: any;
  args: any;
  trainData;
  trainDataLabel;
  testData;
  testDataLabel;
  global_accuracy = 0;
  global_loss = 0;
  round = 0;
  round_acc = [];
  round_loss = [];
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
  async train() {
    const model = this.model;
    // console.log('model_cls args: ', this.args);
    // const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    // const container = { name: 'Loss and accuracy', tab: 'Training' };
    // const ftfvis_options = { callbacks: ['onEpochEnd'] };
    // TODO: fixed here hard coding
    if (this.args.dataset == 'mnist') {
      await model.fit(
        tf.reshape(this.trainData, [this.trainData.shape[0], -1]),
        this.trainDataLabel,
        {
          validationData: [
            tf.reshape(this.testData, [this.testData.shape[0], -1]),
            this.testDataLabel,
          ],
          batchSize: 10,
          epochs: 5,
          // callbacks: [tfvis.show.fitCallbacks(container, metrics, ftfvis_options)],
        },
      );
    } else if (this.args.dataset == 'cifar10') {
      await model.fit(tf.reshape(this.trainData, [500, 32, 32, 3]), this.trainDataLabel, {
        validationData: [tf.reshape(this.testData, [100, 32, 32, 3]), this.testDataLabel],
        batchSize: 10,
        epochs: 50,
        // callbacks: [tfvis.show.fitCallbacks(container, metrics, ftfvis_options)],
      });
    }
  }
  get_model_params() {
    try {
      const out = [];
      let dict = new Object();
      for (let i = 0; i < this.model.getWeights().length; i++) {
        dict = {
          model: this.model.getWeights()[i],
          params: this.model.getWeights()[i].dataSync(),
        };
        out.push(dict);
      }
      const tf_model = JSON.stringify(out);
      // const weights = this.model.getWeights();
      return tf_model;
    } catch (err) {}
  }
  set_model_params(model_parameters: any, verbose = false) {
    // Make sure the pytorch model structure has the same # layers of Tensorflow.js
    if (this.model.getWeights().length == model_parameters.length) {
      this.model.setWeights(model_parameters);
      verbose = true;
      if (verbose) {
        console.log('Updated Tensorflow model from pytorch model');
      } else {
        console.log(
          'ERROR: The model structure of pytorch and tensorflow.js is not aligned! Cannot transfer parameters accordingly.',
        );
      }
    }
  }
}
