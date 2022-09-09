import { Dataset } from './dataset_builder';
import { TrainingInformation } from '../task/training_information';
import { getPreprocessImage } from './preprocessing';

import * as tf from '@tensorflow/tfjs';

export interface DataSplit {
  train: Data;
  validation?: Data;
}

export abstract class Data {
  readonly dataset: Dataset;
  readonly size?: number;
  protected readonly trainingInformation?: TrainingInformation;

  constructor(dataset: Dataset, size?: number, trainingInformation?: TrainingInformation) {
    this.dataset = dataset;
    this.size = size;
    this.trainingInformation = trainingInformation;
  }

  abstract batch(): Data;

  abstract preprocess(): Data;
}

export class ImageData extends Data {
  batch(): Data {
    const batchSize = this.trainingInformation?.batchSize;
    const newDataset = batchSize === undefined ? this.dataset : this.dataset.batch(batchSize);

    return new ImageData(newDataset, this.size, this.trainingInformation);
  }

  preprocess(): Data {
    let newDataset = this.dataset;
    if (this.trainingInformation !== undefined) {
      const preprocessImage = getPreprocessImage(this.trainingInformation);
      newDataset = newDataset.map((x: tf.TensorContainer) => preprocessImage(x));
    }
    return new ImageData(newDataset, this.size, this.trainingInformation);
  }
}

export class TabularData extends Data {
  batch(): Data {
    const batchSize = this.trainingInformation?.batchSize;
    const newDataset = batchSize === undefined ? this.dataset : this.dataset.batch(batchSize);

    return new TabularData(newDataset, this.size, this.trainingInformation);
  }

  preprocess(): Data {
    return this;
  }
}
