import { Range } from 'immutable';

import { tf } from '../..';
import { Dataset } from '../dataset_builder';
import { DataLoader, DataConfig } from './data_loader';
import { Data, ImageData, DataSplit } from '../data';

/**
 * TODO @s314cy:
 * Load labels and correctly match them with their respective images, with the following constraints:
 * 1. Images are given as 1 image/1 file
 * 2. Labels are given as multiple labels/1 file, each label file can contain a different amount of labels
 */
export abstract class ImageLoader<Source> extends DataLoader<Source> {
  abstract readImageFrom(source: Source): Promise<tf.Tensor3D>;

  async load(image: Source, config?: DataConfig): Promise<Dataset> {
    let tensorContainer: tf.TensorContainer;
    if (config === undefined || config.labels === undefined) {
      tensorContainer = await this.readImageFrom(image);
    } else {
      tensorContainer = {
        xs: await this.readImageFrom(image),
        ys: config.labels[0],
      };
    }
    return tf.data.array([tensorContainer]);
  }

  private async buildDataset(
    images: Source[],
    labels: number[],
    indices: number[],
    config?: DataConfig,
  ): Promise<Data> {
    const dataset: tf.data.Dataset<tf.TensorContainer> = tf.data.generator(() => {
      const withLabels = config?.labels !== undefined;

      let index = 0;
      const iterator = {
        next: async () => {
          if (index === indices.length) {
            return { done: true };
          }
          const sample = await this.readImageFrom(images[indices[index]]);
          const label = withLabels ? labels[indices[index]] : undefined;
          const value = withLabels ? { xs: sample, ys: label } : sample;

          index++;

          return {
            value,
            done: false,
          };
        },
      };
      return iterator as unknown as Iterator<tf.Tensor>; // Lazy
    });

    return new ImageData(dataset, indices.length, this.task.trainingInformation);
  }

  async loadAll(images: Source[], config?: DataConfig): Promise<DataSplit> {
    let labels: number[] = [];
    const indices = Range(0, images.length).toArray();
    if (config?.labels !== undefined) {
      const numberOfClasses = this.task.trainingInformation?.LABEL_LIST?.length;
      if (numberOfClasses === undefined) {
        throw new Error('wanted labels but none found in task');
      }

      labels = tf
        .oneHot(tf.tensor1d(config.labels, 'int32'), numberOfClasses)
        .arraySync() as number[];
    }
    if (config?.shuffle === undefined || config?.shuffle) {
      this.shuffle(indices);
    }

    if (config?.validationSplit === undefined) {
      const dataset = await this.buildDataset(images, labels, indices, config);
      return {
        train: dataset,
        validation: dataset,
      };
    }

    const trainSize = Math.floor(images.length * (1 - config.validationSplit));

    const trainIndices = indices.slice(0, trainSize);
    const valIndices = indices.slice(trainSize);

    const trainDataset = await this.buildDataset(images, labels, trainIndices, config);
    const valDataset = await this.buildDataset(images, labels, valIndices, config);

    return {
      train: trainDataset,
      validation: valDataset,
    };
  }

  shuffle(array: number[]): void {
    for (let i = 0; i < array.length; i++) {
      const j = Math.floor(Math.random() * i);
      const swap = array[i];
      array[i] = array[j];
      array[j] = swap;
    }
  }
}
