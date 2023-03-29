import { MnistData } from './mnist_data_loader';
import * as tf from '@tensorflow/tfjs';

export async function exportBatchData(trainBatchSize, testBatchSize) {
  const data = new MnistData();
  await data.load();
  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(trainBatchSize);
    return [d.xs.reshape([trainBatchSize, 28, 28, 1]), d.labels];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(testBatchSize);
    return [d.xs.reshape([testBatchSize, 28, 28, 1]), d.labels];
  });
  const dataSet = {
    trainData: trainXs,
    trainDataLabel: trainYs,
    testData: testXs,
    testDataLabel: testYs,
  };
  return dataSet;
}
