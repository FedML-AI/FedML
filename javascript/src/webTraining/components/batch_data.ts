import { tidy } from '@tensorflow/tfjs';
import { MnistData } from './mnist_data_loader';

export async function exportBatchData(
  args: any,
  client_id: string | number,
  trainBatchSize: number,
  testBatchSize: number,
) {
  console.log('input args ', args);
  const data = new MnistData();
  await data.load();
  const [trainXs, trainYs] = tidy(() => {
    const d = data.nextTrainBatch(trainBatchSize);
    return [d.xs.reshape([trainBatchSize, 28, 28, 1]), d.labels];
  });

  const [testXs, testYs] = tidy(() => {
    const d = data.nextTestBatch(testBatchSize);
    return [d.xs.reshape([testBatchSize, 28, 28, 1]), d.labels];
  });

  const trainDataLength = 1000;
  const testDataLength = 200;
  console.log('check the client_id: ', client_id);
  let client_idx = 1;
  for (let i = 0; i <= args.edgeids.length - 1; i++) {
    if (client_id == args.edgeids[i]) client_idx = i + 1;
  }
  console.log('check the client_idx: ', client_idx);
  // split the dataSet
  const trainDeviceData = trainXs.slice(
    [trainDataLength * (client_idx - 1), 0, 0, 0],
    [trainDataLength, 28, 28, 1],
  );
  const trainDeviceLabel = trainYs.slice(
    [trainDataLength * (client_idx - 1), 0],
    [trainDataLength, 10],
  );
  const testDeviceData = testXs.slice(
    [testDataLength * (client_idx - 1), 0, 0, 0],
    [testDataLength, 28, 28, 1],
  );
  const testDeviceLabel = testYs.slice(
    [testDataLength * (client_idx - 1), 0],
    [testDataLength, 10],
  );

  const dataSet = {
    trainData: trainDeviceData,
    trainDataLabel: trainDeviceLabel,
    testData: testDeviceData,
    testDataLabel: testDeviceLabel,
  };
  return dataSet;
}
