import { tidy } from '@tensorflow/tfjs'
import { Cifar10 } from './cifar10_data_loader'

export async function exportCifar10BatchData(args: any, client_id: string | number, trainBatchSize: number, testBatchSize: number) {
  const data = new Cifar10()
  await data.load()
  const [trainXs, trainYs] = tidy(() => {
    const d = data.nextTrainBatch(trainBatchSize)
    console.log('check the cifar10 d: ', d)
    return [d.xs.reshape([trainBatchSize, 32, 32, 3]), d.ys]
  })

  const [testXs, testYs] = tidy(() => {
    const d = data.nextTestBatch(testBatchSize)
    return [d.xs.reshape([testBatchSize, 32, 32, 3]), d.ys]
  })

  const trainDataLength = 500
  const testDataLength = 100
  console.log('check the cifar10 data args: ', args)
  // split the dataSet
  console.log('check the client_id: ', client_id)
  let client_idx = 1
  for (let i = 0; i <= args.edgeids.length - 1; i++) {
    if (client_id == args.edgeids[i])
      client_idx = i + 1
  }
  console.log('check the client_idx: ', client_idx)
  const trainDeviceData = trainXs.slice(
    [trainDataLength * (client_idx - 1), 0, 0, 0],
    [trainDataLength, 32, 32, 3],
  )
  const trainDeviceLabel = trainYs.slice(
    [trainDataLength * (client_idx - 1), 0],
    [trainDataLength, 10],
  )
  const testDeviceData = testXs.slice(
    [testDataLength * (client_idx - 1), 0, 0, 0],
    [testDataLength, 32, 32, 3],
  )
  const testDeviceLabel = testYs.slice(
    [testDataLength * (client_idx - 1), 0],
    [testDataLength, 10],
  )
  console.log('check the split device data ', trainDeviceData)
  console.log('check the split device label ', trainDeviceLabel)

  const dataSet = {
    trainData: trainDeviceData,
    trainDataLabel: trainDeviceLabel,
    testData: testDeviceData,
    testDataLabel: testDeviceLabel,
  }
  return dataSet
}
