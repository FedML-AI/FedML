import { exportBatchData } from './batch_data'
import { exportCifar10BatchData } from './batch_cifar10_data'

export async function dataLoader(args: any, client_id: any, trainBatchSize: any, testBatchSize: any) {
  let dataSet
  if (args.dataset === 'mnist')
    dataSet = await exportBatchData(args, client_id, trainBatchSize, testBatchSize)
  else if (args.dataset === 'cifar10')
    dataSet = await exportCifar10BatchData(args, client_id, trainBatchSize, testBatchSize)

  return dataSet
}
