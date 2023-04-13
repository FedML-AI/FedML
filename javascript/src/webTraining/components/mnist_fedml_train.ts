import { init } from '../../fedml/fedml_init'
import { FedMLRunner } from '../../fedml/cross_web/client/runner'
import { createModel } from './model'
import { dataLoader } from './dataloader'

type DataLoader = (
  args: any,
  client_id: any,
  trainBatchSize: number,
  testBatchSize: number
) => Promise<{
  trainData: any
  trainDataLabel: any
  testData: any
  testDataLabel: any
}>

export interface Options {
  customDataLoader?: DataLoader
}

export async function fedml_train(
  run_args: any,
  client_id: string | number,
  options?: Options,
) {
  const args = init(run_args, client_id)

  const device = 'cpu'

  let _dataLoader = dataLoader

  if (options && options?.customDataLoader)
    _dataLoader = options.customDataLoader

  const dataset = await _dataLoader(args, client_id, 3000, 600)

  const model = createModel(args, dataset)

  const fedml_runner = new FedMLRunner(args, device, dataset, model, null)

  fedml_runner.run()
}
