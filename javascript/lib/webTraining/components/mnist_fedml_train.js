import { init } from '../../fedml/fedml_init'
import { FedMLRunner } from '../../fedml/cross_web/client/runner'
import { createModel } from './model'
import { dataLoader } from './dataloader'
export async function fedml_train(run_args, client_id, options) {
  const args = init(run_args, client_id)
  const device = 'cpu'
  let _dataLoader = dataLoader
  if (options && (options === null || options === void 0 ? void 0 : options.customDataLoader))
    _dataLoader = options.customDataLoader
  const dataset = await _dataLoader(args, client_id, 3000, 600)
  const model = createModel(args, dataset)
  const fedml_runner = new FedMLRunner(args, device, dataset, model, null)
  fedml_runner.run()
}
