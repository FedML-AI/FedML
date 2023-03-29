import { init } from '../../fedml/fedml_init';
import { createModel } from './mnist_trainer';
import { FedMLRunner } from '../../fedml/cross_web/client/runner';
import { exportBatchData } from './batch_data';

export async function fedml_train() {
  const args = init();

  const device = 'cpu';

  const dataset = await exportBatchData(800, 200);

  const model = createModel();

  const fedml_runner = new FedMLRunner(args, device, dataset, model, null);

  fedml_runner.run();
}
