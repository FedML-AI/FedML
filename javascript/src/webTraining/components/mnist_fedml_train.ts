import { init } from '../../fedml/fedml_init';
import { createModel } from './model';
import { FedMLRunner } from '../../fedml/cross_web/client/runner';
import { useAppStore } from '/@/store/modules/app';
// import { computed } from 'vue';
import { dataLoader } from './dataloader';
const appStore = useAppStore();

export async function fedml_train(run_args, client_id) {
  const args = init(run_args, client_id);

  const device = 'cpu';

  if (args?.dataset == 'cifar10') {
    const dataset = await dataLoader(args, client_id, 3000, 600);

    const model = createModel(args, dataset);

    const fedml_runner = new FedMLRunner(args, device, dataset, model, null);

    fedml_runner.run();
  } else {
    // const dataset = computed(() => appStore.getDeviceDataSet);

    const model = createModel(args, appStore.getDeviceDataSet);

    const fedml_runner = new FedMLRunner(args, device, appStore.getDeviceDataSet, model, null);

    fedml_runner.run();
  }
}
