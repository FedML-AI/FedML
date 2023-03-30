import { ClientTrainer } from '../core/alg_frame/client_trainer';
import { init_client } from './client/client_initializer';
import { toRaw } from 'vue';

export class FedMLCrossSiloClient {
  constructor(args, device, dataset, model, model_trainer: ClientTrainer) {
    console.log('解构输入dataset: ', toRaw(dataset));
    const { trainData, trainDataLabel, testData, testDataLabel } = toRaw(dataset);
    console.log('解构dataset: ', trainData);
    init_client(
      args,
      device,
      args.comm,
      args.rank,
      args.worker_num,
      model,
      trainData,
      trainDataLabel,
      testData,
      testDataLabel,
    );
  }
  run() {
    return;
  }
}
