import type { ClientTrainer } from '../core/alg_frame/client_trainer'
import { init_client } from './client/client_initializer'

export class FedMLCrossSiloClient {
  constructor(args: { comm: any; rank: any; worker_num: any }, device: any, dataset: any, model: any, model_trainer: ClientTrainer) {
    const { trainData, trainDataLabel, testData, testDataLabel } = dataset
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
    )
  }

  run() {

  }
}
