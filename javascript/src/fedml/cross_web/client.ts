import { toRaw } from 'vue'
import type { ClientTrainer } from '../core/alg_frame/client_trainer'
import { init_client } from './client/client_initializer'

export class FedMLCrossSiloClient {
  constructor(args: { comm: any; rank: any; worker_num: any }, device: any, dataset: any, model: any, model_trainer: ClientTrainer) {
    console.log('解构输入dataset: ', toRaw(dataset))
    const { trainData, trainDataLabel, testData, testDataLabel } = toRaw(dataset)
    console.log('解构dataset: ', trainData)
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
