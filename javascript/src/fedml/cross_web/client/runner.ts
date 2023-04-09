import type { Rank, Sequential, Tensor } from '@tensorflow/tfjs'
import { FedMLCrossSiloClient } from '../client'

export class FedMLRunner {
  runner
  constructor(args: any, device: string, dataset: { trainData: any; trainDataLabel: Tensor<Rank>; testData: any; testDataLabel: Tensor<Rank> } | undefined, model: Sequential | undefined, client_trainer: null) {
    console.log('FedMLRunner dataset: ', dataset)
    this.runner = this._init_corss_silo_runner(args, device, dataset, model)
  }

  private _init_corss_silo_runner(args: any, device: any, dataset: any, model: any) {
    const runner = new FedMLCrossSiloClient(args, device, dataset, model, null)
    return runner
  }

  run() {
    this.runner.run()
  }
}
