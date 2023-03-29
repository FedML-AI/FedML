import { FedMLCrossSiloClient } from '../client';

export class FedMLRunner {
  runner;
  constructor(args, device, dataset, model, client_trainer) {
    console.log('FedMLRunner dataset: ', dataset);
    this.runner = this._init_corss_silo_runner(args, device, dataset, model);
  }

  private _init_corss_silo_runner(args, device, dataset, model) {
    const runner = new FedMLCrossSiloClient(args, device, dataset, model, null);
    return runner;
  }

  run() {
    this.runner.run();
  }
}
