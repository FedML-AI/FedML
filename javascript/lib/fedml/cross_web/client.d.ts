import type { ClientTrainer } from '../core/alg_frame/client_trainer';
export declare class FedMLCrossSiloClient {
    constructor(args: {
        comm: any;
        rank: any;
        worker_num: any;
    }, device: any, dataset: any, model: any, model_trainer: ClientTrainer);
    run(): void;
}
