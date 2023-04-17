import type { Rank, Sequential, Tensor } from '@tensorflow/tfjs';
import { FedMLCrossSiloClient } from '../client';
export declare class FedMLRunner {
    runner: FedMLCrossSiloClient;
    constructor(args: any, device: string, dataset: {
        trainData: any;
        trainDataLabel: Tensor<Rank>;
        testData: any;
        testDataLabel: Tensor<Rank>;
    } | undefined, model: Sequential | undefined, client_trainer: null);
    private _init_corss_silo_runner;
    run(): void;
}
