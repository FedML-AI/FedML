export declare class FedMLTrainer {
    trainer: any;
    client_index: any;
    trainData: any;
    trainDataLabel: any;
    testData: any;
    testDataLabel: any;
    local_sample_number: null;
    train_local: null;
    test_local: null;
    device: any;
    args: any;
    constructor(client_index: any, trainData: any, trainDataLabel: any, testData: any, testDataLabel: any, device: any, args: any, model_trainer: any);
    train(round_idx: number): Promise<any>;
    update_model(weights: any): void;
}
