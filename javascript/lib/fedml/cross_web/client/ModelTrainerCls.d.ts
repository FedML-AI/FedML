import type { ClientTrainer } from '../../core/alg_frame/client_trainer';
export declare class ModelTrainerCLS implements ClientTrainer {
    model: any;
    id: any;
    args: any;
    trainData: any;
    trainDataLabel: any;
    testData: any;
    testDataLabel: any;
    global_accuracy: number;
    global_loss: number;
    round: number;
    round_acc: never[];
    round_loss: never[];
    constructor(model: any, trainData: any, trainDataLabel: any, testData: any, testDataLabel: any, args: any);
    set_id(trainer_id: any): void;
    train(): Promise<void>;
    get_model_params(): string | undefined;
    set_model_params(model_parameters: any, verbose?: boolean): void;
}
