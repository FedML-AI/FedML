import { FedMLTrainer } from './fedml_trainer';
import { ModelTrainerCLS } from './ModelTrainerCls';
export declare class TrainerDistAdapter {
    device: string;
    client_rank: any;
    model: any;
    trainData: any;
    trainDataLabel: any;
    testData: any;
    testDataLabel: any;
    model_trainer: ModelTrainerCLS | undefined;
    client_index: number;
    trainer: FedMLTrainer;
    args: Object;
    constructor(args: Object, device: string, client_rank: any, model: any, trainData: any, trainDataLabel: any, testData: any, testDataLabel: any, model_trainer?: null);
    /**
     * @description: get trainer
     */
    get_trainer(client_index: number, trainData: any, trainDataLabel: any, testData: any, testDataLabel: any, device: any, args: any, model_trainer: ModelTrainerCLS | undefined): FedMLTrainer;
    train(round_idx: number | null | undefined): Promise<{
        weights: any;
    } | undefined>;
    update_model(model_params: any): void;
    update_dataset(client_index: number): void;
    create_model_trainer(model: any, trainData: any, trainDataLabel: any, testData: any, testDataLabel: any, args: any): ModelTrainerCLS;
}
