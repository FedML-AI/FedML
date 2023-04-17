import { TrainerDistAdapter } from './fedml_trainer_dist_adapter';
import { ClientMasterManager } from './fedml_client_master_manager';
export declare function init_client(args: any, device: any, comm: any, client_rank: any, client_num: any, model: any, trainData: any, trainDataLabel: any, testData: any, testDataLabel: any, model_trainer?: null): void;
export declare function get_trainer_dist_adapter(args: Object, device: string, client_rank: any, model: any, trainData: any, trainDataLabel: any, testData: any, testDataLabel: any, model_trainer: null | undefined): TrainerDistAdapter;
export declare function get_client_manager_master(args: any, trainer_dist_adapter: TrainerDistAdapter, comm: null | undefined, client_rank: number | undefined, client_num: number | undefined, backend: string | undefined): ClientMasterManager;
