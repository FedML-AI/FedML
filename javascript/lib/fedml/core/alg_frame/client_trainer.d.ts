export declare abstract class ClientTrainer {
    model: any;
    id: number;
    args: any;
    constructor(model: any, args: any);
    set_id(trainer_id: any): void;
    abstract get_model_params(): any;
    abstract set_model_params(model_parameters: any): any;
    abstract train(train_data: any, device: any, args: any): any;
}
