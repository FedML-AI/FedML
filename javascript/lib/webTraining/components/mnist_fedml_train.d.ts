type DataLoader = (args: any, client_id: any, trainBatchSize: number, testBatchSize: number) => Promise<{
    trainData: any;
    trainDataLabel: any;
    testData: any;
    testDataLabel: any;
}>;
export interface Options {
    customDataLoader?: DataLoader;
}
export declare function fedml_train(run_args: any, client_id: string | number, options?: Options): Promise<void>;
export {};
