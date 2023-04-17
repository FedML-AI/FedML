export declare function dataLoader(args: any, client_id: any, trainBatchSize: any, testBatchSize: any): Promise<{
    trainData: any;
    trainDataLabel: import("@tensorflow/tfjs-core").Tensor<import("@tensorflow/tfjs-core").Rank>;
    testData: any;
    testDataLabel: import("@tensorflow/tfjs-core").Tensor<import("@tensorflow/tfjs-core").Rank>;
} | undefined>;
