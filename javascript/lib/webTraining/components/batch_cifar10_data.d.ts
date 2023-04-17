import * as tf from '@tensorflow/tfjs';
export declare function exportCifar10BatchData(args: any, client_id: string | number, trainBatchSize: number, testBatchSize: number): Promise<{
    trainData: any;
    trainDataLabel: tf.Tensor<tf.Rank>;
    testData: any;
    testDataLabel: tf.Tensor<tf.Rank>;
}>;
