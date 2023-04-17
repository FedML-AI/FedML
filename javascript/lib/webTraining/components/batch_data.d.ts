import * as tf from '@tensorflow/tfjs';
export declare function exportBatchData(args: any, client_id: string | number, trainBatchSize: number, testBatchSize: number): Promise<{
    trainData: any;
    trainDataLabel: tf.Tensor2D;
    testData: any;
    testDataLabel: tf.Tensor2D;
}>;
