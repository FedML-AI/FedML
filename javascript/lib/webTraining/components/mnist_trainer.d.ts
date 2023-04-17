import * as tf from '@tensorflow/tfjs';
export interface DataSet {
    trainData: {
        shape: number[];
    };
    trainDataLabel: {
        shape: number[];
    };
}
export declare function createLogisticRegression(dataset?: DataSet): tf.Sequential;
export declare function createConvModel(): tf.Sequential;
