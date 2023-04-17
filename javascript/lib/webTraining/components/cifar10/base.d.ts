import * as tf from '@tensorflow/tfjs';
export declare class DataSet {
    readonly IMG_WIDTH: number;
    readonly IMG_HEIGHT: number;
    readonly TRAIN_IMAGES: string[];
    readonly TRAIN_LABLES: string | number[];
    readonly TEST_IMAGES: string[];
    readonly TEST_LABLES: string | number[];
    readonly NUM_CLASSES: number;
    readonly DATA_PRE_NUM: number;
    readonly IMAGE_SIZE: number;
    trainDatas: Float32Array[];
    testDatas: Float32Array[];
    trainLables: number[];
    testLables: number[];
    trainM: number;
    testM: number;
    trainIndices: Uint32Array;
    testIndices: Uint32Array;
    shuffledTrainIndex: number;
    shuffledTestIndex: number;
    currentTrainIndex: number;
    getPath(_src: string): void;
    loadImg(_src: string): void;
    loadImages(_srcs: string[]): void;
    load(): Promise<void>;
    nextBatch(batchSize: number, dataType: string, [data, lables]: [Float32Array[], number[]], index: Function): {
        xs: tf.Tensor2D;
        ys: tf.Tensor<tf.Rank>;
    };
    nextTrainBatch(batchSize?: number): {
        xs: tf.Tensor2D;
        ys: tf.Tensor<tf.Rank>;
    };
    nextTestBatch(batchSize?: number): {
        xs: tf.Tensor2D;
        ys: tf.Tensor<tf.Rank>;
    };
}
