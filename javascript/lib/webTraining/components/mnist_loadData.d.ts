import * as tf from '@tensorflow/tfjs';
/**
 * A class that fetches the sprited MNIST dataset and returns shuffled batches.
 *
 * NOTE: This will get much easier. For now, we do data fetching and
 * manipulation manually.
 */
export declare class LoadImgData {
    shuffledTrainIndex: number;
    shuffledTestIndex: number;
    src: string;
    trainIndices: Uint32Array;
    datasetImages: Float32Array;
    constructor(src: string);
    load(): Promise<unknown>;
    batch_train_data(): tf.Tensor2D;
}
