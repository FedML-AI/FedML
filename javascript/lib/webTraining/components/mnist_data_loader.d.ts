/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as tf from '@tensorflow/tfjs';
/**
 * A class that fetches the sprited MNIST dataset and returns shuffled batches.
 *
 * NOTE: This will get much easier. For now, we do data fetching and
 * manipulation manually.
 */
export declare class MnistData {
    shuffledTrainIndex: number;
    shuffledTestIndex: number;
    datasetImages: Float32Array;
    datasetLabels: Uint8Array;
    trainIndices: Uint32Array;
    testIndices: Uint32Array;
    trainImages: any;
    testImages: any;
    trainLabels: any;
    testLabels: any;
    constructor();
    load(): Promise<void>;
    nextTrainBatch(batchSize: any): {
        xs: tf.Tensor2D;
        labels: tf.Tensor2D;
    };
    nextTestBatch(batchSize: any): {
        xs: tf.Tensor2D;
        labels: tf.Tensor2D;
    };
    nextBatch(batchSize: number, data: any[], index: {
        (): number;
        (): number;
        (): any;
    }): {
        xs: tf.Tensor2D;
        labels: tf.Tensor2D;
    };
}
