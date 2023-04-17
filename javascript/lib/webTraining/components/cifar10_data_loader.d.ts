import { DataSet } from './cifar10/base';
export declare class Cifar10 extends DataSet {
    TRAIN_IMAGES: string[];
    TRAIN_LABLES: number[];
    TEST_IMAGES: string[];
    TEST_LABLES: any;
    loadImg(src: string): Promise<Float32Array>;
    loadImages(srcs: string[]): Promise<Float32Array[]>;
    load(): Promise<void>;
}
