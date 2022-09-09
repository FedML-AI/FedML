import * as tf from '@tensorflow/tfjs';
import { TrainingInformation } from '../task/training_information';

type PreprocessImage = (image: tf.TensorContainer) => tf.TensorContainer;

export type Preprocessing = ImagePreprocessing;

export interface ImageTensorContainer extends tf.TensorContainerObject {
  xs: tf.Tensor3D | tf.Tensor4D;
  ys: tf.Tensor1D | number | undefined;
}

export enum ImagePreprocessing {
  Normalize = 'normalize',
  Resize = 'resize',
}

export function getPreprocessImage(info: TrainingInformation): PreprocessImage {
  const preprocessImage: PreprocessImage = (
    tensorContainer: tf.TensorContainer,
  ): tf.TensorContainer => {
    // TODO unsafe cast, tfjs does not provide the right interface
    let { xs, ys } = tensorContainer as ImageTensorContainer;
    if (info.preprocessingFunctions.includes(ImagePreprocessing.Normalize)) {
      xs = xs.div(tf.scalar(255));
    }
    if (
      info.preprocessingFunctions.includes(ImagePreprocessing.Resize) &&
      info.RESIZED_IMAGE_H !== undefined &&
      info.RESIZED_IMAGE_W !== undefined
    ) {
      xs = tf.image.resizeBilinear(xs, [info.RESIZED_IMAGE_H, info.RESIZED_IMAGE_W]);
    }
    return {
      xs,
      ys,
    };
  };
  return preprocessImage;
}
