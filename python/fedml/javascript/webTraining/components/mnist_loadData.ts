import * as tf from '@tensorflow/tfjs';

const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;

const NUM_DATASET_ELEMENTS = 1;

const NUM_TRAIN_ELEMENTS = NUM_DATASET_ELEMENTS;

/**
 * A class that fetches the sprited MNIST dataset and returns shuffled batches.
 *
 * NOTE: This will get much easier. For now, we do data fetching and
 * manipulation manually.
 */
export class LoadImgData {
  shuffledTrainIndex;
  shuffledTestIndex;
  src;
  trainIndices;
  datasetImages;

  constructor(src) {
    this.shuffledTrainIndex = 0;
    this.shuffledTestIndex = 0;
    this.src = src;
  }

  async load() {
    // Make a request for the MNIST sprited image.
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    // document.body.appendChild(canvas);
    this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
    return new Promise((resolve, reject) => {
      img.crossOrigin = '';
      img.onload = () => {
        img.width = img.naturalWidth;
        img.height = img.naturalHeight;
        const datasetBytesBuffer = new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);
        const chunkSize = 1;
        canvas.width = img.width;
        canvas.height = img.height;
        for (let i = 0; i < 1; i++) {
          const datasetBytesView = new Float32Array(
            datasetBytesBuffer,
            i * IMAGE_SIZE * chunkSize * 4,
            IMAGE_SIZE * chunkSize,
          );
          ctx.drawImage(img, 0, 0, img.width, img.height, 0, 0, img.width, img.height);
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          for (let j = 0; j < imageData.data.length / 4; j++) {
            // All channels hold an equal value since the image is grayscale, so
            // just read the red channel.
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
        }
        this.datasetImages = new Float32Array(datasetBytesBuffer);
        const data = this.batch_train_data();
        resolve(data);
      };
      img.src = this.src;
    });
  }

  batch_train_data() {
    const batchImagesArray = new Float32Array(1 * IMAGE_SIZE);
    const idx = () => {
      this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.trainIndices.length;
      return this.trainIndices[this.shuffledTrainIndex];
    };
    const image = this.datasetImages.slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
    batchImagesArray.set(image, 1 * IMAGE_SIZE);
    const xs = tf.tensor2d(batchImagesArray, [1, IMAGE_SIZE]);
    return xs;
  }
}
