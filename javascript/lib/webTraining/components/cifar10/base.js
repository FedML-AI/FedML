import * as tf from '@tensorflow/tfjs'
export class DataSet {
  constructor() {
    this.IMG_WIDTH = 32
    this.IMG_HEIGHT = 32
    this.NUM_CLASSES = 10
    this.DATA_PRE_NUM = 10000
    this.IMAGE_SIZE = this.IMG_WIDTH * this.IMG_HEIGHT * 3
    this.trainM = 0
    this.testM = 0
    this.shuffledTrainIndex = 0
    this.shuffledTestIndex = 0
    this.currentTrainIndex = 0
  }

  getPath(_src) { }
  loadImg(_src) { }
  loadImages(_srcs) { }
  async load() { }
  nextBatch(batchSize, dataType, [data, lables], index) {
    const batchImagesArray = new Float32Array(batchSize * this.IMAGE_SIZE)
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const batchLabelsArray = new Uint8Array(batchSize * this.NUM_CLASSES)
    const batchLables = []
    console.log('check the ready slice ', data)
    for (let i = 0; i < batchSize; i++) {
      const idx = index()
      const currentIdx = idx % this.DATA_PRE_NUM
      const dataIdx = Math.floor(idx / this.DATA_PRE_NUM)
      let image
      if (dataType === 'train') {
        image = data[dataIdx].slice(currentIdx * this.IMAGE_SIZE, currentIdx * this.IMAGE_SIZE + this.IMAGE_SIZE)
        batchImagesArray.set(image, i * this.IMAGE_SIZE)
        batchLables.push(lables[idx])
      }
      else if (dataType === 'test') {
        image = data[0].slice(currentIdx * this.IMAGE_SIZE, currentIdx * this.IMAGE_SIZE + this.IMAGE_SIZE)
        batchImagesArray.set(image, i * this.IMAGE_SIZE)
        batchLables.push(lables[idx])
      }
    }
    const xs = tf.tensor2d(batchImagesArray, [batchSize, this.IMAGE_SIZE])
    const ys = tf.oneHot(batchLables, this.NUM_CLASSES)
    return { xs, ys }
  }

  nextTrainBatch(batchSize = this.trainM) {
    this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.trainIndices.length
    return this.nextBatch(batchSize, 'train', [this.trainDatas, this.trainLables], () => {
      this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.trainIndices.length
      return this.trainIndices[this.shuffledTrainIndex]
    })
  }

  nextTestBatch(batchSize = this.testM) {
    this.shuffledTestIndex = (this.shuffledTestIndex + 1) % this.testIndices.length
    return this.nextBatch(batchSize, 'test', [this.testDatas, this.testLables], () => {
      this.shuffledTestIndex = (this.shuffledTestIndex + 1) % this.testIndices.length
      return this.testIndices[this.shuffledTestIndex]
    })
  }
}
