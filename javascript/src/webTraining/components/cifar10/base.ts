import * as tf from '@tensorflow/tfjs'

export class DataSet {
  readonly IMG_WIDTH: number = 32
  readonly IMG_HEIGHT: number = 32
  readonly TRAIN_IMAGES!: string[]
  readonly TRAIN_LABLES!: string | number[]
  readonly TEST_IMAGES!: string[]
  readonly TEST_LABLES!: string | number[]
  readonly NUM_CLASSES: number = 10
  readonly DATA_PRE_NUM: number = 10000
  readonly IMAGE_SIZE: number = this.IMG_WIDTH * this.IMG_HEIGHT * 3

  // get IMAGE_SIZE() {
  //   return this.IMG_WIDTH * this.IMG_HEIGHT * 3;
  // }

  trainDatas!: Float32Array[]
  testDatas!: Float32Array[]
  trainLables!: number[]
  testLables!: number[]

  trainM = 0
  testM = 0
  trainIndices!: Uint32Array
  testIndices!: Uint32Array
  shuffledTrainIndex = 0
  shuffledTestIndex = 0

  currentTrainIndex = 0

  getPath(_src: string) {}

  loadImg(_src: string) {}

  loadImages(_srcs: string[]) {}

  async load() {}

  nextBatch(
    batchSize: number,
    dataType: string,
    [data, lables]: [Float32Array[], number[]],
    index: Function,
  ) {
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
        image = data[dataIdx].slice(
          currentIdx * this.IMAGE_SIZE,
          currentIdx * this.IMAGE_SIZE + this.IMAGE_SIZE,
        )
        batchImagesArray.set(image, i * this.IMAGE_SIZE)
        batchLables.push(lables[idx])
      }
      else if (dataType === 'test') {
        image = data[0].slice(
          currentIdx * this.IMAGE_SIZE,
          currentIdx * this.IMAGE_SIZE + this.IMAGE_SIZE,
        )
        batchImagesArray.set(image, i * this.IMAGE_SIZE)
        batchLables.push(lables[idx])
      }
    }
    const xs = tf.tensor2d(batchImagesArray, [batchSize, this.IMAGE_SIZE])
    const ys = tf.oneHot(batchLables, this.NUM_CLASSES)

    return { xs, ys }
  }

  nextTrainBatch(batchSize: number = this.trainM) {
    this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.trainIndices.length

    return this.nextBatch(batchSize, 'train', [this.trainDatas, this.trainLables], () => {
      this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.trainIndices.length
      return this.trainIndices[this.shuffledTrainIndex]
    })
  }

  nextTestBatch(batchSize: number = this.testM) {
    this.shuffledTestIndex = (this.shuffledTestIndex + 1) % this.testIndices.length

    return this.nextBatch(batchSize, 'test', [this.testDatas, this.testLables], () => {
      this.shuffledTestIndex = (this.shuffledTestIndex + 1) % this.testIndices.length
      return this.testIndices[this.shuffledTestIndex]
    })
  }
}
