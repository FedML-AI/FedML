import * as tf from '@tensorflow/tfjs'
import { DataSet } from './cifar10/base'

import train_lables from './cifar10/train_labels.json'
const data_batch_1 = 'https://fedml.s3.us-west-1.amazonaws.com/data_batch_1.png'
const data_batch_2 = 'https://fedml.s3.us-west-1.amazonaws.com/data_batch_2.png'
const data_batch_3 = 'https://fedml.s3.us-west-1.amazonaws.com/data_batch_3.png'
const data_batch_4 = 'https://fedml.s3.us-west-1.amazonaws.com/data_batch_4.png'
const data_batch_5 = 'https://fedml.s3.us-west-1.amazonaws.com/data_batch_5.png'
const test_batch = 'https://fedml.s3.us-west-1.amazonaws.com/test_batch.png'

const test_lables = JSON.parse(JSON.stringify(train_lables))

export class Cifar10 extends DataSet {
  TRAIN_IMAGES = [data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5]
  TRAIN_LABLES = train_lables
  TEST_IMAGES = [test_batch]
  TEST_LABLES = test_lables

  loadImg(src: string): Promise<Float32Array> {
    return new Promise((resolve, reject) => {
      const img = new Image()
      img.crossOrigin = 'Anonymous'
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d') as CanvasRenderingContext2D
      img.src = src
      img.onload = () => {
        canvas.width = img.naturalWidth
        canvas.height = img.naturalHeight

        const datasetBytesBuffer = new ArrayBuffer(canvas.width * canvas.height * 3 * 4)
        const datasetBytesView = new Float32Array(datasetBytesBuffer)

        ctx.drawImage(img, 0, 0, canvas.width, canvas.height, 0, 0, canvas.width, canvas.height)
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
        for (let j = 0, i = 0; j < imageData.data.length; j++) {
          if ((j + 1) % 4 === 0)
            continue
          datasetBytesView[i++] = imageData.data[j] / 255
        }

        resolve(datasetBytesView)
      }
      img.onerror = reject
    })
  }

  loadImages(srcs: string[]): Promise<Float32Array[]> {
    return Promise.all(srcs.map(this.loadImg))
    // .then(async imgsBytesView => imgsBytesView
    //   .reduce((preView, currentView) => this.float32Concat(preView, currentView)))
  }

  async load() {
    this.trainDatas = await this.loadImages(this.TRAIN_IMAGES)
    this.testDatas = await this.loadImages(this.TEST_IMAGES)

    this.trainLables = this.TRAIN_LABLES
    this.testLables = this.TEST_LABLES

    this.trainM = this.trainLables.length
    this.testM = this.testLables.length

    // Create shuffled indices into the train/test set for when we select a
    // random dataset element for training / validation.
    this.trainIndices = tf.util.createShuffledIndices(this.trainM)
    this.testIndices = tf.util.createShuffledIndices(this.testM)
  }
}
