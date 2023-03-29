import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

export const mnist_trainer = function () {
  const model = createModel();
  function doTrain(xs, ys, txs, tys) {
    clearTsvisTab();
    model.fit(xs, ys, {
      validationData: [txs, tys],
      batchSize: 200,
      epochs: 50,
      callbacks: tfvis.show.fitCallbacks({ name: 'training_effect' }, ['loss', 'acc'], {
        callbacks: ['onEpochEnd'],
      }),
    });
  }
  function clearTsvisTab() {
    const tabConEl = document.querySelector('#tfjs-visor-container');
    if (tabConEl) {
      document.body.removeChild(tabConEl);
    }
  }
  async function saveModel() {
    const res = await model.save('localstorage://model');
    console.log(res);
  }
  return { model, doTrain, saveModel };
};

// create training model
export function createModel() {
  const model = tf.sequential();
  model.add(
    tf.layers.conv2d({ inputShape: [28, 28, 1], kernelSize: 3, filters: 16, activation: 'relu' }),
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
  model.add(tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: 'relu' }));
  model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
  model.add(tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: 'relu' }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));
  // const model = tf.sequential();
  // model.add(
  //   tf.layers.conv2d({
  //     inputShape: [28, 28, 1],
  //     kernelSize: 5,
  //     filters: 16,
  //     strides: 1,
  //     activation: 'relu',
  //     kernelInitializer: 'varianceScaling',
  //   }),
  // );
  // model.add(
  //   tf.layers.maxPool2d({
  //     poolSize: [2, 2],
  //     strides: [2, 2],
  //   }),
  // );
  // model.add(
  //   tf.layers.conv2d({
  //     kernelSize: 5,
  //     filters: 16,
  //     strides: 1,
  //     activation: 'relu',
  //     kernelInitializer: 'varianceScaling',
  //   }),
  // );
  // model.add(
  //   tf.layers.maxPool2d({
  //     poolSize: [2, 2],
  //     strides: [2, 2],
  //   }),
  // );
  // model.add(tf.layers.flatten());
  // model.add(
  //   tf.layers.dense({
  //     units: 10,
  //     activation: 'softmax',
  //     kernelInitializer: 'varianceScaling',
  //   }),
  // );
  model.compile({
    optimizer: tf.train.sgd(0.01),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });
  return model;
}
