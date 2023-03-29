import * as tf from '@tensorflow/tfjs';

// create logisticRegression training model
export function createLogisticRegression(dataset) {
  console.log('check the create model data ', dataset);
  const dataDim = dataset.trainData.shape;
  let inputShape = 1;
  for (let index = 1; index < dataDim.length - 1; index++) {
    inputShape *= dataset.trainData.shape[index];
  }
  let outputShape = 1;
  if (dataset.trainDataLabel.shape.length != 1) {
    outputShape = dataset.trainDataLabel.shape.at(-1);
  }
  // console.log('outputShape', dataset.trainDataLabel.shape);

  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [inputShape],
      units: outputShape,
      activation: 'sigmoid',
    }),
  );
  model.compile({
    optimizer: tf.train.sgd(0.03),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });
  return model;
}

export function createConvModel() {
  // Creating CIFAR-10 cnn model
  const model = tf.sequential();
  model.add(
    tf.layers.conv2d({
      inputShape: [32, 32, 3], // picture size
      kernelSize: [5, 5],
      filters: 6, // out_channels in pytorch
      strides: (1, 1),
      activation: 'relu',
      kernelInitializer: 'varianceScaling',
    }),
  );
  model.add(
    tf.layers.maxPool2d({
      poolSize: 2,
      strides: 2,
    }),
  );
  model.add(
    tf.layers.conv2d({
      // inputShape: [14, 14, 6],
      kernelSize: [5, 5],
      filters: 16,
      strides: (1, 1),
      activation: 'relu',
      kernelInitializer: 'varianceScaling',
    }),
  );
  model.add(
    tf.layers.maxPool2d({
      poolSize: 2,
      strides: 2,
    }),
  );
  model.add(tf.layers.flatten());
  model.add(
    tf.layers.dense({
      inputShape: 16 * 5 * 5,
      units: 120,
      activation: 'relu',
    }),
  );
  model.add(
    tf.layers.dense({
      inputShape: 120,
      units: 84,
      activation: 'relu',
    }),
  );
  model.add(
    tf.layers.dense({
      inputShape: 84,
      units: 10,
      activation: 'relu',
    }),
  );
  model.compile({
    optimizer: tf.train.sgd(0.01),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });

  console.log(model.summary());
  return model;
}
