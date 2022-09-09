import { assert, expect } from 'chai';
import fs from 'fs';
// import _ from 'lodash'

import { dataset, tasks } from '../..';
import { List, Map, Range } from 'immutable';
import * as tf from '@tensorflow/tfjs-node';

export class NodeImageLoader extends dataset.ImageLoader<string> {
  async readImageFrom(source: string): Promise<tf.Tensor3D> {
    return tf.node.decodeImage(fs.readFileSync(source)) as tf.Tensor3D;
  }
}

const readFilesFromDir = (dir: string): string[] =>
  fs.readdirSync(dir).map((file: string) => dir + file);
const DIRS = {
  CIFAR10: './example_training_data/CIFAR10/',
};
const FILES = Map(DIRS).map(readFilesFromDir).toObject();

describe('image loader', () => {
  it('loads single sample without label', async () => {
    const file = './example_training_data/9-mnist-example.png';
    const singletonDataset = await new NodeImageLoader(tasks.mnist.task).load(file);
    const imageContent = tf.node.decodeImage(fs.readFileSync(file));
    await Promise.all(
      (
        await singletonDataset.toArrayForTest()
      ).map(async (entry) => {
        expect(await imageContent.bytes()).eql(await (entry as tf.Tensor).bytes());
      }),
    );
  });

  it('loads multiple samples without labels', async () => {
    const imagesContent = FILES.CIFAR10.map((file) => tf.node.decodeImage(fs.readFileSync(file)));
    const datasetContent = await (
      await new NodeImageLoader(tasks.cifar10.task).loadAll(FILES.CIFAR10, { shuffle: false })
    ).train.dataset.toArray();
    expect(datasetContent.length).equal(imagesContent.length);
    expect((datasetContent[0] as tf.Tensor3D).shape).eql(imagesContent[0].shape);
  });

  it('loads single sample with label', async () => {
    const path = DIRS.CIFAR10 + '0.png';
    const imageContent = tf.node.decodeImage(fs.readFileSync(path));
    const datasetContent = await (
      await new NodeImageLoader(tasks.cifar10.task).load(path, { labels: ['example'] })
    ).toArray();
    expect((datasetContent[0] as any).xs.shape).eql(imageContent.shape);
    expect((datasetContent[0] as any).ys).eql('example');
  });

  it('loads multiple samples with labels', async () => {
    const labels = Range(0, 24).map((label) => label % 10);
    const stringLabels = labels.map((label) => label.toString());
    const oneHotLabels = List(tf.oneHot(labels.toArray(), 10).arraySync() as number[]);

    const imagesContent = List(
      FILES.CIFAR10.map((file) => tf.node.decodeImage(fs.readFileSync(file))),
    );
    const datasetContent = List(
      await (
        await new NodeImageLoader(tasks.cifar10.task).loadAll(FILES.CIFAR10, {
          labels: stringLabels.toArray(),
          shuffle: false,
        })
      ).train.dataset.toArray(),
    );

    expect(datasetContent.size).equal(imagesContent.size);
    datasetContent
      .zip(imagesContent)
      .zip(oneHotLabels)
      .forEach(([[actual, sample], label]) => {
        if (typeof actual !== 'object' || !('xs' in actual && 'ys' in actual)) {
          throw new Error('unexpected type');
        }
        const { xs, ys } = actual as { xs: tf.Tensor; ys: number[] };
        expect(xs.shape).eql(sample?.shape);
        expect(ys).eql(label);
      });
  });

  it('loads samples in order', async () => {
    const loader = new NodeImageLoader(tasks.cifar10.task);
    const dataset = await (
      await loader.loadAll(FILES.CIFAR10, { shuffle: false })
    ).train.dataset.toArray();

    List(dataset)
      .zip(List(FILES.CIFAR10))
      .forEach(async ([s, f]) => {
        const sample = (await (await loader.load(f)).toArray())[0];
        if (!tf.equal(s as tf.Tensor, sample as tf.Tensor).all()) {
          assert(false);
        }
      });
    assert(true);
  });

  it('shuffles list', async () => {
    const loader = new NodeImageLoader(tasks.cifar10.task);
    const list = Range(0, 100_000).toArray();
    const shuffled = [...list];

    loader.shuffle(shuffled);
    expect(list).to.not.eql(shuffled);

    shuffled.sort((a, b) => a - b);
    expect(list).to.eql(shuffled);
  });

  it('shuffles samples', async () => {
    const loader = new NodeImageLoader(tasks.cifar10.task);
    const dataset = await (
      await loader.loadAll(FILES.CIFAR10, { shuffle: false })
    ).train.dataset.toArray();
    const shuffled = await (
      await loader.loadAll(FILES.CIFAR10, { shuffle: true })
    ).train.dataset.toArray();

    const misses = List(dataset)
      .zip(List(shuffled))
      .map(
        ([d, s]) =>
          tf
            .notEqual(d as tf.Tensor, s as tf.Tensor)
            .any()
            .dataSync()[0],
      )
      .reduce((acc: number, e) => acc + e);
    assert(misses > 0);
  });
  it('validation split', async () => {
    const validationSplit = 0.2;
    const imagesContent = FILES.CIFAR10.map((file) => tf.node.decodeImage(fs.readFileSync(file)));
    const datasetContent = await new NodeImageLoader(tasks.cifar10.task).loadAll(FILES.CIFAR10, {
      shuffle: false,
      validationSplit: validationSplit,
    });

    const trainSize = Math.floor(imagesContent.length * (1 - validationSplit));
    expect((await datasetContent.train.dataset.toArray()).length).equal(trainSize);
    if (datasetContent.validation === undefined) {
      assert(false);
    }
    expect((await datasetContent.validation.dataset.toArray()).length).equal(
      imagesContent.length - trainSize,
    );
  });
});
