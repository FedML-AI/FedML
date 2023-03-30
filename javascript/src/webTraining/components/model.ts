import { createLogisticRegression, createConvModel } from './mnist_trainer';

export function createModel(args, dataSet) {
  console.log(dataSet, 'dataSet');
  let model;
  if (args.dataset == 'mnist') {
    model = createLogisticRegression(dataSet);
  } else if (args.dataset == 'cifar10') {
    model = createConvModel();
  }
  return model;
}
