import { createConvModel, createLogisticRegression } from './mnist_trainer';
import type { DataSet } from './mnist_trainer';

export function createModel(args: any, dataSet?: DataSet) {
  console.log(dataSet, 'dataSet');
  let model;
  if (args.dataset == 'mnist') model = createLogisticRegression(dataSet);
  else if (args.dataset == 'cifar10') model = createConvModel();

  return model;
}
