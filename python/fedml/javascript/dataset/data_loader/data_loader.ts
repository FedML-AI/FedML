import { Dataset } from '../dataset_builder';
import { DataSplit } from '../data';
import { Task } from '../../task';

export interface DataConfig {
  features?: string[];
  labels?: string[];
  shuffle?: boolean;
  validationSplit?: number;
}

export abstract class DataLoader<Source> {
  protected task: Task;

  constructor(task: Task) {
    this.task = task;
  }

  abstract load(source: Source, config: DataConfig): Promise<Dataset>;

  abstract loadAll(sources: Source[], config: DataConfig): Promise<DataSplit>;
}
