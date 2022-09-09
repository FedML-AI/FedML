import * as tf from '@tensorflow/tfjs';

import { DataConfig, DataLoader } from './data_loader/data_loader';
import { DataSplit } from './data';
import { Task } from '@/task';

export type Dataset = tf.data.Dataset<tf.TensorContainer>;

export class DatasetBuilder<Source> {
  private readonly task: Task;
  private readonly dataLoader: DataLoader<Source>;
  private sources: Source[];
  private readonly labelledSources: Map<string, Source[]>;
  private built: boolean;

  constructor(dataLoader: DataLoader<Source>, task: Task) {
    this.dataLoader = dataLoader;
    this.task = task;
    this.sources = [];
    this.labelledSources = new Map();
    this.built = false;
  }

  addFiles(sources: Source[], label?: string): void {
    if (this.built) {
      this.resetBuiltState();
    }
    if (label === undefined) {
      this.sources = this.sources.concat(sources);
    } else {
      const currentSources = this.labelledSources.get(label);
      if (currentSources === undefined) {
        this.labelledSources.set(label, sources);
      } else {
        this.labelledSources.set(label, currentSources.concat(sources));
      }
    }
  }

  clearFiles(label?: string): void {
    if (this.built) {
      this.resetBuiltState();
    }
    if (label === undefined) {
      this.sources = [];
    } else {
      this.labelledSources.delete(label);
    }
  }

  // If files are added or removed, then this should be called since the latest
  // version of the dataset_builder has not yet been built.
  private resetBuiltState(): void {
    this.built = false;
  }

  private getLabels(): string[] {
    // We need to duplicate the labels as we need one for each soure.
    // Say for label A we have sources [img1, img2, img3], then we
    // need labels [A, A, A].
    let labels: string[][] = [];
    Array.from(this.labelledSources.values()).forEach((sources, index) => {
      const sourcesLabels = Array.from({ length: sources.length }, (_) => index.toString());
      labels = labels.concat(sourcesLabels);
    });
    return labels.flat();
  }

  async build(config?: DataConfig): Promise<DataSplit> {
    // Require that at leat one source collection is non-empty, but not both
    if (this.sources.length > 0 === this.labelledSources.size > 0) {
      throw new Error('invalid sources');
    }

    let dataTuple: DataSplit;
    if (this.sources.length > 0) {
      // Labels are contained in the given sources
      const defaultConfig = {
        features: this.task.trainingInformation?.inputColumns,
        labels: this.task.trainingInformation?.outputColumns,
        ...config,
      };
      dataTuple = await this.dataLoader.loadAll(this.sources, defaultConfig);
    } else {
      // Labels are inferred from the file selection boxes
      const defaultConfig = {
        labels: this.getLabels(),
        ...config,
      };
      const sources = Array.from(this.labelledSources.values()).flat();
      dataTuple = await this.dataLoader.loadAll(sources, defaultConfig);
    }
    // TODO @s314cy: Support .csv labels for image datasets (supervised training or testing)
    this.built = true;
    return dataTuple;
  }

  isBuilt(): boolean {
    return this.built;
  }

  size(): number {
    return Math.max(this.sources.length, this.labelledSources.size);
  }
}
