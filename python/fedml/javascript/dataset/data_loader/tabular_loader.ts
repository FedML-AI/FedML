import { DataLoader, DataConfig } from './data_loader';
import { DataSplit, TabularData } from '../data';
import { Dataset } from '../dataset_builder';
import { Task } from '../../task';
import { tf } from '../..';
import { List, Map, Set } from 'immutable';

// window size from which the dataset shuffling will sample
const BUFFER_SIZE = 1000;

export abstract class TabularLoader<Source> extends DataLoader<Source> {
  private readonly delimiter: string;

  constructor(task: Task, delimiter: string) {
    super(task);
    this.delimiter = delimiter;
  }

  /**
   * Creates a CSV dataset object based off the given source.
   * @param source File object, URL string or local file system path.
   * @param csvConfig Object expected by TF.js to create a CSVDataset.
   * @returns The CSVDataset object built upon the given source.
   */
  abstract loadTabularDatasetFrom(
    source: Source,
    csvConfig: Record<string, unknown>,
  ): tf.data.CSVDataset;

  /**
   * Expects delimiter-separated tabular data made of N columns. The data may be
   * potentially split among several sources. Every source should contain N-1
   * feature columns and 1 single label column.
   * @param source List of File objects, URLs or file system paths.
   * @param config
   * @returns A TF.js dataset built upon read tabular data stored in the given sources.
   */
  async load(source: Source, config?: DataConfig): Promise<Dataset> {
    /**
     * Prepare the CSV config object based off the given features and labels.
     * If labels is empty, then the returned dataset is comprised of samples only.
     * Otherwise, each entry is of the form `{ xs, ys }` with `xs` as features and `ys`
     * as labels.
     */
    if (config?.features === undefined) {
      // TODO @s314cy
      throw new Error('not implemented');
    }
    const columnConfigs = Map(
      Set(config.features).map((feature) => [feature, { required: false, isLabel: false }]),
    ).merge(Set(config.labels).map((label) => [label, { required: true, isLabel: true }]));

    const csvConfig = {
      hasHeader: true,
      columnConfigs: columnConfigs.toObject(),
      configuredColumnsOnly: true,
      delimiter: this.delimiter,
    };

    const dataset = this.loadTabularDatasetFrom(source, csvConfig)
      .map((t) => {
        if (typeof t === 'object' && 'xs' in t && 'ys' in t) {
          return t;
        }
        throw new Error('expected TensorContainerObject');
      })
      .map((t) => {
        // TODO order may not be stable between tensor
        const { xs, ys } = t as Record<string, Record<string, number>>;
        return {
          xs: Object.values(xs),
          ys: Object.values(ys),
        };
      });
    return config?.shuffle === undefined || config?.shuffle
      ? dataset.shuffle(BUFFER_SIZE)
      : dataset;
  }

  /**
   * Creates the CSV datasets based off the given sources, then fuses them into a single CSV
   * dataset.
   */
  async loadAll(sources: Source[], config: DataConfig): Promise<DataSplit> {
    const datasets = await Promise.all(
      sources.map(async (source) => await this.load(source, { ...config, shuffle: false })),
    );
    let dataset = List(datasets).reduce((acc: Dataset, dataset) => acc.concatenate(dataset));
    dataset = config?.shuffle ? dataset.shuffle(BUFFER_SIZE) : dataset;
    const data = new TabularData(
      dataset,
      // dataset.size does not work for csv datasets
      // https://github.com/tensorflow/tfjs/issues/5845
      undefined,
      this.task.trainingInformation,
    );
    // TODO: Implement validation split for tabular data (tricky due to streaming)
    return {
      train: data,
    };
  }
}
