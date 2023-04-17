import { init_client } from './client/client_initializer'
export class FedMLCrossSiloClient {
  constructor(args, device, dataset, model, model_trainer) {
    // console.log('dataset: ', toRaw(dataset))
    // const { trainData, trainDataLabel, testData, testDataLabel } = toRaw(dataset)
    // console.log('dataset: ', trainData)
    const { trainData, trainDataLabel, testData, testDataLabel } = dataset
    init_client(args, device, args.comm, args.rank, args.worker_num, model, trainData, trainDataLabel, testData, testDataLabel)
  }

  run() {
  }
}
