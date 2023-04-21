# @fedml/spider

> A Lightweight and Customizable Federated Learning Javascript SDK for Web Browser, Backed by [FedML MLOps](https://open.fedml.ai)

## Usage

- Install it with any node module manager
  ```bash
  # npm
  npm install @fedml/spider -S
  # yarn
  yarn add @fedml/spider -S
  # pnpm
  pnpm install @fedml/spider -S
  ```

- Install peer dependencies

  `npm i @tensorflow/tfjs @tensorflow/tfjs-vis -S`

- Import `fedml_train` function to start train
  ``` javascript
  import { fedml_train } from '@fedml/spider'

  // prepare running args 
  const client_id = YOUR_CLIENT_ID
  const run_args = await AnyFunctionFetchRunArgs(...)

  // start training
  fedml_train(run_args, client_id, {
    // customDataLoader?: <Optional: Your custom data loader>
  })
  ```

- API

  ```ts
  type DataLoader = (
    args: any,
    client_id: any,
    trainBatchSize: number,
    testBatchSize: number
  ) => Promise<{
    trainData: any
    trainDataLabel: any
    testData: any
    testDataLabel: any
  }>

  interface Options {
    customDataLoader?: DataLoader
  }

  function fedml_train(
    run_args: any,
    client_id: string | number,
    options?: Options
  ): Promise<void>
  ```

## Contributing
We recommend that you use `pnpm`(https://pnpm.io/installation) as the package manager.

### catalog
- `src`: Here the source codes, and `src/index.ts` is the entry file for rollup bundle.
- `dist`: The rollup bundle artifacts is output here. This will be not be committed to the GitHub codebase, however, will be released to npm.

### script
- `pnpm install`: Bootstrap dependencies of this project.
- `pnpm run dev`: Starting rollup bundler with `watch` mode.
- `pnpm run lint:fix`: Format source code style under `src` folder with eslint and prettier.
- `pnpm run release`(**Authorization required**
): Release a upgraded version package to npm, then mark and push a released tag to GitHub.
