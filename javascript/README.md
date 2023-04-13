# @fedml/spider

> Collaborative Learning from Scattered Data on Browser

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
- Import `fedml_train` function to start train
  ```js
  // in your js or ts code
  import { fedml_train } from '@fedml/spider'
  ```
- Basic api

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

## Develop

- `npm install`
- `npm run dev`
