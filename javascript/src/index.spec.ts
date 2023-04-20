// import { describe, expect, it } from 'vitest'
import { fedml_train } from './index'

const run_args = {
  // TODO: inject test data(avoid submit secret tokens)
}

await fedml_train(run_args, 0)

// describe('fedml_train', () => {
//   it('fedml_train', async () => {
//     console.log(result)
//     expect(result).toBe(undefined)
//   })
// })
