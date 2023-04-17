import path from 'path'
import { readFileSync } from 'fs'
import { defineConfig } from 'rollup'
// import esbuild from 'rollup-plugin-esbuild'
import resolve from '@rollup/plugin-node-resolve'
import commonjs from '@rollup/plugin-commonjs'
import terser from '@rollup/plugin-terser'
import json from '@rollup/plugin-json'
import { babel } from '@rollup/plugin-babel'
import autoExternal from 'rollup-plugin-auto-external'
import bundleSize from 'rollup-plugin-bundle-size'
import nodePolyfills from 'rollup-plugin-polyfill-node'

const lib = JSON.parse(readFileSync('./package.json'))
const OUTPUT_FILENAME = 'FedmlSpider'
const MOD_NAME = 'FedmlSpider'
const NAMED_INPUT = './lib/index.js'
const DEFAULT_INPUT = './lib/FedmlSpider.js'

const buildConfig = ({
  es5 = true,
  browser = true,
  minifiedVersion = true,
  ...config
}) => {
  const { file } = config.output
  const ext = path.extname(file)
  const basename = path.basename(file, ext)
  const extArr = ext.split('.')
  extArr.shift()

  const build = ({ minified = false }) => ({
    input: NAMED_INPUT,
    ...config,
    output: {
      ...config.output,
      file: `${path.dirname(file)}/${basename}.${(minified ? ['min', ...extArr] : extArr).join('.')}`,
    },
    plugins: [
      json(),
      resolve({ browser }),
      commonjs(),
      nodePolyfills(),
      // esbuild({
      //   target: 'chrome58',
      //   // target: 'node14',
      // }),
      minified && terser(),
      minified && bundleSize(),
      ...(es5
        ? [babel({
            babelHelpers: 'bundled',
            presets: ['@babel/preset-env'],
          })]
        : []),
      ...(config.plugins || []),
    ],
  })

  const configs = [
    build({ minified: false }),
  ]

  if (minifiedVersion)
    configs.push(build({ minified: true }))

  return configs
}

export default defineConfig(() => {
  const year = new Date().getFullYear()
  const banner = `// ${MOD_NAME} v${lib.version} Copyright (c) ${year} ${lib.author} and contributors`

  return [
    // browser ESM bundle for CDN
    ...buildConfig({
      input: NAMED_INPUT,
      output: {
        file: `dist/esm/${OUTPUT_FILENAME}.mjs`,
        format: 'esm',
        generatedCode: {
          constBindings: true,
        },
        exports: 'named',
        banner,
      },
    }),

    // Browser UMD bundle for CDN
    ...buildConfig({
      input: DEFAULT_INPUT,
      es5: true,
      output: {
        file: `dist/${OUTPUT_FILENAME}.js`,
        name: MOD_NAME,
        format: 'umd',
        exports: 'default',
        banner,
      },
    }),

    // Browser CJS bundle
    ...buildConfig({
      input: DEFAULT_INPUT,
      es5: false,
      minifiedVersion: false,
      output: {
        file: `dist/browser/${MOD_NAME}.cjs`,
        name: MOD_NAME,
        format: 'cjs',
        exports: 'default',
        banner,
      },
    }),

    // Node.js commonjs bundle
    {
      input: DEFAULT_INPUT,
      output: {
        file: `dist/node/${MOD_NAME}.cjs`,
        format: 'cjs',
        generatedCode: {
          constBindings: true,
        },
        exports: 'default',
        banner,
      },
      plugins: [
        json(),
        autoExternal(),
        resolve(),
        commonjs(),
        // esbuild({
        //   target: 'node14',
        // }),
      ],
    },
  ]
})
