import esbuild from 'rollup-plugin-esbuild';
import dts from 'rollup-plugin-dts';
import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';
import json from '@rollup/plugin-json';
import alias from '@rollup/plugin-alias';
import terser from '@rollup/plugin-terser';

const entries = ['src/index.ts'];

function createPlugins(minify = false) {
  const plugins = [
    alias({
      entries: [{ find: /^node:(.+)$/, replacement: '$1' }],
    }),
    resolve({
      preferBuiltins: true,
    }),
    json(),
    commonjs(),
    esbuild({
      target: 'chrome58',
      // target: 'node14',
    }),
  ];

  minify && plugins.push(terser());

  return plugins;
}

const globals = {
  '@tensorflow/tfjs': 'tf',
  'aws-sdk': 'AWS',
};

/**
 * @see https://rollupjs.org/configuration-options/#external
 */
const external = Object.keys(globals);

/**
 *
 * @param {string} input
 * @param {boolean} umd
 */
function createBundle(input, umd = false) {
  return {
    input,
    output: umd
      ? [
          // UMD format bundle for browser CDN inject with `script` tag.
          {
            file: input.replace('src/', 'dist/umd/').replace('.ts', umd ? '.min.js' : '.js'),
            name: 'FedmlSpider',
            format: 'umd',
            exports: 'named',
            globals,
          },
        ]
      : [
          // ESM format for rollup、vite or webpack applications.
          {
            file: input.replace('src/', 'dist/').replace('.ts', '.mjs'),
            format: 'esm',
            globals,
          },
          // CJS format for nodejs enviroment.
          {
            file: input.replace('src/', 'dist/').replace('.ts', '.cjs'),
            format: 'cjs',
            globals,
          },
        ],
    external,
    plugins: createPlugins(umd),
  };
}

export default [
  ...entries.map((input) => createBundle(input)),

  ...entries.map((input) => createBundle(input, true)),

  ...entries.map((input) => ({
    input,
    output: {
      file: input.replace('src/', '').replace('.ts', '.d.ts'),
      format: 'esm',
    },
    external,
    plugins: [dts({ respectExternal: true })],
  })),
];
