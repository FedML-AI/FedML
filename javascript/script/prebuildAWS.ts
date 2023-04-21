import { execSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
// @ts-ignore
import build from 'aws-sdk/dist-tools/browser-builder.js';

process.env.MINIFY = '1';

try {
  execSync('cd node_modules/aws-sdk && pnpm i');
} catch (error) {
  console.log(error.message || "ERR_ON: execSync('cd node_modules/aws-sdk && pnpm i')");
  process.exit(-1);
}

build({}, (err, code) => {
  if (err) return process.exit(-1);
  mkdirSync('src/libs', { recursive: true });
  writeFileSync('src/libs/aws-sdk.js', code, { encoding: 'utf-8' });
  console.log('✅：prebiuld aws-sdk into src/libs/aws-sdk.js');
});
