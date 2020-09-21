#!/usr/bin/env bash
echo Create tmpDir
rm -rf tmpDir
mkdir tmpDir
cd tmpDir
echo Decompress Jar
pwd
jar -xvf ../build/intermediates/bundles/release/classes.jar