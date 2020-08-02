cd
echo Create tmpDir
rm -rf tmpDir
mkdir tmpDir
cd tmpDir
echo Decompress Jar
jar -xvf ../build/intermediates/bundles/release/classes.jar