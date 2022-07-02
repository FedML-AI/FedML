# commit MNN
cd fedmlsdk/MobileNN/MNN
git add -A
git commit -m "commit changes in MNN"
git push
cd ../../../


# commit MNN submodule changes in MobileNN
cd fedmlsdk/MobileNN
git submodule update --remote
git add MNN
git commit -m "updated MNN submodule in MobileNN"
git push
cd ../../

# commit MobileNN
cd fedmlsdk/MobileNN
git add -A
git commit -m "commit changes in MobileNN"
git push
cd ../../

# commit MobileNN submodule changes in fedmlsdk
cd fedmlsdk
git submodule update --remote
git add MobileNN
git commit -m "commit MobileNN submodule changes in fedmlsdk"
git push
cd ..

# commit fedmlsdk
cd fedmlsdk
git add ./
git commit -m "commit fedmlsdk"
git push
cd ..


# commit fedmlsdk submodule changes in android
git submodule update --remote
git add fedmlsdk
git commit -m "commit fedmlsdk submodule changes in android"
git push
