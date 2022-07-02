# commit MNN
cd fedmlsdk/MobileNN/MNN
git add -A
git commit -m "update MNN"
git push
cd ../../../

# commit MNN submodule changes in MobileNN
cd fedmlsdk/MobileNN
git add MNN
git commit -m "updated MNN submodule"
git push
cd ../../

# commit MobileNN
cd fedmlsdk/MobileNN
git add -A
git commit -m "submit Mobile MNN changes"
git push
cd ../../

# commit MobileNN submodule changes in fedmlsdk
cd fedmlsdk
git add MobileNN
git commit -m "updated MobileNN submodule"
git push
cd ..

# commit fedmlsdk
git pull
git add fedmlsdk
git commit -m "updated fedmlsdk submodule"
git push


