## FedCV

```bash
mkdir fedcv
cd fedcv
git init
git remote add -f origin https://github.com/FedML-AI/FedML.git
git config core.sparsecheckout true
echo "app/fedcv" >> .git/info/sparse-checkout
git pull origin master
```
