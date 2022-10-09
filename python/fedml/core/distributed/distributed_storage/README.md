## Web3 storage
Use Web3 storage as federated learning distributed storage for reading and writing models.
You should register account at https://web3.storage and set config parameters:
token, upload_uri, download_uri.
If you want to use secret key to encrypt models, you should set secret key by calling Context().add("ipfs_secret_key", "your secret key")

## Theta edge store
Use Theta EdgeStore as federated learning distributed storage for reading and writing models.
You should setup theta edgestore based on https://docs.thetatoken.org/docs/theta-edge-store-setup and set config parameters:
store_home_dir, upload_uri, download_uri.
If you want to use secret key to encrypt models, you should set secret key by calling Context().add("ipfs_secret_key", "your secret key")

