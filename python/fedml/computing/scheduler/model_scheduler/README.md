## 1 Device Login:
Login as fedml cloud master device: 
```fedml model device login $user_id_or_api_key --cloud -m```

Login as fedml cloud slave device:
```fedml model device login $user_id_or_api_key --cloud```

Login as on premise master device: 
```fedml model device login $user_id_or_api_key -premise -m```

Login as on premise slave device:
```fedml model device login $user_id_or_api_key -premise```


## 2. Model Card:
Create local model repository: 
```fedml model create -n $model_name```

Delete local model repository: 
```fedml model delete -n $model_name```

Add file to local model repository: 
```fedml model add -n $model_name -p $model_file_path```

Remove file from local model repository: 
```fedml model remove -n $model_name -f $model_file_name```

List model in the local model repository: 
```fedml model list -n $model_name```

List model in the remote model repository:
```fedml model list-remote -n $model_name -u $user_id -k $user_api_key```

Build local model repository as zip model package: 
```fedml model package -n $model_name```

Push local model repository to ModelOps(open.fedml.ai): 
```fedml model push -n $model_name -u $user_id -k $user_api_key```

Pull remote model(ModelOps) to local model repository: 
```fedml model pull -n $model_name -u $user_id -k $user_api_key```


## 3. Model Package:
Create local model repository: 
```fedml model create -n $model_name```

Delete local model repository: 
```fedml model delete -n $model_name -f $model_file_name```

Add file to local model repository: 
```fedml model add -n $model_name -p $model_file_path```

Remove file from local model repository: 
```fedml model remove -n $model_name -f $model_file_name```

List model in the local model repository: 
```fedml model list -n $model_name```

Build local model repository as zip model package: 
```fedml model package -n $model_name```


## 4. Model Deploy:
```
fedml model deploy -n $model_name -dt --on_premise --cloud -d $master_device_id -u $user_id -k $user_api_key -p $deployment_extra_params
```
