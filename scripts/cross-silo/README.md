

# login to the management node and setup ssh and pdsh
```

# install docker in Amazon Linux 2
sudo amazon-linux-extras install docker
sudo service docker start


cp ssh/config ~/.ssh/
cp ssh/*.pem ~/.ssh/
chmod 600 ~/.ssh/config
chown $USER ~/.ssh/config

chmod 400 ~/.ssh/*.pem

# first time login
ssh trpc-server
ssh trpc-client0
```