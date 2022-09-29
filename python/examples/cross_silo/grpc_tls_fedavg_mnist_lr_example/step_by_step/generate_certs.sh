echo "Generating certificates"
OUT_PATH=config

# 1. Generate CA's private key and self-signed certificate
openssl req -x509 -newkey rsa:4096 -days 365 -nodes -keyout $OUT_PATH/ca-key.pem -out $OUT_PATH/ca-cert.pem -subj "/CN=EXAMPLE"
openssl x509 -in $OUT_PATH/ca-cert.pem -noout

# 2. Generate server private key and certificate signing request (CSR)
openssl req -newkey rsa:4096 -nodes -keyout $OUT_PATH/server-key.pem -out $OUT_PATH/server-req.pem -subj "/CN=127.0.0.1,localhost"

# 3. Use CA's private key to sign server CSR and get back the signed certificate
openssl x509 -req -in $OUT_PATH/server-req.pem -days 60 -CA $OUT_PATH/ca-cert.pem -CAkey $OUT_PATH/ca-key.pem -CAcreateserial -out $OUT_PATH/server-cert.pem -extfile $OUT_PATH/server-ext.cnf
openssl x509 -in $OUT_PATH/server-cert.pem -noout