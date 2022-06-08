wget --no-check-certificate --no-proxy https://fedmol.s3.us-west-1.amazonaws.com/datasets/subgraph_level/FB15k-237/FB15k-237.zip && \
unzip FB15k-237.zip && rm FB15k-237.zip && \
mv ./FB15k-237/* ./ && \
rm -rf __MACOSX && \
rm -rf FB15k-237