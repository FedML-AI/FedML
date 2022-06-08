wget --no-check-certificate --no-proxy https://fedmol.s3.us-west-1.amazonaws.com/datasets/subgraph_level/YAGO3-10/YAGO3-10.zip && \
unzip YAGO3-10.zip && rm YAGO3-10.zip && \
mv ./YAGO3-10/* ./ && \
rm -rf __MACOSX && \
rm -rf YAGO3-10