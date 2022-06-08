wget --no-check-certificate --no-proxy https://fedmol.s3.us-west-1.amazonaws.com/datasets/subgraph_level/wn18rr/wn18rr.zip && \
unzip wn18rr.zip && rm wn18rr.zip && \
mv ./wn18rr/* ./ && \
rm -rf __MACOSX && \
rm -rf wn18rr