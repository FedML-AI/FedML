rm -rf data
# svn checkout https://github.com/jeniyat/WNUT_2020_NER/trunk/data # A wrong place; and plz don't use unnecessary tool
wget "https://raw.githubusercontent.com/juand-r/entity-recognition-datasets/master/data/WNUT17/CONLL-format/data/train/wnut17train.conll"

wget "https://raw.githubusercontent.com/juand-r/entity-recognition-datasets/master/data/WNUT17/CONLL-format/data/dev/emerging.dev.conll"

wget "https://raw.githubusercontent.com/juand-r/entity-recognition-datasets/master/data/WNUT17/CONLL-format/data/test/emerging.test.annotated"