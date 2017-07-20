
python mention2vec.py /scratch/scratch/m2v ../data/ner/conll2003/conll2003.train --train --dynet-mem 2048 --dynet-seed 12345 --dev ../data/ner/conll2003/conll2003.dev --emb ../data/embeddings/sskip.100.vectors

python mention2vec.py /tmp/m2v ../data/ner/conll2003/conll2003.train.small --train --dynet-mem 2048 --dynet-seed 42 --dev ../data/ner/conll2003/conll2003.train.small --emb ../data/embeddings/sskip.100.vectors.small --epochs 50