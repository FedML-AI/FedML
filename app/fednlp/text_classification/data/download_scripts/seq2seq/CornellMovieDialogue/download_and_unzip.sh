rm cornell_movie_dialogs_corpus.zip
rm -rf rm cornell_movie_dialogs_corpus
wget http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip
unzip cornell_movie_dialogs_corpus.zip
mv cornell\ movie-dialogs\ corpus/ cornell_movie_dialogs_corpus/
rm cornell_movie_dialogs_corpus.zip
mv "cornell movie-dialogs corpus" "cornell-movie-dialogs-corpus"