# Constants

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

PAD_LABEL = "O"
UNK_LABEL = "O"

N_CLIENTS = 10


# Variables
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.lang.zh import Chinese
from spacy.lang.de import German
from spacy.lang.ru import Russian
from spacy.lang.cs import Czech


def get_tokenizer(lang):
    if lang == "zh":
        nlp = Chinese()
    elif lang == "en":
        nlp = English()
    elif lang == "cs":
        nlp = Czech()
    elif lang == "de":
        nlp = German()
    elif lang == "ru":
        nlp = Russian()
    else:
        raise Exception("Unacceptable language.")
    tokenizer = Tokenizer(nlp.vocab)
    return tokenizer


zh_tokenizer = get_tokenizer("zh")
en_tokenizer = get_tokenizer("en")
cs_tokenizer = get_tokenizer("cs")
de_tokenizer = get_tokenizer("de")
ru_tokenizer = get_tokenizer("ru")
