import os
import math
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

# if download with script in data folder 
# data_dir shoule be '../../../../data/fednlp/text_classification/20Newsgroups/20news-18828'

class Dataloader:
    def __init__(self,data_dir,batch_size=1):
        self.data_dir = data_dir
        self.batch_size  = batch_size
        self.folders = []
        self.files = []
        self.X = []
        self.sequence_length = []
        self.labels = []
        self.vocab = dict()
        self.pad_token = "<PAD>"

    def padding_data(self, max_sequence_length):
        for i, single_x in enumerate(self.X):
            if len(single_x) <= max_sequence_length:
                for _ in range(len(single_x), max_sequence_length):
                    single_x.append(self.pad_token)
            else:
                single_x = single_x[:max_sequence_length]
    
    def gather_files(self):
        self.folders = [f for f in os.listdir(self.data_dir)]
        print( os.listdir(self.data_dir))
        self.files = []
        print(self.folders)
        for folder_name in self.folders:
            folder_path = os.path.join(self.data_dir, folder_name)
            self.files.append([os.path.join(folder_path,f) for f in os.listdir(folder_path)])

    #remove header
    def remove_header(self,lines):
        for i in range(len(lines)):
            if(lines[i] == '\n'):
                start = i+1
                break
        new_lines = lines[start:]
        return new_lines
    
    def tokenize(self,document):
        nlp = English()
        # Create a blank Tokenizer with just the English vocab
        tokenizer = Tokenizer(nlp.vocab)
        tokens = tokenizer(document)
        for i in list(tokens):
            if i not in self.vocab:
                self.vocab[i] = len(self.vocab)

    #parse all the data set 
    def process_data(self,file_path):
        document = ""
        with open(file_path,"r",errors = 'ignore') as f:
            content = f.readlines()
            content = self.remove_header(content)
            for i in content:
                temp = i.lstrip("> ").replace("/\\","").replace("*","").replace("^","")
                document = document + temp
            self.tokenize(document) 
        sentence_token = document.split("  ")
        sentence_token = [i for i in sentence_token if len(i) > 0]
        seq_length = [len(i) for i in sentence_token]
        self.sequence_length.extend(seq_length)
        return sentence_token, max(seq_length)

    def data_loader(self):
        max_sequence_length = -math.inf
        document = []
        self.gather_files()

        for i in range(len(self.files)):
            for j in self.files[i]:
                self.labels.append(self.folders[i])
                document, max_seq_doc =  self.process_data(j)
                self.X.extend(document)
                max_sequence_length = max(max_sequence_length,max_seq_doc)
        
        self.vocab[self.pad_token] = len(self.vocab)
        self.padding_data(max_sequence_length)

                
        
        return self.X, self.labels, self.vocab, \
            self.sequence_length, max_sequence_length
    
