from keras import preprocessing
from transformers import RobertaTokenizerFast

class Arxiv_preprocess:
    def __init__(self, padding:int):
        self.tokenizer = preprocessing.text.Tokenizer()
        self.padding = padding
        self.sentences = []
    
    def add_sentence(self, sentence):
        self.sentences.append(sentence)

    def fit_tokenizer(self):
        assert len(self.sentences)>0, 'sentences list is empty!'
        self.tokenizer.fit_on_texts(self.sentences)
        self.vocab_length = len(self.tokenizer.word_index) + 1

    def get_padded_sentence(self, sentence:str):
        return preprocessing.sequence.pad_sequences(self.tokenizer.texts_to_sequences([sentence]), self.padding, padding='post')[0]
    
class Bart_tokenizer(RobertaTokenizerFast):
    def __init__(self, padding="max_length", truncation=True, max_length=100):
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    
    def __call__(self, doc):
        return self.tokenizer(doc,padding=self.padding, truncation=self.truncation, max_length=self.max_length)