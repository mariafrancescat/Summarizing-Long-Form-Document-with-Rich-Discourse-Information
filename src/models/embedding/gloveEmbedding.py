import torch
from torch import nn
import gensim.downloader as api
from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm

class Glove25Embedding(nn.Module):
  def __init__(self, tokenizer):
    super(Glove25Embedding,self).__init__()
    vocab_length = tokenizer.vocab_length
    embedding_size = 25
    #glove = api.load("glove-twitter-25")
    glove = KeyedVectors.load('./data/models/glove25/glove25.bin')

    embedding_matrix = np.zeros((vocab_length, embedding_size))
    for word, index in tokenizer.tokenizer.word_index.items():
      #TODO: sometimes word is not present in glove dictionary. how to manage it?
      #Solution one: remove in advance out of vocab words in preprocess
      #Solution two: return a zero vector for these words
      try:
        embedding_vector = glove.get_vector(word)
      except:
        continue #solution one
        # embedding_vector = np.zeros(embedding_size) solution two

      if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
    embedding_matrix = torch.tensor(embedding_matrix)

    self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
    with torch.no_grad():
      self.embedding.weight = torch.nn.parameter.Parameter(embedding_matrix)
    
  def forward(self, x):
    return self.embedding(x.to(torch.int32)).float()