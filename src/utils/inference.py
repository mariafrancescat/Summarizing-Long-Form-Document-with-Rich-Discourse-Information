import numpy as np
from src.models.heroes import *

class Inference:

    @staticmethod
    def inferenceFrom(modelClass):
        mapping = {
            ContentRanking : Inference.contentRanking
        }
        return mapping[modelClass]

    @staticmethod
    def contentRanking(model, docs):
        assert isinstance(model,ContentRanking), 'Model not instance of ContentRanking'
        sections_importance, sentences_importance = model(docs)
        
        number_of_sections = 2
        number_of_sentences = 5

        for doc in sections_importance:
            importance_list = [section.item() for section in doc]
            best_setcions = np.argpartition(importance_list,-number_of_sections)[-number_of_sections:]        
        #TODO
        #return document digest
