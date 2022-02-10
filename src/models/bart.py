from transformers import BartForConditionalGeneration

class Bart:
    def __init__(self, device='cpu'):
        self.model = BartForConditionalGeneration.from_pretrained("data/models/bart-cnn")
    
    def generate(self, ids, num_beams=4, max_length=50, early_stopping=True):
        return self.model.generate(ids,num_beams=num_beams, max_length=max_length, early_stopping=early_stopping)
    
    def load(self, model):
        pass