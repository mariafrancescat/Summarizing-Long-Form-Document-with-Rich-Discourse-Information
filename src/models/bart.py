from transformers import BartForConditionalGeneration

class Bart:
    def __init__(self, device='cpu'):
        self.model = BartForConditionalGeneration.from_pretrained("data/models/bart")