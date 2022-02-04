from transformers import BartModel

class Bart:
    def __init__(self, device='cpu'):
        self.model = BartModel.from_pretrained("data/models/bart")