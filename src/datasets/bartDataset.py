from datasets import Dataset

class BartDataset(Dataset):
    def __init__(self, data_path:str, params:dict, tokenizer,
     validation_set=False, pre_trained_tokenizer = None, fromWrapper=False, datasetFromWrapper = None):
        
        self.tokenizer = tokenizer['class'](**tokenizer['params'])

        if not fromWrapper:
            f = open(f'./data/{data_path}','r')
            data = json.load(f)
            f.close()
            # self.dataset = ...
        else:
            self.dataset = datasetFromWrapper

        inputs = self.tokenizer(self.dataset["Text"])
        outputs = self.tokenizer(self.dataset["Summary"])

        self.dataset = self.dataset.add_column("input_ids", inputs.input_ids)
        self.dataset = self.dataset.add_column("attention_mask", inputs.attention_mask)
        self.dataset = self.dataset.add_column("decoder_input_ids", outputs.input_ids)
        self.dataset = self.dataset.add_column("decoder_attention_mask", outputs.attention_mask)
        self.dataset = self.dataset.add_column("labels",[[-100 if token == self.tokenizer.tokenizer.pad_token_id else token for token in labels] for labels in outputs.input_ids.copy()])
        
        self.dataset = self.dataset.remove_columns(["Text", "Summary"])

    def __call__(self):
        return self.dataset