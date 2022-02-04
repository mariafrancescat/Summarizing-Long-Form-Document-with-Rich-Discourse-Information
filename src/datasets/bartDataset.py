from datasets import Dataset

class BartDataset:
    def __init__(self, data_path:str, params:dict, tokenizer,
     validation_set=False, pre_trained_tokenizer = None, fromWrapper=False, datasetFromWrapper = None):
        self.tokenizer = tokenizer

        if not fromWrapper:
            f = open(f'./data/{data_path}','r')
            data = json.load(f)
            f.close()
            # dataset = ...
        else:
            dataset = datasetFromWrapper
    
        inputs = tokenizer(dataset["Text"], padding="max_length", truncation=True, max_length=params['encoder_max_length'])
        outputs = tokenizer(dataset["Summary"], padding="max_length", truncation=True, max_length=params['decoder_max_length'])

        dataset["input_ids"] = inputs.input_ids
        dataset["attention_mask"] = inputs.attention_mask
        dataset["decoder_input_ids"] = outputs.input_ids
        dataset["decoder_attention_mask"] = outputs.attention_mask
        dataset["labels"] = outputs.input_ids.copy()

        # because RoBERTa automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
        # We have to make sure that the PAD token is ignored
        dataset["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]
        dataset.remove_columns(["Text", "Summary"])