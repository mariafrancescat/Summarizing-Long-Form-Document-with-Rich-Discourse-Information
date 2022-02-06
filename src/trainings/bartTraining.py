from transformers import Seq2SeqTrainer
from dataclasses import dataclass, field
from typing import Optional
import os
import datasets

class BartTraining(Seq2SeqTrainer):
    def __init__(self, model, dataloader, validation_loader, epochs, loss, optimizer, config, outputManager):
        dataset = dataloader.dataset
        valid_dataset = validation_loader.dataset

        super().__init__(model=model.model, 
            args=dataloader.args, train_dataset=dataset, eval_dataset=valid_dataset,
            compute_metrics=BartTraining.compute_metrics)
        
        os.environ["WANDB_DISABLED"] = "true"

    def train(self):
        super().train()
    
    @staticmethod
    def compute_metrics(pred):
        rouge = datasets.load_metric("rouge")
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # all unnecessary tokens are removed
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

        return {
            "rouge2_precision": round(rouge_output.precision, 4),
            "rouge2_recall": round(rouge_output.recall, 4),
            "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
        }