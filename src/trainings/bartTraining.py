from transformers import Seq2SeqTrainer
from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    label_smoothing: Optional[float] = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to SortishSamler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    adafactor: bool = field(default=False, metadata={"help": "whether to use adafactor"})
    encoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Encoder layer dropout probability. Goes into model.config."}
    )
    decoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Decoder layer dropout probability. Goes into model.config."}
    )
    dropout: Optional[float] = field(default=None, metadata={"help": "Dropout probability. Goes into model.config."})
    attention_dropout: Optional[float] = field(
        default=None, metadata={"help": "Attention dropout probability. Goes into model.config."}
    )
    lr_scheduler: Optional[str] = field(
        default="linear", metadata={"help": f"Which lr scheduler to use."}
    )


class BartTraining(Seq2SeqTrainer):
    def __init__(self, model, dataloader, validation_loader, epochs, loss, optimizer, config, outputManager):
        dataset = dataloader.dataset
        valid_dataset = validation_loader.dataset

        training_args = Seq2SeqTrainingArguments(
            output_dir="./",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            predict_with_generate=True,
            evaluate_during_training=True,
            do_train=True,
            do_eval=True,
            logging_steps=2, 
            save_steps=16, 
            eval_steps=500, 
            warmup_steps=500, 
            overwrite_output_dir=True,
            save_total_limit=1,
            fp16=True, 
        )
        super.__init__(BartTraining, model=model, 
            args=training_args, train_dataset=dataset, eval_dataset=valid_dataset,
            compute_metrics=BartTraining.compute_metrics)
    
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