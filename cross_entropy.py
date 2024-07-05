import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, AutoModelForTokenClassification, TrainerState, AutoConfig
import torch
import matplotlib.pyplot as plt
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, PeftConfig
import wandb
import yaml
import os
import sys
import traceback
import numpy as np
import pandas as pd
import code
###imports
from utils import tokenise_input_seq_and_labels, get_Data, compute_metrics

model1 = AutoModelForTokenClassification.from_pretrained('/home/roucoulab/Desktop/slurm_tmpdir/model/nucleotide-transformer-v2-50m-multi-species', num_labels=2, trust_remote_code=True, output_attentions=False)
device = torch.device("cuda")
peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS, 
        inference_mode=False, 
        r=1,
        lora_alpha= 32, 
        lora_dropout=0.1, 
        target_modules=['query','value'],
        )

model = get_peft_model(model1, peft_config) # transform our classifier into a peft model
# for param in model.parameters():
#     if param.requires_grad:
#         param.data =param.data.float()
# model.print_trainable_parameters()
#code.interact(local=locals())
model.to(device)
#make data and tokenize it

chrm_split={
    1:{
        'train': ["3", "5", "7", "11", "13", "15", "19", "21", "X"],
        'val': ["1", "9", "17"],
        'test': ["2", "4", "6", "8", "10", "12", "14", "16", "18", "20", "22", "Y"]},
    2:{
        'train': ["2", "6", "8", "10", "14", "16", "18", "22", "Y"],
        'val': ["4","12", "20"],
        'test': ["1", "3", "5", "7", "9", "11", "13", "15", "17", "19", "21", "X"]}}

split=1

tokenizer = AutoTokenizer.from_pretrained('/home/roucoulab/Desktop/slurm_tmpdir/model/nucleotide-transformer-v2-50m-multi-species',trust_remote_code=True)
train, val, test=get_Data("/home/roucoulab/Desktop/nutr_tutorial/nutr_data_test.csv", ',', 'sequence', 'labels', tokenizer, chrm_split, 1)

#decide step and save strategy
steps_per_epoch = len(train)
save_eval_freq = steps_per_epoch // 4

train_args = TrainingArguments(
                    output_dir=f"/home/roucoulab/Desktop/SLURM_TMPDIR/metric1-split{split}",
                    evaluation_strategy='steps',
                    save_strategy='steps',
                    save_steps=save_eval_freq,
                    logging_steps= save_eval_freq,
                    learning_rate=5e-4,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps= 4,
                    per_device_eval_batch_size= 1,
                    num_train_epochs= 1,
                    load_best_model_at_end=False, 
                    metric_for_best_model='pr_auc',
                    label_names=['labels'],
                    dataloader_drop_last=True,
                    max_steps= steps_per_epoch,
                    auto_find_batch_size=False,
                    disable_tqdm=False,
                    )

# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         outputs = model(**inputs)
#         print(outputs)
#         logits = outputs.logits
#         labels = inputs['labels']
#         loss_fct = torch.nn.CrossEntropyLoss()
#         loss = loss_fct(logits.view(-1, 2), labels.view(-1))
#         print(loss)
#         sys.exit()
#         return (loss, outputs) if return_outputs else loss
    
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs['labels']
        #print(outputs)
        # Mask for valid labels (not padding)
        valid_mask = labels != -100

        # Adjust logits and labels according to the valid mask
        valid_logits = logits.view(-1, self.model.config.num_labels)[valid_mask.view(-1)]
        valid_labels = labels[valid_mask]

        # Create a tensor of weights for valid labels
        weights = torch.tensor([0.1, 10], device=valid_logits.device)

        # Compute the loss manually for each label and apply weights
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)  # Compute loss without reduction
        loss = loss_fct(valid_logits, valid_labels)

        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
        model=model,
        args=train_args,
        train_dataset=train,
        eval_dataset=val,
        compute_metrics=compute_metrics,
        )
trainer.train()

model.save_pretrained(f"/home/roucoulab/Desktop/SLURM_TMPDIR/metric1-split{split}-final")
# predictions, labels, metrics = trainer.predict(test.remove_columns(['transcript_name', args.input_sequence_col,'chrm','token']))
output = trainer.predict(test.remove_columns(['transcript_name', 'sequence','chrm','token']))
mask=output.label_ids!=-100



am_pred=np.argmax(output.predictions,axis=2)
filtered_pred = [subarray[mask[i]].tolist() for i, subarray in enumerate(am_pred)]
filtered_labels = [subarray[mask[i]].tolist() for i, subarray in enumerate(output.label_ids)]
#filtered_metrics = compute_metrics(filtered_pred, filtered_labels)

# np.save('preditcions.npy', predictions)
# np.save('pred_labels.npy', labels)
# np.save('pred_in.npy', test['input_ids'])
# np.save('pred_inlabels.npy', test['labels'])

output_dict={'t_name':test['transcript_name'],
    'input_ids':test['input_ids'],
    'token':test['token'],
    'labels':test['labels'],
    'predictions':filtered_pred,
    'true_labels':filtered_labels,
    'sequence':test[args.input_sequence_col]}
output_df = pd.DataFrame(output_dict)
output_df.to_csv(f"{args.output_dir}-split{split}/output.csv", index=False)

test_metrics = {f"test/{k}": v for k, v in output.metrics.items()}

print(test_metrics)