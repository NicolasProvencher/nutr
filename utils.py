from sklearn.metrics import matthews_corrcoef, roc_auc_score, precision_recall_curve, auc
import torch.nn.functional as F
import torch
import numpy as np
from datasets import Dataset
import pandas as pd





def compute_metrics(eval_pred):
    """Computes F1 score for binary classification"""
    predictions, references = eval_pred.predictions, eval_pred.label_ids

    mask = references != -100
    predictions = predictions[mask]
    references = references[mask]

    np.savetxt('prediction.txt', np.argmax(predictions, axis= -1))
    np.savetxt('reference.txt', references)
    predictions = np.argmax(predictions, axis=-1).flatten()
    references = references.flatten()
    precision, recall, _ = precision_recall_curve(references, predictions)
    auc_score = auc(recall, precision)
    # print(f'prediction 2 {predictions}')
    # print(len(predictions))
    # print(f'reference 2 {references}')
    # print(len(references))
    r = {'rocauc': roc_auc_score(references, predictions),
        'pr_auc': auc_score
        }
    return r



def tokenise_input_seq_and_labels(example, max_length, tokenizer, label_name, sequence_name):
    
    labels = example[label_name]
    new_labels = []
    for i in range(0, len(labels), 6):
        segment = labels[i:i+6]
        if '1' in segment:
            new_labels.append(1)
        else:            new_labels.append(0)
    # print(len(labels)/6)
    # print(len(labels) % 6)
    # print(len(new_labels))
    if ((len(labels) % 6)) >1:
        segment = labels[-(len(labels) % 6)+1:]
        # print(f'segment {len(segment)}')
        for i in segment:
            if i==1:
                new_labels.append(1)
            else:
                new_labels.append(0)
    # print(len(new_labels))
    #print(new_labels)
    labels_tensor = torch.tensor(new_labels)
    #print(labels_tensor)
    if len(labels_tensor) < max_length:
        labels_tensor = F.pad(labels_tensor, pad=(0, max_length - len(labels_tensor)), value=-100)

    example['labels'] = labels_tensor
    token=tokenizer(example[sequence_name],return_tensors="pt",padding="max_length", max_length = max_length)
    #print(example['labels'])
    example['input_ids'] = token['input_ids'][0]
    example['attention_mask'] = token['attention_mask'][0]
    # print(token['input_ids'].shape)
    # print(token['input_ids'][-5:])
    # print(type(example))
    
    return example


def get_Data(csv_path, separator, input_sequence_col, label_col, tokenizer, chrm_split, split):
    max_length = tokenizer.model_max_length
    df = pd.read_csv(csv_path, sep=separator, usecols=[input_sequence_col, label_col, 'chrm'])

    # chrm_split dict should be in the form:
    # chrm_split={
    #     1:{'train':[],
    #         'val':[],
    #         'test':[]},
    #     2:{'train':[],
    #         'val':[],
    #         'test':[]},}


    train = Dataset.from_pandas(df.loc[df['chrm'].isin(chrm_split[split]['train'])])
    val = Dataset.from_pandas(df.loc[df['chrm'].isin(chrm_split[split]['val'])])
    test = Dataset.from_pandas(df.loc[df['chrm'].isin(chrm_split[split]['test'])])
    for i in [train, val, test]:
        i = i.map(tokenise_input_seq_and_labels, fn_kwargs={"label_name": label_col, "sequence_name": input_sequence_col, "max_length": max_length, "tokenizer": tokenizer})
        i = i.remove_columns(input_sequence_col)
        i=i.remove_columns('chrm')
    return train, val, test


# from transformers import Trainer, TrainingArguments
# from transformers.modeling_utils import unwrap_model
# from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

# class CustomTrainer(Trainer):
#     def __init__(self, model, args: TrainingArguments, label_weights=None, **kwargs):
#         super().__init__(model, args, **kwargs)
#         self.label_weights = label_weights

#     def compute_loss(self, model, inputs, return_outputs=False):
#         if self.label_smoother is not None and "labels" in inputs:
#             labels = inputs.pop("labels")
#         else:
#             labels = None
#         outputs = model(**inputs)

#         if self.args.past_index >= 0:
#             self._past = outputs[self.args.past_index]

#         if labels is not None:
#             # Apply weights to labels here
#             if self.label_weights is not None:
#                 weights = torch.tensor(self.label_weights)[labels]
#                 labels = labels * weights

#             if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
#                 loss = self.label_smoother(outputs, labels, shift_labels=True)
#             else:
#                 loss = self.label_smoother(outputs, labels)
#         else:
#             if isinstance(outputs, dict) and "loss" not in outputs:
#                 raise ValueError(
#                     "The model did not return a loss from the inputs, only the following keys: "
#                     f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
#                 )
#             loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

#         return (loss, outputs) if return_outputs else loss
    



    '''

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    '''