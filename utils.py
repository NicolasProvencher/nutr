from sklearn.metrics import matthews_corrcoef, roc_auc_score, precision_recall_curve, auc, confusion_matrix
import torch.nn.functional as F
import torch
import numpy as np
from datasets import Dataset
import pandas as pd
import code





def compute_metrics(eval_pred):
    """Computes F1 score for binary classification"""
    predictions, references = eval_pred.predictions, eval_pred.label_ids
    mask = references != -100
    predictions = predictions[mask]
    references = references[mask]
    predictions = np.argmax(predictions, axis=-1).flatten()
    references = references.flatten()
    precision, recall, _ = precision_recall_curve(references, predictions)
    auc_score = auc(recall, precision)
    tn, fp, fn, tp = confusion_matrix(references, predictions).ravel()
    # print(f'prediction 2 {predictions}')    # Compute FNR and FPR
    r = {'rocauc': roc_auc_score(references, predictions),
        'pr_auc': auc_score,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,

        }
    print(r)
    return r




# def compute_metrics(eval_pred):
#     """Computes ROC AUC and PR AUC for binary classification"""
#     predictions, labels = eval_pred.predictions, eval_pred.label_ids

#     mask = labels != -100
#     predictions = predictions[mask]
#     labels = labels[mask]
#     torch_predictions = torch.from_numpy(predictions)
#     torch_labels = torch.from_numpy(labels)

#     # Compute ROC AUC
#     roc_auc = AUROC(task='binary')(torch_predictions, torch_labels)

#     # Compute PR AUC
#     pr_curve = PrecisionRecallCurve(task='binary')
#     precision, recall, _ = pr_curve(torch_predictions, torch_labels)
#     pr_auc = auc(recall, precision)
#         # Compute confusion matrix
#     tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
#     sk_predictions = np.argmax(predictions, axis=-1).flatten()
#     sk_references = labels.flatten()
#     sk_precision, sk_recall, _ = precision_recall_curve(sk_references, sk_predictions)
#     sk_auc_score = auc(sk_recall, sk_precision)
#     # print(f'prediction 2 {predictions}')    # Compute FNR and FPR


#     return {
#         'roc_auc': roc_auc.item(),
#         'pr_auc': pr_auc.item(),
#         'tn': tn,
#         'fp': fp,
#         'fn': fn,
#         'tp': tp,
#         'sk_rocauc': roc_auc_score(sk_references, sk_predictions),
#         'sk_pr_auc': sk_auc_score
#     }

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

    #add a -100 to ignore the <cls> token
    new_labels.insert(0, -100)
    labels_tensor = torch.tensor(new_labels)

    if len(labels_tensor) < max_length:
        labels_tensor = F.pad(labels_tensor, pad=(0, max_length - len(labels_tensor)), value=-100)

    example['labels'] = labels_tensor
    token=tokenizer(example[sequence_name],return_tensors="pt",padding="max_length", max_length = max_length)
    example['input_ids'] = token['input_ids'][0]
    example['attention_mask'] = token['attention_mask'][0]
    example['token']=tokenizer.decode(token['input_ids'][0].tolist())

    
    return example


def get_Data(csv_path, separator, input_sequence_col, label_col, tokenizer, chrm_split, split):
    max_length = tokenizer.model_max_length
    print('max')
    print(max_length)
    print()
    df = pd.read_csv(csv_path, sep=separator, usecols=[input_sequence_col, label_col, 'chrm', 'transcript_name']).reset_index(drop=True)
    df['chrm'] = df['chrm'].astype(str)
    print(df.shape)
    df = df[df['sequence'].apply(len) <= (max_length*6)-26]
    print(df.shape)
    #code.interact(local=locals())

    # chrm_split dict should be in the form:
    # chrm_split={<split1>:{'train':[],'val':[],'test':[]},<split2>:{'train':[],'val':[],'test':[]}, ... }


    datasets = {
        'train': Dataset.from_pandas(df.loc[df['chrm'].isin(chrm_split[split]['train'])]),
        'val': Dataset.from_pandas(df.loc[df['chrm'].isin(chrm_split[split]['val'])]),
        'test': Dataset.from_pandas(df.loc[df['chrm'].isin(chrm_split[split]['test'])])
    }

    for name, dataset in datasets.items():
        datasets[name] = dataset.map(tokenise_input_seq_and_labels, fn_kwargs={"label_name": label_col, "sequence_name": input_sequence_col, "max_length": max_length, "tokenizer": tokenizer})
        datasets[name] = datasets[name].remove_columns([ '__index_level_0__'])
        if name != 'test':
            datasets[name] = datasets[name].remove_columns(['transcript_name',input_sequence_col,'chrm','token'])

    return datasets['train'], datasets['val'], datasets['test']


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