from sklearn.metrics import matthews_corrcoef, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_chrm_split
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
    df = pd.read_csv(csv_path, sep=separator, usecols=[input_sequence_col, label_col])

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
    return train, val, test


