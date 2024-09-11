from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, confusion_matrix, precision_score, recall_score, average_precision_score, precision_recall_curve, PrecisionRecallDisplay, roc_curve, auc 
import torch.nn.functional as F
import torch
import numpy as np
import wandb
from datasets import Dataset
from scipy.special import softmax
import pandas as pd
import code
import ast
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import io
from PIL import Image
import PIL.ImageOps
import seaborn as sns



def plot_roc(predictions_prob, references, n_classes):

    # Binarize the output
    if n_classes == 2:
        predictions_prob = predictions_prob[:, 1]
        fpr, tpr, _ = roc_curve(references, predictions_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:0.2f})')
    else:
        references = label_binarize(references, classes=range(n_classes))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(references[:, i], predictions_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i],
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    buf=io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    if n_classes == 2:
        return roc_auc, buf
    else:
        return np.nanmean([x for x in roc_auc.values()]), buf

def plot_pr(predictions_prob, references, n_classes):
    if n_classes == 2:
        predictions_prob = predictions_prob[:, 1]
        precision, recall, _ = precision_recall_curve(references, predictions_prob)
        average_precision = average_precision_score(references, predictions_prob)

        # Plot the PR curve
        plt.plot(recall, precision, label=f'PR curve (area = {average_precision:0.2f})')
    else:
        references = label_binarize(references, classes=range(n_classes))
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(references[:, i], predictions_prob[:, i])
            average_precision[i] = average_precision_score(references[:, i], predictions_prob[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            references.ravel(), predictions_prob.ravel()
        )
        average_precision["micro"] = average_precision_score(references, predictions_prob, average="micro")
        # setup plot details

        _, ax = plt.subplots(figsize=(7, 8))

        f_scores = np.linspace(0.2, 0.8, num=4)
        lines, labels = [], []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
            plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

        display = PrecisionRecallDisplay(
            recall=recall["micro"],
            precision=precision["micro"],
            average_precision=average_precision["micro"],
        )
        display.plot(ax=ax, name="Micro-average precision-recall", color="gold")

        for i in range(n_classes):
            display = PrecisionRecallDisplay(
                recall=recall[i],
                precision=precision[i],
                average_precision=average_precision[i],
            )
            display.plot(ax=ax, name=f"Precision-recall for class {i}")

        # add the legend for the iso-f1 curves
        handles, labels = display.ax_.get_legend_handles_labels()
        handles.extend([l])
        labels.extend(["iso-f1 curves"])
        # set the legend and the axes
        ax.legend(handles=handles, labels=labels, loc="best")
        ax.set_title("Extension of Precision-Recall curve to multi-class")
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png')
    buf1.seek(0)
    plt.close()
    if n_classes == 2:
        return average_precision, buf1
    else:
        return np.mean([i for i in average_precision.values() if i!=-0.00]), buf1

def plot_confusion_matrix(references, predictions_hard, class_names):
    cm = confusion_matrix(references, predictions_hard)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Convert plot to NumPy array
    fig = plt.gcf()
    fig.canvas.draw()
    cm_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    cm_image = cm_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()

    return cm_image

def compute_metrics(eval_pred):
    """Computes metrics for multiclass classification"""
    predictions, references = eval_pred.predictions, eval_pred.label_ids
    mask = references != -100
    predictions = predictions[mask]
    references = references[mask]
    predictions_prob = softmax(predictions, axis=-1)
    predictions_hard = np.argmax(predictions, axis=-1)

    #metric calculation
    #prob
    roc_auc, buf = plot_roc(predictions_prob, references, 2)
    pr_auc, buf1 = plot_pr(predictions_prob, references, 2)
    roc_image = np.array(Image.open(buf))
    pr_image = np.array(Image.open(buf1))
    cm_image = plot_confusion_matrix(references, predictions_hard, class_names=[str(i) for i in range(2)])
    #hard
    tn, fp, fn, tp = confusion_matrix(references, predictions_hard).ravel()
    print(confusion_matrix(references, predictions_hard).ravel())
    # Prepare results dictionary
    r = {
        'rocauc': roc_auc,
        'roc_curve': wandb.Image(roc_image),
        'pr_auc': pr_auc,
        'pr_curve': wandb.Image(pr_image),
        'confusion_matrix': wandb.Image(cm_image),
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'mcc': matthews_corrcoef(references, predictions_hard),
        'accuracy': accuracy_score(references, predictions_hard),
        'f1': f1_score(references, predictions_hard),
        'precision': precision_score(references, predictions_hard),
        'recall': recall_score(references, predictions_hard),
        'tpr': tp/(tp+fn) ,
        'fpr': fp/(fp+tn)
    }

    return r

def compute_metrics_cov(eval_pred):
    """Computes metrics for multiclass classification"""
    predictions, references = eval_pred.predictions, eval_pred.label_ids
    mask = references != -100
    predictions = predictions[mask]
    references = references[mask]
    predictions_prob = softmax(predictions, axis=-1)
    predictions_hard = np.argmax(predictions, axis=-1)

    #metric calculation
    #prob
    roc_auc, buf = plot_roc(predictions_prob, references, 14)
    pr_auc, buf1 = plot_pr(predictions_prob, references, 14)
    roc_image = np.array(Image.open(buf))
    pr_image = np.array(Image.open(buf1))
    cm_image = plot_confusion_matrix(references, predictions_hard, class_names=[str(i) for i in range(14)])
    #hard
    #tn, fp, fn, tp = confusion_matrix(references, predictions_hard).ravel()
    print(confusion_matrix(references, predictions_hard).ravel())
    # Prepare results dictionary
    r = {
        'rocauc': roc_auc,
        'roc_curve': wandb.Image(roc_image),
        'pr_auc': pr_auc,
        'pr_curve': wandb.Image(pr_image),
        'confusion_matrix': wandb.Image(cm_image),
        # 'tn': tn,
        # 'fp': fp,
        # 'fn': fn,
        # 'tp': tp,
        'mcc': matthews_corrcoef(references, predictions_hard),
        'accuracy': accuracy_score(references, predictions_hard),
        'f1': f1_score(references, predictions_hard, average='weighted'),
        'precision': precision_score(references, predictions_hard, average='weighted'),
        'recall': recall_score(references, predictions_hard, average='weighted'),
        # 'tpr': tp/(tp+fn) ,
        # 'fpr': fp/(fp+tn)
    }

    return r




def tokenise_input_seq_and_labels(example, max_length, tokenizer, label_name, sequence_name):
    
    labels = example[label_name]
    new_labels = []
    for i in range(0, len(labels), 6):
        segment = labels[i:i+6]
        # if '1' in segment:
        #     new_labels.append(1)
        # else:            
        #   new_labels.append(0)
        new_labels.append(int(max(segment)))

    if ((len(labels) % 6)) >1:
        segment = labels[-(len(labels) % 6)+1:]
        for j in segment:
            new_labels.append(j)

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

    #train=df.loc[df['chrm'].isin(chrm_split[split]['train'])][:500]


    datasets = {
        'train': Dataset.from_pandas(df.loc[df['chrm'].isin(chrm_split[split]['train'])][:5000].assign(labels=lambda x: x['labels'].apply(ast.literal_eval))),
        'val': Dataset.from_pandas(df.loc[df['chrm'].isin(chrm_split[split]['val'])][:5000].assign(labels=lambda x: x['labels'].apply(ast.literal_eval))),
        'test': Dataset.from_pandas(df.loc[df['chrm'].isin(chrm_split[split]['test'])][:5000].assign(labels=lambda x: x['labels'].apply(ast.literal_eval)))
    }

    for name, dataset in datasets.items():
        datasets[name] = dataset.map(tokenise_input_seq_and_labels, fn_kwargs={"label_name": label_col, "sequence_name": input_sequence_col, "max_length": max_length, "tokenizer": tokenizer})
        datasets[name] = datasets[name].remove_columns([ '__index_level_0__'])
        if name != 'test':
            datasets[name] = datasets[name].remove_columns(['transcript_name',input_sequence_col,'chrm','token'])
    # a=[i for i in datasets['train'] if len(i['labels'])>1000]
    # b=[i for i in datasets['val'] if len(i['labels'])>1000]
    # c=[i for i in datasets['test'] if len(i['labels'])>1000]
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