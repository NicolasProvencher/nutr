import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, AutoModelForTokenClassification, AutoModelForSequenceClassification
import torch
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model
import pandas as pd
import wandb
from datasets import Dataset
import torch.nn.functional as F
import os

###imports
from utils import compute_metrics_f1_score, tokenise_input_seq_and_labels

def parse_arguments():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Description of your program.')
    parser.add_argument('--train_file', help='Train CSV input file', type=str)
    parser.add_argument('--test_file', help='Test CSV input file', type=str)
    parser.add_argument('--val_file', help='Validation input CSV file', type=str)
    parser.add_argument('--separator', default=',', help='Separator of the CSV input file')
    parser.add_argument('--input_sequence_col', default='data', help='Name of the column containing input sequences')
    parser.add_argument('--label_col', default='labels', help='Name of the column containing labels')
    parser.add_argument('--model_directory', help='Path to the directory containing the model files', type=str)
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_labels_promoter', type=int, default=2, help='Number of labels for the promoter')
    parser.add_argument('--offline_wandb_path', help='Offline wandb path')
    parser.add_argument('--wandb_project_name', help='Wandb project')
    parser.add_argument('--wandb_run_name', help='Wandb run name')





    parser.add_argument('--remove_unused_columns', default=False, help='Remove unused columns')
    parser.add_argument('--evaluation_strategy', default="steps", help='Evaluation strategy')
    parser.add_argument('--save_strategy', default="steps", help='Save strategy')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--per_device_train_batch_size', type=int, default=batch_size, help='Per device train batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1, help='Per device eval batch size')
    parser.add_argument('--num_train_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--logging_steps', type=int, default=10, help='Logging steps')
    parser.add_argument('--load_best_model_at_end', default=True, help='Load the best model at the end')
    parser.add_argument('--metric_for_best_model', default="f1_score", help='Metric for best model')
    parser.add_argument('--dataloader_drop_last', default=True, help='Drop last batch in dataloader')
    parser.add_argument('--report_to', default='wandb', help='Report to')
    parser.add_argument('--logging_dir', default="./logs", help='Logging directory')
    
    # Add arguments using add_argument() method
    # Example:
    # parser.add_argument('-f', '--file', help='Path to input file')

    # Parse the command-line arguments
    args = parser.parse_args()






    return args






def main():
    # Parse the command-line arguments
    args = parse_arguments()
    device = torch.device("cuda")
    num_labels_promoter = args.num_labels_promoter
    model = AutoModelForTokenClassification.from_pretrained(args.model_directory)
    model.to(device)
    peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS, inference_mode=False, r=1, lora_alpha= 32, lora_dropout=0.1, target_modules= ["query", "value"],
            )
    lora_classifier = get_peft_model(model, peft_config) # transform our classifier into a peft model
    lora_classifier.print_trainable_parameters()
    lora_classifier.to(device) # Put the model on the GPU
    tokenizer = AutoTokenizer.from_pretrained(args.model_directory)
    max_length = tokenizer.model_max_length

    val = Dataset.from_pandas(pd.read_csv("val_chrm.csv", sep=",", usecols=[sequence_name, 'labels']).iloc[:20])
    val_tok=val.map(tokenise_input_seq_and_labels, fn_kwargs={"label_name": args.label_col, "sequence_name": args.input_sequence_col, "max_length": max_length})
    val_tok = val_tok.remove_columns(arg.sequence_name)
    
    train = Dataset.from_pandas(pd.read_csv("train_chrm.csv", sep=",", usecols=[args.sequence_name, 'labels']).iloc[:20])
    train_tok = train.map(tokenise_input_seq_and_labels, fn_kwargs={"label_name": args.label_col, "sequence_name": args.input_sequence_col, "max_length": max_length})
    train_tok = train_tok.remove_columns(args.sequence_name)
    os.environ['WANDB_DIR'] = args.offline_wandb_path
    wandb.init(mode='offline', project=args.wandb_project_name, name=args.wandb_run_name)
    train_args = TrainingArguments(
        f"{args.wandb_project_name}-finetuned-lora-NucleotideTransformer",
        remove_unused_columns=args.remove_unused_columns,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps= args.gradient_accumulation_steps,
        per_device_eval_batch_size= args.batch_size,
        num_train_epochs= args.num_train_epochs,
        logging_steps= args.logging_steps,
        load_best_model_at_end=args.load_best_model_at_end,  # Keep the best model according to the evaluation
        metric_for_best_model=args.metric_for_best_model,
        label_names=args.labels_col,
        dataloader_drop_last=args.dataloader_drop_last,
        #max_steps= 1000,
        report_to=args.report_to,
        logging_dir=args.logging_dir,
        )
    trainer = Trainer(
    model.to(device),
    train_args,
    train_dataset= train,
    eval_dataset= val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_f1_score,
    )
    train_results = trainer.train()
    wandb.finish()
    



if __name__ == '__main__':
    main()
