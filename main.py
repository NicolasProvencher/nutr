import argparse
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, AutoModelForTokenClassification
import torch
import matplotlib.pyplot as plt
from peft import LoraConfig, TaskType, get_peft_model
import wandb

import os
import sys

###imports
from utils import compute_metrics_f1_score, tokenise_input_seq_and_labels, get_Data

def load_config():
    # Load arguments from a YAML file
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_arguments():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Description of your program.')

    #arguments for input
    parser.add_argument('--train_file', help='Train CSV input file', type=str)
    parser.add_argument('--test_file', help='Test CSV input file', type=str)
    parser.add_argument('--val_file', help='Validation input CSV file',  type=str)
    parser.add_argument('--separator', default=',', help='Separator of the CSV input file')
    parser.add_argument('--input_sequence_col', default='data', help='Name of the column containing input sequences')
    parser.add_argument('--label_col', default='labels', help='Name of the column containing labels')

    #arguments for model loading
    parser.add_argument('--model_directory', help='Path to the directory containing the model files', type=str)
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of labels for the promoter')

    #arguments for LoRa
    parser.add_argument('--task_type', default=TaskType.TOKEN_CLS, help='Task type')
    parser.add_argument('--inference_mode', type=bool, default=False, help='Inference mode')
    parser.add_argument('--r', type=int, default=1, help='R')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRa alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRa dropout')
    parser.add_argument('--target_modules', nargs='+', default=["query", "value"], help='Target modules')
    args, _ = parser.parse_known_args()

    #argument for model training
    parser.add_argument('--remove_unused_columns', default=False, help='Remove unused columns')
    parser.add_argument('--evaluation_strategy', default="steps", help='Evaluation strategy')
    parser.add_argument('--save_strategy', default="steps", help='Save strategy')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--per_device_train_batch_size', type=int, default=args.batch_size, help='Per device train batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=64, help='Per device eval batch size')
    parser.add_argument('--num_train_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--logging_steps', type=int, default=10, help='Logging steps')
    parser.add_argument('--load_best_model_at_end', default=True, help='Load the best model at the end')
    parser.add_argument('--metric_for_best_model', default="f1_score", help='Metric for best model')
    parser.add_argument('--dataloader_drop_last', default=True, help='Drop last batch in dataloader')
    parser.add_argument('--report_to', default='wandb', help='Report to')
    parser.add_argument('--logging_dir', default="./logs", help='Logging directory')


    #arguments for wandb
    parser.add_argument('--offline_wandb_path', help='Offline wandb path')
    parser.add_argument('--wandb_project_name', help='Wandb project')
    parser.add_argument('--wandb_run_name', help='Wandb run name')
    
    config=load_config()
    config_args = [f'--{k}={v}' for k, v in config.items() if k in vars(parser.parse_args())]
    args = parser.parse_args(args=config_args)
    return args

import yaml
import argparse

def parse_arguments():
    # Load arguments from a YAML file
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Description of your program.')

    # Use the values from the YAML file if they exist, otherwise use the default values
    parser.add_argument('--model_directory', help='Path to the directory containing the model files', default=config.get('model_directory', "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"), type=str)
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 1), help='Batch size')
    # ... repeat for other arguments ...

    args = parser.parse_args()
    return args




def main():
    # Parse the command-line arguments
    args = parse_arguments()
    print(args.num_labels)
    device = torch.device("cuda")
    model = AutoModelForTokenClassification.from_pretrained(args.model_directory, num_labels=args.num_labels)
    model.to(device)

    #TODO check if target module can vary and why
    peft_config = LoraConfig(
            task_type=args.task_type, inference_mode=args.inference_mode, r=args.r, lora_alpha= args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=args.target_modules,
            )
    lora_classifier = get_peft_model(model, peft_config) # transform our classifier into a peft model
    lora_classifier.print_trainable_parameters()
    lora_classifier.to(device) # Put the model on the GPU

    tokenizer = AutoTokenizer.from_pretrained(args.model_directory)


    val=get_Data(args.val_file, args.separator, args.input_sequence_col, args.label_col, tokenizer)
    train= get_Data(args.train_file, args.separator, args.input_sequence_col, args.label_col, tokenizer)

    #os.environ['WANDB_DIR'] = args.offline_wandb_path
    #wandb.init(mode='offline', project=args.wandb_project_name, name=args.wandb_run_name)

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
        load_best_model_at_end=args.load_best_model_at_end, 
        metric_for_best_model=args.metric_for_best_model,
        label_names=args.label_col,
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
    sys.exit()
    train_results = trainer.train()
    wandb.finish()
    



if __name__ == '__main__':
    main()
