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

def load_config(config_file):
    # Load arguments from a YAML file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_arguments():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Description of your program.')

    parser.add_argument('--config_file', default='config.yml', help='Path to the config file', type=str)
    args, _ = parser.parse_known_args()


    #arguments for input

    parser.add_argument('--input_file', help='Train CSV input file', type=str)
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
    parser.add_argument('--r', type=int, default=4, help='R')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRa alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRa dropout')
    parser.add_argument('--target_modules', nargs='+', default=["query", "value"], help='Target modules')
    args, _ = parser.parse_known_args()


    #argument for model training
    parser.add_argument('--evaluation_strategy', default="steps", help='Evaluation strategy')
    parser.add_argument('--save_strategy', default="steps", help='Save strategy')
    parser.add_argument('--save_steps', type=int, default=50, help='Save steps')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=64, help='Per device eval batch size')
    parser.add_argument('--num_train_epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--logging_steps', type=int, default=25, help='Logging steps')
    parser.add_argument('--load_best_model_at_end', default=False, help='Load the best model at the end')
    parser.add_argument('--metric_for_best_model', default="pr_auc", help='Metric for best model')
    parser.add_argument('--dataloader_drop_last', default=True, help='Drop last batch in dataloader')
    parser.add_argument('--report_to', default='wandb', help='Report to')
    parser.add_argument('--output_dir', default="./output", help='Output directory')



    parser.add_argument('--offline_wandb_path', help='Offline wandb path')
    parser.add_argument('--wandb_project_name', help='Wandb project')
    parser.add_argument('--wandb_run_name', help='Wandb run name')
    parser.add_argument('--predict', action='store_true', help='Predict mode')
    args, _ = parser.parse_known_args()



    #load config from yml file\
    config=load_config(args.config_file)
    for k, v in config.items():
        setattr(args, k, v)
    args.chrm_split = config['chrm_split']

    return args


def main():
    for split in range(1, 2):

        # Parse the command-line arguments
        args = parse_arguments()

        device = torch.device("cuda")

        
        if args.predict==True:#this check if we ask a prediction
            #this check if the fold already exist
            if os.path.exists(f"{args.output_dir}-split{split}"):

                #model loading and lora modifications loading
                model = AutoModelForTokenClassification.from_pretrained(f"{args.output_dir}-split{split}-final", num_labels=args.num_labels, trust_remote_code=True)
                peft_model = PeftModel.from_pretrained(model, model_id=f"{args.output_dir}-split{split}-final", num_labels=args.num_labels)
                peft_model.to(device)
                tokenizer = AutoTokenizer.from_pretrained(args.model_directory,trust_remote_code=True)

                #get the input datas 

                _, _, test=get_Data(args.input_file, args.separator, args.input_sequence_col, args.label_col, tokenizer, args.chrm_split, split)
                
                # Initialize the TrainingArguments
                training_args = TrainingArguments(
                    output_dir=f"{args.output_dir}-split{split}-final",
                    per_device_eval_batch_size=args.batch_size,
                )

                # Initialize the Trainer
                trainer = Trainer(
                    model=peft_model,
                    args=training_args,
                    compute_metrics=compute_metrics,
                )

                # Run the prediction task
                predictions, labels, metrics = trainer.predict(test)

                # Save the predictions, labels, and metrics
                np.save(f"{args.output_dir}-split{split}-final/predictions.npy", predictions)
                np.save(f"{args.output_dir}-split{split}-final/labels.npy", labels)
                np.save(f"{args.output_dir}-split{split}-final/metrics.npy", metrics)

                print("Predictions, labels, and metrics saved to predictions.npy, labels.npy, and metrics.npy")
            else:
                continue
        
       
        else: #this is if predict isnt given, maybe add a mode instead TODO?
            #check if fold already exist
            if os.path.exists(f"{args.output_dir}-split{split}/output.csv"):
                continue
            elif os.path.exists(f"{args.output_dir}-split{split}"):
                #TODO add a way to do prediction from the saved model
                sys.exit(f"Output directory {args.output_dir}-split{split} already exists. this means the trainer has been done completly or partially. ")
            else: #this runs when the run and fold dont exists
                try:#this is so i can end every wandb run
                    wandb.init(mode='offline', project=args.wandb_project_name, name=f"{args.wandb_run_name}-split{split}", dir=args.offline_wandb_path)
                    """
                    base_dir = f"{args.output_dir}-split{split}"
                     code for checkpoint loading that doesnt seem it going to be implmented
                      #if os.path.exists(base_dir): #batch size is there to stop checkpoint loading till merge happen
                        #dirs = os.listdir(base_dir)
                        #checkpoint_dirs = [d for d in dirs if d.startswith("checkpoint-")]
                        #latest_checkpoint_number = max(int(d.split('-')[1]) for d in checkpoint_dirs)
                        #latest_checkpoint_path = os.path.join(base_dir, f"checkpoint-{latest_checkpoint_number}")
                        #print(f"Loading model from {latest_checkpoint_path}")
        
                    #check if checkpoint exist and load model accordingly
                        #model = AutoModelForTokenClassification.from_pretrained(latest_checkpoint_path, num_labels=args.num_labels, trust_remote_code=True, output_attentions=False)
                    #else:
                    """
                    #load model and load modification
                    model1 = AutoModelForTokenClassification.from_pretrained(args.model_directory, num_labels=args.num_labels, trust_remote_code=True, output_attentions=False)
                    peft_config = LoraConfig(
                            task_type=args.task_type, 
                            inference_mode=args.inference_mode, 
                            r=args.r, lora_alpha= args.lora_alpha, 
                            lora_dropout=args.lora_dropout, 
                            target_modules=args.target_modules,
                            )
                    
                    model = get_peft_model(model1, peft_config) # transform our classifier into a peft model
                    model.print_trainable_parameters()
                    model.to(device)
                    #make data and tokenize it
                    tokenizer = AutoTokenizer.from_pretrained(args.model_directory,trust_remote_code=True)
                    train, val, test=get_Data(args.input_file, args.separator, args.input_sequence_col, args.label_col, tokenizer, args.chrm_split, split)

                    #decide step and save strategy
                    steps_per_epoch = len(train)
                    save_eval_freq = steps_per_epoch // 2

                    print(args.save_strategy)
                    print(args.evaluation_strategy)
                    print(save_eval_freq)
                    print(args.num_train_epochs)
                    print(steps_per_epoch)

                    a=0

                    if a==1:#this if is for checkpointing
                    #if os.path.exists(base_dir) and a==1:#this if is for checkpointing
                        print('not implemented yet')
                        # # model1 = AutoModelForTokenClassification.from_pretrained(latest_checkpoint_path, num_labels=args.num_labels, trust_remote_code=True, output_attentions=False)
                        # # model = PeftModel.from_pretrained(model1, latest_checkpoint_path)
                        # # model.to(device)
                        # #model1 = AutoModelForTokenClassification.from_pretrained(args.model_directory, num_labels=args.num_labels, trust_remote_code=True, output_attentions=False)

                        # model = AutoModelForTokenClassification.from_pretrained(latest_checkpoint_path, num_labels=args.num_labels, trust_remote_code=True)
                        # peft_model = PeftModel.from_pretrained(model, model_id=latest_checkpoint_path, num_labels=args.num_labels)

                        # #model1 = AutoModelForTokenClassification.from_pretrained(latest_checkpoint_path, num_labels=args.num_labels, trust_remote_code=True, output_attentions=False)
                        # #model = PeftModel.from_pretrained(model1, model_id=latest_checkpoint_path, num_labels=args.num_labels)

                        # train_args = TrainingArguments(
                        #     output_dir=f"{args.output_dir}-split{split}",
                        #     remove_unused_columns=args.remove_unused_columns,
                        #     evaluation_strategy=args.evaluation_strategy,
                        #     save_strategy=args.save_strategy,
                        #     save_steps=half_epoch_steps,
                        #     learning_rate=args.learning_rate,
                        #     per_device_train_batch_size=args.batch_size,
                        #     gradient_accumulation_steps= args.gradient_accumulation_steps,
                        #     per_device_eval_batch_size= args.batch_size,
                        #     num_train_epochs= args.num_train_epochs,
                        #     logging_steps= half_epoch_steps,
                        #     load_best_model_at_end=args.load_best_model_at_end, 
                        #     metric_for_best_model=args.metric_for_best_model,
                        #     label_names=['labels'],
                        #     dataloader_drop_last=args.dataloader_drop_last,
                        #     max_steps= steps_per_epoch,
                        # )
                        # trainer = Trainer(
                        #     model=peft_model,
                        #     args=train_args,
                        #     train_dataset=train,
                        #     eval_dataset=val,
                        #     compute_metrics=compute_metrics,                             
                        # )

                        # trainer.train(resume_from_checkpoint=latest_checkpoint_path, without_checkpoint_model=True)
                    #run normal training part
                    else:#this is run when we not checkpointing (which isnt possible for now)

                        train_args = TrainingArguments(
                            output_dir=f"{args.output_dir}-split{split}",
                            evaluation_strategy=args.evaluation_strategy,
                            save_strategy=args.save_strategy,
                            save_steps=save_eval_freq,
                            logging_steps= save_eval_freq,
                            learning_rate=args.learning_rate,
                            per_device_train_batch_size=args.batch_size,
                            gradient_accumulation_steps= args.gradient_accumulation_steps,
                            per_device_eval_batch_size= args.batch_size,
                            num_train_epochs= args.num_train_epochs,
                            load_best_model_at_end=args.load_best_model_at_end, 
                            metric_for_best_model=args.metric_for_best_model,
                            label_names=['labels'],
                            dataloader_drop_last=args.dataloader_drop_last,
                            max_steps= steps_per_epoch,
                            auto_find_batch_size=False,
                            disable_tqdm=False,

                        )
                        #print(val)
                        trainer = Trainer(
                            model=model,
                            args=train_args,
                            train_dataset=train,
                            eval_dataset=val,
                            compute_metrics=compute_metrics,
                        )
                        trainer.train()
                    
                    model.save_pretrained(f"{args.output_dir}-split{split}-final")
                    # predictions, labels, metrics = trainer.predict(test.remove_columns(['transcript_name', args.input_sequence_col,'chrm','token']))
                    train_args.dataloader_drop_last = False
                    output = trainer.predict(test.remove_columns(['transcript_name', args.input_sequence_col,'chrm','token']))
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
                    wandb.log(test_metrics)
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                finally:
                    wandb.finish()




if __name__ == '__main__':
    main()
