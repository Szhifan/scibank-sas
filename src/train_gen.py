from sklearn.metrics import accuracy_score
import torch
import os 
import json
import argparse
import wandb 
import torch
from torch.amp import GradScaler, autocast
import logging
import numpy as np
from collections import deque, defaultdict
from models import ASAG_T5_COND_GEN, get_tokenizer
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm, trange
from utils import (
    set_seed,
    configure_logging,
    batch_to_device,
    mean_dequeue,
    get_optimizer_step,
    eval_report,
    save_report,
    save_prediction
    )
from data_prep import SB_Dataset_conditional_generation,LABEL2ID
from torch.utils.data import DataLoader
import pandas as pd 


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
logger = logging.getLogger(__name__)
print("Using device:", DEFAULT_DEVICE)

def add_training_args(parser):
    """
    add training related args 
    """ 
    # add experiment arguments 
    parser.add_argument('--model-name', default='bert-base-uncased', type=str, help='model type to use')
    parser.add_argument('--label-mode', default='3-ways', type=str, help='label mode to use')
    parser.add_argument('--seed', default=42, type=int, help='random seed for initialization')
    # Add optimization arguments
    parser.add_argument('--batch-size', default=32, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--max-epoch', default=3, type=int, help='force stop training at specified epoch')
    parser.add_argument('--clip-norm', default=1, type=float, help='clip threshold of gradients')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--patience', default=3, type=int,
                        help='number of epochs without improvement on validation set before early stopping')
    parser.add_argument('--grad-accumulation-steps', default=1, type=int, help='number of updates steps to accumulate before performing a backward/update pass')
    parser.add_argument('--weight-decay', default=0.01, type=float, help='weight decay for Adam')
    parser.add_argument('--adam-epsilon', default=1e-8, type=float, help='epsilon for Adam optimizer')
    parser.add_argument('--warmup-proportion', default=0.05, type=float, help='proportion of warmup steps')
    # Add checkpoint arguments
    parser.add_argument('--save-dir', default='results/checkpoints', help='path to save checkpoints')
    parser.add_argument('--no-save', action='store_true', help='don\'t save models or checkpoints')
    parser.add_argument('--checkpoint', default=None, type=str, help='path to a checkpoint to load from')
    # other arguments
    parser.add_argument('--dropout', type=float,default=0.1 ,metavar='D', help='dropout probability')
    parser.add_argument('--freeze-layers',default=10,type=int, metavar='F', help='number of encoder layers in bert whose parameters to be frozen')
    parser.add_argument('--freeze-embeddings', action='store_true', help='freeze the embeddings')
    parser.add_argument('--freeze-encoder', action='store_true', help='freeze the encoder')
    parser.add_argument('--test-only', action='store_true', help='test model only')
    parser.add_argument('--fp16', action='store_true', help='use 16-bit float precision instead of 32-bit')
    parser.add_argument('--log-wandb',action='store_true', help='log experiment to wandb')
def get_args():
    parser = argparse.ArgumentParser()
    add_training_args(parser)
    args = parser.parse_args()
    return args

def build_optimizer(model, args,total_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_proportion * total_steps,
        num_training_steps=total_steps,
    )
    # if checkpoint path is provided, load optimizer and scheduler states
    if args.checkpoint is not None:
        checkpoint_path = os.path.join(args.checkpoint, "checkpoint")
        if os.path.exists(checkpoint_path):
            optimizer_path = os.path.join(checkpoint_path, "optimizer.pt")
            scheduler_path = os.path.join(checkpoint_path, "scheduler.pt")
            if os.path.isfile(optimizer_path) and os.path.isfile(scheduler_path):
                map_location = DEFAULT_DEVICE
                optimizer.load_state_dict(torch.load(optimizer_path, map_location=map_location))
                scheduler.load_state_dict(torch.load(scheduler_path, map_location=map_location))
                logger.info("Loaded optimizer and scheduler from checkpoint.")

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.save_dir, "checkpoint/optimizer.pt")) and os.path.isfile(
        os.path.join(args.save_dir, "checkpoint/scheduler.pt")
    ):
        map_location = DEFAULT_DEVICE
        optimizer_path = os.path.join(args.save_dir, "checkpoint/optimizer.pt")
        scheduler_path = os.path.join(args.save_dir, "checkpoint/scheduler.pt")
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=map_location))
        scheduler.load_state_dict(torch.load(scheduler_path, map_location=map_location))
        logger.info("Loaded the saved scheduler and optimizer.")
    return optimizer, scheduler 


    
def export_cp(model, optimizer, scheduler, args, model_name="model.pt"):
 
    # save model checkpoint 
    output_dir = os.path.join(args.save_dir, "checkpoint")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    # Save a trained model
    torch.save(model_to_save.state_dict(), os.path.join(output_dir, model_name))
    # Save training arguments
    training_config = args.__dict__.copy()
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=4) 
    logger.info("Saving model checkpoint to %s", output_dir)
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    logger.info("Saving optimizer and scheduler states to %s", output_dir)

def load_model(args):
    model = ASAG_T5_COND_GEN(
        model_name=args.model_name,
        freeze_layers=args.freeze_layers,
        freeze_embeddings=args.freeze_embeddings,
        
    )
    # if checkpoint is provided, load the model state
    if args.checkpoint is not None:
        checkpoint_path = os.path.join(args.checkpoint, "checkpoint", "model.pt")
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            logger.info("Loaded model from checkpoint: %s", checkpoint_path)
    checkpoint_path = os.path.join(args.save_dir, "checkpoint", "model.pt")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        logger.info("Loaded model from checkpoint: %s", checkpoint_path)
    else:
        logger.info("No checkpoint found. Initializing new model.")
    model = model.to(DEFAULT_DEVICE)
    return model
def import_cp(args, total_steps):
    # check if cp exists 
    checkpoint_path = os.path.join(args.save_dir, "checkpoint", "model.pt")
    if os.path.exists(checkpoint_path):
        logger.info("found checkpoint, loading model and optimizer")
    
    training_config_path = os.path.join(args.save_dir, "checkpoint", "training_config.json")
    if os.path.exists(training_config_path):
        with open(training_config_path, "r") as f:
            training_config = json.load(f)
        if training_config["model_name"] != args.model_name:
            logger.warning("Model type mismatch. Expected %s, but found %s", args.model_name, training_config["model_name"])
        if training_config["label_mode"] != args.label_mode:
            logger.warning("Label mode mismatch. Expected %s, but found %s", args.label_mode, training_config["label_mode"])
        
    model = load_model(args)
    optimizer, scheduler = build_optimizer(model, args,total_steps) 
    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler
    }
def train_epoch(
        model,
        train_dataset,
        val_dataset,
        optimizer,
        scheduler,
        args): 
    model.zero_grad()
    best_metric = np.inf 
    loss_history = deque(maxlen=10) 
    num_epochs = args.max_epoch + int(args.fp16 and DEFAULT_DEVICE != "cpu")
 
    train_iterator = trange(num_epochs, position=0, leave=True, desc="Epoch") 
    scaler = GradScaler(enabled=args.fp16 and DEFAULT_DEVICE == "cuda")
    bad_epochs = 0
    for epoch in train_iterator:
        train_dataloader = DataLoader(
            train_dataset, 
            num_workers=0,
            pin_memory=True,
            batch_size=args.batch_size, 
            collate_fn=SB_Dataset_conditional_generation.collate_fn,
            shuffle=True) 
        epoch_iterator = tqdm(
            train_dataloader, 
            desc="Iteration", 
            position=1, 
            leave=True, 

        )

 
        for step, (batch, _) in enumerate(epoch_iterator):
            model.train()
            batch = batch_to_device(batch, DEFAULT_DEVICE)
            with autocast(device_type=DEFAULT_DEVICE, enabled=args.fp16):  # mixed precision training
                model_output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["decoder_input_ids"],
                    decoder_attention_mask=batch["decoder_attention_mask"],
                )
                loss = model_output.loss / args.grad_accumulation_steps  # normalize loss for gradient accumulation
                tr_loss = loss.item() * args.grad_accumulation_steps  # scale back for logging
                scaler.scale(loss).backward()
            loss_history.append(tr_loss)
            if (step + 1) % args.grad_accumulation_steps == 0:  # perform optimizer step after accumulation
                if args.clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            epoch_iterator.set_description(
                "Epoch {}|Training: loss {:.4f} ≈".format(
                    epoch,
                    mean_dequeue(loss_history),
          
            ))

            wandb.log({
                "train": {
                    "loss:": tr_loss,
                }
            })
        # Evaluate on validation dataset
        eval_loss = evaluate(
            model,
            val_dataset,
            batch_size=args.batch_size,
            is_test=False,
        )
        if eval_loss < best_metric:
            best_metric = eval_loss
            export_cp(model, optimizer, scheduler, args, model_name="model.pt")
            logger.info("Best model saved at epoch %d", epoch)
        elif eval_loss > best_metric:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            logger.info("Early stopping at epoch %d", epoch)
            break
       

        logger.info("Epoch %d: Validation loss: %.4f, F1: %.4f, Accuracy: %.4f", epoch, eval_loss)
        wandb.log({
            "eval": {
                "loss": eval_loss,
            }
        })
        
@torch.no_grad()
def evaluate(
    model,
    dataset,
    batch_size,
    is_test=False,
):
    dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    collate_fn=SB_Dataset_conditional_generation.collate_fn,
    shuffle=False)

    data_iterator = tqdm(dataloader, desc="Evaluating", position=0 if is_test else 2, leave=True)

    model.eval()
    eval_loss = []
    for step, (batch, _) in enumerate(data_iterator):
        batch = batch_to_device(batch, DEFAULT_DEVICE)
        model_output = model(**batch)
        loss = model_output.loss
        eval_loss.append(loss.item())
        eval_loss = np.mean(eval_loss)
    return eval_loss

@torch.no_grad()
def inference(model, tokenizer, test_dataset):
    """
    Generate model output based on the test dataset.

    Args:
        model: The trained model.
        tokenizer: The tokenizer corresponding to the model.
        test_dataset: The test dataset for generation.
        max_length (int): Maximum length of the generated output.

    Returns:
        list of str: Generated outputs for each input in the test dataset.
    """
    model.eval()
    dataloader = DataLoader(
        test_dataset,
        batch_size=16,  # Adjust batch size as needed
        collate_fn=SB_Dataset_conditional_generation.collate_fn,
        shuffle=False
    )
    generated_outputs = []
    for batch, _ in tqdm(dataloader, desc="Generating Outputs"):
        batch = batch_to_device(batch, DEFAULT_DEVICE)
        outputs = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=5,       # Short labels — usually 1-3 tokens
            num_beams=4,            # Beam search for better accuracy
            early_stopping=True,    # Stop when the best sequence ends
            return_dict_in_generate=False,
            output_scores=False     # Unless you need log-probs
        )
        generated_outputs.extend(
            [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        )
    return generated_outputs
def main(args):
   
    if args.freeze_encoder:
        args.freeze_layers = 114514
    set_seed(args.seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    configure_logging(filename=os.path.join(args.save_dir, "train.log"))
    wandb.login()
    if args.log_wandb:
        wandb.init(
            project="sb-generation",
            config=vars(args),
            name=f"{args.model_name}_{args.label_mode}",
            dir=args.save_dir,
        )
    else:
        wandb.init(mode="disabled")
    logger.info("Training arguments: %s", args)
    # Load the dataset
    sbank = SB_Dataset_conditional_generation(label_mode=args.label_mode)
    sbank.encode_all_splits(get_tokenizer(args.model_name))
    sb_dict = sbank.data_dict
    steps_per_epoch = int(np.ceil(len(sb_dict["train"]) / args.batch_size)) 
    total_steps = args.max_epoch * steps_per_epoch


 
    # Load the checkpoint 
    cp = import_cp(args, total_steps)
    model = cp["model"]
    optimizer = cp["optimizer"]
    scheduler = cp["scheduler"]

    if not args.test_only:
        model.train()
        wandb.watch(model)
        # Build optimizer and scheduler

        # Training loop
        logger.info("***** Running training *****")
        logger.info("Num examples = %d", len(sb_dict["train"]))
        logger.info("  Num Epochs = %d", args.max_epoch)
        logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
        train_stats = train_epoch(
            model,
            sb_dict["train"],
            sb_dict["val"],
            optimizer,
            scheduler,
            args)  
        logger.info("***** Training finished *****")
    # Evaluate on test dataset
    

    for test_split in ["test_ua", "test_uq", "test_ud"]:
        test_dataset = sb_dict[test_split]
        logger.info(f"***** Running evaluation on {test_split} set *****")
        logger.info("  Num examples = %d", len(test_dataset))
        test_generated = inference(
            model,
            sbank.tokenizer,
            test_dataset
        )
        
        
        generated_path = os.path.join(args.save_dir, f"{test_split}_generated.txt")
        with open(generated_path, "w") as f:
            for item in test_generated:
                f.write("%s\n" % item)
        logger.info(f"Generated outputs saved to {generated_path}")
  
        # save_prediction(
        #     test_predictions,
        #     ID2LABEL[args.label_mode],
        #     os.path.join(args.save_dir, f"{test_split}_predictions.csv")
        # )
    if args.no_save:
        logger.info("No-save flag is set. Deleting checkpoint.")
        checkpoint_dir = os.path.join(args.save_dir, "checkpoint")
        if os.path.exists(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                file_path = os.path.join(checkpoint_dir, file)
                try:
                    if file_path.endswith(".pt") and os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    logger.error("Error deleting file %s: %s", file_path, e)
     
if __name__ == "__main__":
    args = get_args()
    # Set up logging
    main(args)
    