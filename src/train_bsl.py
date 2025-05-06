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
from models import BertClassifier, get_tokenizer
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
from sbank_utils import SbankDatasetInstance,LABEL_MAPS
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
    parser.add_argument('--model-type', default='bert', type=str, help='model type to use')
    parser.add_argument('--label-mode', default='3-ways', type=str, help='label mode to use')
    parser.add_argument('--test-mode', default='test_ua', type=str, help='test mode to use')
    # Add optimization arguments
    parser.add_argument('--batch-size', default=32, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--max-epoch', default=3, type=int, help='force stop training at specified epoch')
    parser.add_argument('--clip-norm', default=4.0, type=float, help='clip threshold of gradients')
    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate')
    parser.add_argument('--lr2', default=3e-5, type=float, help='learning rate for the second optimizer')
    parser.add_argument('--patience', default=3, type=int,
                        help='number of epochs without improvement on validation set before early stopping')
    parser.add_argument('--warm-up-steps',default=100,type=int)
    parser.add_argument('--weight-decay', default=0.01, type=float, help='weight decay for Adam')
    parser.add_argument('--adam-epsilon', default=1e-8, type=float, help='epsilon for Adam optimizer')
    parser.add_argument('--warmup-proportion', default=0.05, type=float, help='proportion of warmup steps')
    # Add checkpoint arguments
    parser.add_argument('--save-dir', default='results/checkpoints', help='path to save checkpoints')
    parser.add_argument('--no-save', action='store_true', help='don\'t save models or checkpoints')
    # other arguments
    parser.add_argument('--dropout', type=float,default=0.1 ,metavar='D', help='dropout probability')
    parser.add_argument('--freeze-layers',default=10,type=int, metavar='F', help='number of encoder layers in bert whose parameters to be frozen')
    parser.add_argument('--freeze-embeddings', action='store_true', help='freeze the embeddings')
    parser.add_argument('--freeze-encoder', action='store_true', help='freeze the encoder')
    parser.add_argument('--test-only', action='store_true', help='test model only')
    parser.add_argument('--fp16', action='store_true', help='use 16-bit float precision instead of 32-bit')
def get_args():
    parser = argparse.ArgumentParser()
    add_training_args(parser)
    args = parser.parse_args()
    return args

def build_optimizer(model, args,total_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and "classifier_heads" not in n
            ],
            "weight_decay": args.weight_decay,
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and "classifier_heads" not in n],
            "weight_decay": 0.0,
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model. named_parameters() if "classifier_heads" in n],
            "weight_decay": args.weight_decay,
            "lr": args.lr2,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_proportion * total_steps,
        num_training_steps=total_steps,
    )

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
    model = BertClassifier(
        model_type=args.model_type,
        num_labels=len(LABEL_MAPS[args.label_mode])
    )
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
        args.__dict__.update(training_config)
        logger.info("found existing training arguments, overwriting the new one")
    model = load_model(args)
    optimizer, scheduler = build_optimizer(model, args,total_steps) 
    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler
    }
def train_epoch(model, train_dataset,val_dataset, optimizer, scheduler, args):
    model.zero_grad()
    best_metric = 0 
    loss_history = deque(maxlen=10) 
    acc_history = deque(maxlen=10)
    num_epochs = args.max_epoch + int(args.fp16 and DEFAULT_DEVICE != "cpu")

    train_iterator = trange(num_epochs, position=0, leave=True, desc="Epoch") 
    scaler = GradScaler(enabled=args.fp16 and DEFAULT_DEVICE == "cuda")
    for epoch in train_iterator:
        train_dataloader = DataLoader(
            train_dataset, 
            num_workers=0,
            pin_memory=True,
            batch_size=args.batch_size, 
            collate_fn=SbankDatasetInstance.collate_fn,
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
            with autocast(device_type=DEFAULT_DEVICE,enabled=args.fp16): # mixed precision training
                model_output = model(**batch)
                loss = model_output.loss
                tr_loss = loss.item()
                scaler.scale(loss).backward()
            label_id = batch["label_id"].detach().cpu().numpy()
            logits = model_output.logits.detach().cpu().numpy()

            pred_id = np.argmax(logits, axis=1) 
            acc = accuracy_score(label_id, pred_id)
       
            acc_history.append(acc) 
            loss_history.append(tr_loss)
            if args.clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step() 
            epoch_iterator.set_description(
                "Epoch {}|Training: loss {:.4f} acc {:.4f} ≈".format(
                    epoch,
                    mean_dequeue(loss_history),
                    mean_dequeue(acc_history),
          
            ))
            accuracy = np.mean(list(acc_history))

            wandb.log({
                "train": {
                    "loss:": tr_loss,
                    "accuracy": accuracy
                }
            })
        # Evaluate on validation dataset
        val_predictions, val_loss = evaluate(
            model,
            val_dataset,
            batch_size=args.batch_size,
            is_test=False,
        )
        eval_metrics = eval_report(val_predictions, label2id=LABEL_MAPS[args.label_mode])
        eval_f1 = eval_metrics["overall_f1"]
        eval_acc = eval_metrics["overall_acc"]
        if eval_f1 > best_metric:
            best_metric = eval_f1
            if not args.no_save:
                export_cp(model, optimizer, scheduler, args, model_name="model.pt")
                logger.info("Best model saved at epoch %d", epoch)
        logger.info("Epoch %d: Validation loss: %.4f, F1: %.4f, Accuracy: %.4f", epoch, val_loss, eval_f1, eval_acc)
        wandb.log({
            "eval": {
                "loss": val_loss,
                "f1": eval_f1,
                "accuracy": eval_acc
            }
        })
        save_report(
            eval_metrics,
            os.path.join(args.save_dir, "val_results.json"),
        )
        
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
        collate_fn=SbankDatasetInstance.collate_fn,
        shuffle=False) 
    
    data_iterator = tqdm(dataloader, desc="Evaluating", position=0 if is_test else 2, leave=True)

 
    model.eval()
    eval_loss = 0
    last_step = 1 
    predictions = defaultdict(list)
    for step, (batch, meta) in enumerate(data_iterator):
        batch = batch_to_device(batch, DEFAULT_DEVICE)
        model_output = model(**batch)
        loss = model_output.loss
        logits = model_output.logits.detach().cpu().numpy()
        eval_loss += loss.item()

        pred_id = np.argmax(logits, axis=1)
        # collect data to put in the prediction dict
        predictions["pred_id"].extend(pred_id)
        for key, value in meta.items():
            predictions[key].extend(value)
    pred_df = pd.DataFrame(predictions)
    return pred_df, eval_loss / last_step 
    


def main(args):
    if args.freeze_encoder:
        args.freeze_layers = 8964
    set_seed()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    configure_logging(filename=os.path.join(args.save_dir, "train.log"))
    wandb.login()
    wandb.init(project="sb-baseline", config=vars(args))
    logger.info("Training arguments: %s", args)
    # Load the dataset
    tok = get_tokenizer(args.model_type)
    dataset = SbankDatasetInstance()
    split = dataset.get_training_split(val_ratio=0.1, seed=42)
    train_dataset = SbankDatasetInstance.get_encoding(tok, split["train"])
    val_dataset = SbankDatasetInstance.get_encoding(tok, split["val"])
    test_dataset = SbankDatasetInstance.get_encoding(tok, split["test"])
    steps_per_epoch = int(np.ceil(len(train_dataset) / args.batch_size)) 
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
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.max_epoch)
        logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
        train_stats = train_epoch(
            model,
            train_dataset,
            val_dataset,
            optimizer,
            scheduler,
            args
        )
        logger.info("***** Training finished *****")
    # Evaluate on test dataset
    logger.info("***** Running evaluation on test set *****")
    logger.info("  Num examples = %d", len(test_dataset))
    test_predictions, test_loss = evaluate(
        model,
        test_dataset,
        batch_size=args.batch_size,
        is_test=True,
    )
    test_metrics = eval_report(test_predictions, label2id=LABEL_MAPS[args.label_mode])
    test_f1 = test_metrics["overall_f1"]
    test_acc = test_metrics["overall_acc"]
    logger.info("Test set results: Loss: %.4f, F1: %.4f, Accuracy: %.4f", test_loss, test_f1, test_acc)
    save_report(
        test_metrics,
        os.path.join(args.save_dir, "test_results.json"),
    )
    save_prediction(
        test_predictions,
        os.path.join(args.save_dir, "test_predictions.csv"),
    )
    wandb.log({
        "test": {
            "loss": test_loss,
            "f1": test_f1,
            "accuracy": test_acc
        }
    })
    
if __name__ == "__main__":
    args = get_args()
    # Set up logging
    main(args)
    