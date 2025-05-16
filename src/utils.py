import os
import json
import logging
import random
import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score,confusion_matrix






def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_logging(filename=None, level=logging.INFO):
    logging.basicConfig(
        filename=filename,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
    )

def batch_to_device(batch, device):
    """
    Move the batch to the specified device.
    """
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device) 
    return batch 


def mean_dequeue(deque):
    """
    Calculate the mean of the last N elements in a deque.
    """
    if len(deque) == 0:
        return 0
    return sum(deque) / len(deque)


def get_optimizer_step(optimizer):
    try:
        for params in optimizer.param_groups[0]["params"]:
            params_state = optimizer.state[params]
            if "step" in params_state:
                return params_state["step"]

        return -1
    except KeyError:
        return -1 
    

   
def metrics_calc(pred_id, label_id):
    """
    Calculate the metrics for the predictions.
    """
    f1 = f1_score(label_id, pred_id, average="macro")
    acc = accuracy_score(label_id, pred_id)
    metrics = {
        "f1": f1,
        "accuracy": acc,
    }
    return metrics
def metrics_calc_label(pred_id, label_id, label2id):
    """
    Calculate the F1 score for each label.
    """
    metrics = {}
    for label, label_idx in label2id.items():
        # Get binary arrays for the current label
        binary_preds = (pred_id == label_idx).astype(int)
        binary_labels = (label_id == label_idx).astype(int)
        
        # Calculate confusion matrix for the current label
        tn, fp, fn, tp = confusion_matrix(binary_labels, binary_preds, labels=[0, 1]).ravel()
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        key = "{}_f1".format(label)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        metrics[key] = f1
    overall_f1 = f1_score(label_id, pred_id, average="macro")
    overall_acc = accuracy_score(label_id, pred_id)
    metrics["overall_f1"] = overall_f1
    metrics["overall_acc"] = overall_acc
    return metrics
        
def eval_report(pred_df, label2id, group_by=None):
    """
    Report the evaluation result, print the overall F1 and accuracy to the logger.
    Additionally, create a dictionary that stores the results, sorted by the code of the datapoint,
    along with the overall metrics.
    """
    results = {}

    # Calculate overall metrics
    overall_metrics = metrics_calc_label(pred_df["pred_id"].values, pred_df["label_id"].values, label2id)
    results["overall_f1"] = overall_metrics["overall_f1"]
    results["overall_acc"] = overall_metrics["overall_acc"]

    for label, _ in label2id.items():
        results[f"overall_f1_{label}"] = overall_metrics[f"{label}_f1"]

    # Calculate metrics for each group if group_by is provided
    if group_by:
        groups = pred_df[group_by].unique()
        for group in groups:
            group_df = pred_df[pred_df[group_by] == group]
            group_preds = group_df["pred_id"].values
            group_labels = group_df["label"].values
            group_metrics = metrics_calc_label(group_preds, group_labels, label2id)

            results[f"{group}_f1"] = group_metrics["overall_f1"]
            results[f"{group}_acc"] = group_metrics["overall_acc"]

            for label, _ in label2id.items():
                results[f"{group}_f1_{label}"] = group_metrics[f"{label}_f1"]

    return results

def save_report(metrics, path):
    """
    Save the metrics to a JSON file.
    """
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4) 
def save_prediction(pred_df,id2label, path):
    """
    conver the predictions to the original labels and save them to a CSV file.
    """

    pred_df["pred_label"] = [id2label[pred] for pred in pred_df["pred_id"].values]
    with open(path, "w") as f:
        pred_df.to_csv(f, index=False) 


def build_optimizer(model, args,total_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and "classifier" not in n
            ],
            "weight_decay": args.weight_decay,
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and "classifier" not in n],
            "weight_decay": 0.0,
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model. named_parameters() if "classifier" in n],
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
    model = ASAG_CrossEncoder(
        model_name=args.model_name,
        num_labels=len(LABEL2ID[args.label_mode]),
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