from datasets import load_dataset,disable_caching
import json 
import tqdm 
import torch
from sentence_transformers import (
    SentenceTransformerTrainingArguments,
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerModelCardData
)
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction,BinaryClassificationEvaluator
disable_caching()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_contrastive = "data/contrastive_train_3ways.json"


def get_contrastive(dataset):
    """
    Generates contrastive pairs for each question ID in the dataset.
    Returns:
        A dictionary where keys are question IDs and values are lists of corresponding student answers.
    """ 
    dataset_contrastive = []
    dataset_df = dataset.to_pandas()
    dataset_gp_qid = dataset_df.groupby("qid")
    for qid, group in dataset_gp_qid:
        group = group.to_dict(orient="records")
        for i in tqdm.tqdm(range(len(group))):
            for j in range(i + 1, len(group)):
                if group[i]["qid"] == group[j]["qid"]:
                    pair = {
                        "qid": group[i]["qid"],
                        "student_answer_1": group[i]["student_answer"],
                        "student_answer_2": group[j]["student_answer"],
                        "label_id_1": group[i]["label_id"],
                        "label_id_2": group[j]["label_id"],
                        "label_contrastive": 1 if group[i]["label_id"] == group[j]["label_id"] else 0,
                        "id_1": group[i]["id"],
                        "id_2": group[j]["id"],
                        "question": group[i]["question"],
                        "reference_answer": group[i]["reference_answer"],
                    }
                    dataset_contrastive.append(pair)
    return dataset_contrastive

contrastive_dataset = load_dataset("json", data_files=path_contrastive)["train"]
contrastive_dataset = contrastive_dataset.rename_columns({
    "student_answer_1": "sentence1",
    "student_answer_2": "sentence2",
    "label_contrastive": "label"
})
cols_to_keep = ["sentence1", "sentence2", "label"]
contrastive_dataset = contrastive_dataset.select_columns(cols_to_keep)          # Re-orders columns too

tr_ds,val_ds = contrastive_dataset.train_test_split(test_size=0.1, seed=42).values()
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="results/contrastive_cosine",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    # Device specification
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    logging_steps=500,
    run_name="results/contrastive_cosine",  # Will be used in W&B if `wandb` is installed
)

# training 
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L12-v2"
)
dev_evaluator = BinaryClassificationEvaluator(
    sentences1=val_ds["sentence1"],
    sentences2=val_ds["sentence2"],
    labels=val_ds["label"],
    name = "contrastive_eval",
    show_progress_bar=True,

)

loss_fn =CosineSimilarityLoss(model=model)
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=tr_ds,
    eval_dataset=val_ds,
    loss=loss_fn,
    evaluator=dev_evaluator,
)


trainer.train()
trainer.save_model("results/contrastive_cosine/final_model")