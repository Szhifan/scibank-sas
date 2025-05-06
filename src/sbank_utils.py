from typing import Literal
from datasets import load_dataset,disable_caching
from transformers import AutoTokenizer
import torch


FIELDS = [
    "id",
    "question",
    "reference_answer",
    "student_answer",
    "label",
]
LABEL_MAPS = {
    "5-ways": {
        "correct": 0,
        "contradictory": 1,
        "partially_correct_incomplete": 2,
        "irrelevant": 3,
        "non_domain": 4,
    },
    "3-ways": {
        "correct": 0,
        "contradictory": 1,
        "incorrect": 2,
    },
    "2-ways": {
        "correct": 0,
        "incorrect": 1,
    },
}

class SbankDataset:
    def __init__(self,test_mode="test_ua",label_mode="3-ways"):
        self.test = test_mode
        self.label_mode = label_mode  
        self.ds_original = load_dataset("nkazi/SciEntsBank")  
        self.train_ds = self.ds_original["train"]
        self.test_ds = self.ds_original[test_mode]
        self._format_label(label_format=label_mode)
        
    def _format_label(self, label_format: Literal["5-ways", "3-ways", "2-ways"] = "3-ways") -> None:
        """
        Formats the label based on the specified label format.

        Args:
            label_format (Literal["5-ways", "3-ways", "2-ways"]): The label format, must be one of the allowed values.
        """
        def map_label(label: str) -> str:
            if label_format == "3-ways":
                if label in ["partially_correct_incomplete", "irrelevant", "non_domain"]:
                    return "incorrect"
                return label
            elif label_format == "2-ways":
                if label in ["contradictory", "partially_correct_incomplete", "irrelevant", "non_domain"]:
                    return "incorrect"
                return "correct"
            return label  # For "5-ways", no mapping is needed.

        self.train_ds = self.train_ds.map(
            lambda example: {"label_id": map_label(example["label"])}
        )
        self.test_ds = self.test_ds.map(
            lambda example: {"label_id": map_label(example["label"])}
        ) 
    def get_training_split(self, val_ratio=0.1, seed=42):
        """
        Splits the train dataset into train and validation sets.

        Args:
            val_ratio (float): The ratio of the validation set size to the train set size.
            seed (int): Random seed for reproducibility.

        Returns:
            tuple: A tuple containing the train and validation datasets.
        """
 
        train_val_split = self.train_ds.train_test_split(test_size=val_ratio, seed=seed)
        self.train_ds = train_val_split["train"]
        self.val_ds = train_val_split["test"]
        return {
            "train": self.train_ds,
            "val": self.val_ds,
            "test": self.test_ds,
        }


class SbankDatasetInstance(SbankDataset):
    def __init__(self, test="test_ua", format_label="3-ways"):
        super().__init__(test, format_label)
        
    @staticmethod
    def get_encoding(tokenizer,dataset):
        def tokenize_function(example, tokenizer):
            encoding = tokenizer(
                example["reference_answer"],
                example["student_answer"],
                truncation=True,
            ) 
            example["input_ids"] = encoding["input_ids"]
            example["attention_mask"] = encoding["attention_mask"]
            if "token_type_ids" in encoding:
                example["token_type_ids"] = encoding["token_type_ids"]
            return example 
        return dataset.map(
            lambda example: tokenize_function(example, tokenizer)
        )
    @staticmethod
    def collate_fn(input_batch):
        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["input_ids"]) for x in input_batch], batch_first=True)
        attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["attention_mask"]) for x in input_batch], batch_first=True)
        if "token_type_ids" in input_batch[0]:

            token_type_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["token_type_ids"]) for x in input_batch], batch_first=True)
        else: 
            token_type_ids = None
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "label_id": torch.tensor([x["label_id"] for x in input_batch]),

        } 

        meta = {
            "id": [x["id"] for x in input_batch],
            "question": [x["question"] for x in input_batch],
            "reference_answer": [x["reference_answer"] for x in input_batch],
            "student_answer": [x["student_answer"] for x in input_batch],
            "label": [x["label"] for x in input_batch],
        }
        return batch, meta 
            
if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    ds = SbankDatasetInstance(test="test_ua", format_label="3-ways")
    split = ds.get_training_split(val_ratio=0.1, seed=42)
    tr_ds = ds.get_encoding(tok, split["train"])    
    tr_loader = torch.utils.data.DataLoader(
        tr_ds,
        batch_size=2,
        collate_fn=SbankDatasetInstance.collate_fn,
        shuffle=True,
    )
    for batch, meta in tr_loader:
        print(batch)
        print(meta)
        break
    