from typing import Literal
from datasets import load_dataset,enable_caching,disable_caching
from transformers import AutoTokenizer
import torch

enable_caching()
FIELDS = [
    "id",
    "question",
    "reference_answer",
    "student_answer",
    "label",
]
LABEL2ID = {
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

ID2LABEL = {
    "5-ways": [k for k, v in LABEL2ID["5-ways"].items()],
    "3-ways": [k for k, v in LABEL2ID["3-ways"].items()],
    "2-ways": [k for k, v in LABEL2ID["2-ways"].items()],
}

class SbankDataset:
    def __init__(self,label_mode="3-ways"):
        self.label_mode = label_mode  
        self.data_dict = load_dataset("nkazi/SciEntsBank")  
        for split in self.data_dict:
            self.data_dict[split] = self._format_label(self.data_dict[split], label_format=label_mode)
        self.get_training_split(val_ratio=0.1, seed=42)
    def _format_label(self,dataset,label_format: Literal["5-ways", "3-ways", "2-ways"] = "3-ways") -> None:
        """
        Formats the label based on the specified label format. The label is already in integer. 
        The function first maps the integer labels to string labels, and then maps them back to integers based on the specified format.

        Args:
            label_format (Literal["5-ways", "3-ways", "2-ways"]): The label format, must be one of the allowed values.
        """
        dataset = dataset.rename_column("label", "label_id")
        def map_label(label: int) -> int:
            if label_format == "3-ways":
                if label > 1:
                    # Map labels 3 ("irrelevant") and 4 ("non_domain") to 2 ("incorrect")
                    return 2
                else:
                    return label
            elif label_format == "2-ways":
                if label > 0:  # Map all labels except 0 ("correct") to 1 ("incorrect")
                    return 1
                return 0
            return label  # For "5-ways", no mapping is needed.
        dataset = dataset.map(lambda x: {"label_id": map_label(x["label_id"])})
        id2label = ID2LABEL[label_format]
        dataset = dataset.map(lambda x: {"label": id2label[x["label_id"]]})
        return dataset

    def get_training_split(self, val_ratio=0.1, seed=42):
        """
        Splits the train dataset into train and validation sets.

        Args:
            val_ratio (float): The ratio of the validation set size to the train set size.
            seed (int): Random seed for reproducibility.

        Returns:
            tuple: A tuple containing the train and validation datasets.
        """
 
        train_val_split = self.data_dict["train"].train_test_split(
            test_size=val_ratio,
            seed=seed,
            shuffle=True
        )
        self.data_dict["train"] = train_val_split["train"]
        self.data_dict["val"] = train_val_split["test"]


class SbankDatasetInstance(SbankDataset):
    def __init__(self, format_label="3-ways"):
        super().__init__(format_label)

    def encode_all_splits(self, tokenizer):
        """
        Encodes all splits in the data_dict using the provided tokenizer.

        Args:
            tokenizer: The tokenizer to use for encoding.
        """
        for split in self.data_dict:
            self.data_dict[split] = self.get_encoding(tokenizer, self.data_dict[split])
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
    ds = SbankDatasetInstance(format_label="3-ways") 

    split = ds.get_training_split(val_ratio=0.1, seed=42)
    tr_ds = ds.get_encoding(tok, split["train"])    
    tr_loader = torch.utils.data.DataLoader(
        tr_ds,
        batch_size=16,
        collate_fn=SbankDatasetInstance.collate_fn,
        shuffle=True,
    )
    for batch, meta in tr_loader:
        print(batch["label_id"])
        print(meta["label"])
        break
