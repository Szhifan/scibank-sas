from typing import Literal
from datasets import load_dataset,enable_caching,Dataset
from transformers import AutoTokenizer
import torch
import json
import tqdm 
    
def get_qid(dataset: Dataset):
    def help(id):
        qid = ".".join(id.split(".")[:2])
        return qid 

    dataset = dataset.map(
            lambda x: {"qid": help(x["id"])}) 
    return dataset

def get_contrastive(dataset):
    """
    Generates contrastive pairs for each question ID in the dataset.
    Returns:
        A dictionary where keys are question IDs and values are lists of corresponding student answers.
    """ 
    dataset = get_qid(dataset)
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

class SB_Dataset:
    def __init__(self,label_mode="3-ways"):
        self.label_mode = label_mode  
        self.data_dict = load_dataset("nkazi/SciEntsBank")  
        for split in self.data_dict:
            self.data_dict[split] = self._format_label(self.data_dict[split], label_format=label_mode)

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


class SB_Dataset_BSL(SB_Dataset):
    def __init__(self, label_mode="3-ways",input_field="ref+ans"):
        super().__init__(label_mode)
        self.input_field = input_field
        self.get_training_split()
    def encode_all_splits(self, tokenizer):
        """
        Encodes all splits in the data_dict using the provided tokenizer.

        Args:
            tokenizer: The tokenizer to use for encoding.
        """
        for split in self.data_dict:
            self.data_dict[split] = self.get_encoding(tokenizer, self.data_dict[split],self.input_field)
    @staticmethod
    def get_encoding(tokenizer,dataset,input_field="ref+ans"):
        def tokenize_function(example, tokenizer,input_field):
            if input_field == "ref+ans":
                encoding = tokenizer(
                    example["reference_answer"],
                    example["student_answer"],
                    truncation=True,
                ) 
            elif input_field == "ans":
                encoding = tokenizer(
                    example["student_answer"],
                    truncation=True,
                )
            elif input_field == "q+ans":
                encoding = tokenizer(
                    example["question"],
                    example["student_answer"],
                    truncation=True,
                )
            example["input_ids"] = encoding["input_ids"]
            example["attention_mask"] = encoding["attention_mask"]
            if "token_type_ids" in encoding:
                example["token_type_ids"] = encoding["token_type_ids"]
            return example 
        return dataset.map(
            lambda example: tokenize_function(example, tokenizer,input_field),
        )


class SB_Dataset_SentenceEmbeddings(SB_Dataset):
    def __init__(self, label_mode="3-ways"):
        super().__init__(label_mode)
        self.get_training_split()

    
    def encode_all_splits(self, tokenizer):
        """
        Encodes all splits in the data_dict using the provided tokenizer.
        Encodes reference answer and student answer separately.

        Args:
            tokenizer: The tokenizer to use for encoding.
        """
        for split in self.data_dict:
            self.data_dict[split] = self.get_encoding(tokenizer, self.data_dict[split])
    
    @staticmethod
    def get_encoding(tokenizer, dataset):
        def tokenize_function(example, tokenizer):
            # Encode reference answer and student answer separately
            ref_encoding = tokenizer(
                example["reference_answer"],
                truncation=True,
            )
            student_encoding = tokenizer(
                example["student_answer"],
                truncation=True,
            )
            
            example["input_ids_1"] = ref_encoding["input_ids"]
            example["attention_mask_1"] = ref_encoding["attention_mask"]
            example["input_ids_2"] = student_encoding["input_ids"]
            example["attention_mask_2"] = student_encoding["attention_mask"]
            
            if "token_type_ids" in ref_encoding:
                example["token_type_ids_1"] = ref_encoding["token_type_ids"]
                example["token_type_ids_2"] = student_encoding["token_type_ids"]
            
            return example
        
        return dataset.map(
            lambda example: tokenize_function(example, tokenizer),
        )


    @staticmethod 
    def collate_fn(input_batch):
        input_ids_1 = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["input_ids_1"]) for x in input_batch], batch_first=True)
        attention_mask_1 = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["attention_mask_1"]) for x in input_batch], batch_first=True)
        input_ids_2 = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["input_ids_2"]) for x in input_batch], batch_first=True)
        attention_mask_2 = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["attention_mask_2"]) for x in input_batch], batch_first=True)
        
        if "token_type_ids_1" in input_batch[0]:
            token_type_ids_1 = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["token_type_ids_1"]) for x in input_batch], batch_first=True)
            token_type_ids_2 = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["token_type_ids_2"]) for x in input_batch], batch_first=True)
        else:
            token_type_ids_1 = None
            token_type_ids_2 = None
        
        batch = {
            "input_ids_1": input_ids_1,
            "attention_mask_1": attention_mask_1,
            "input_ids_2": input_ids_2,
            "attention_mask_2": attention_mask_2,
            "token_type_ids_1": token_type_ids_1,
            "token_type_ids_2": token_type_ids_2,
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

    ds = SB_Dataset("2-ways")
    train_ds = ds.data_dict["train"]
    train_contrastive = get_contrastive(train_ds)
    with open("data/contrastive_train_2ways.json", "w") as f:
        json.dump(train_contrastive, f, indent=4)
    

    