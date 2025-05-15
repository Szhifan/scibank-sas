import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

import torch
from transformers import RobertaModel, BertModel, AutoModel
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
from typing import Optional, Tuple
MODELMAP = {
    "roberta": "roberta-base",
    "bert": "bert-base-uncased",
    "ms-marco-12":"cross-encoder/ms-marco-MiniLM-L12-v2",
    "ms-marco-6":"cross-encoder/ms-marco-MiniLM-L6-v2",
    "nli-deberta-v3-large": "cross-encoder/nli-deberta-v3-large",
}
def get_tokenizer(model_type: str) -> AutoTokenizer:
    if model_type not in MODELMAP:
        raise ValueError(f"Model type {model_type} not supported. Choose from {list(MODELMAP.keys())}.")
    return AutoTokenizer.from_pretrained(MODELMAP[model_type])
@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
class ClassifierHead(nn.Module):
    def __init__(self, config, num_labels: int):
     
        super(ClassifierHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dense(features)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
class ASAG_CrossEncoder(nn.Module):
    def __init__(self, model_type: str, num_labels: int, freeze_layers: int = 0, freeze_embeddings: bool = False):
    
        super(ASAG_CrossEncoder, self).__init__()
        if model_type not in MODELMAP:
            raise ValueError(f"Model type {model_type} not supported. Choose from {list(MODELMAP.keys())}.")
        
        self.bert = AutoModel.from_pretrained(MODELMAP[model_type])
        self.classifier = ClassifierHead(self.bert.config, num_labels)
        self.num_labels = num_labels
        if freeze_layers > 0:
            self.freeze_layers(freeze_layers) 
        if freeze_embeddings:
            self.freeze_embeddings()

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        token_type_ids: Optional[torch.Tensor] = None, 
        label_id: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        encoder_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = encoder_outputs.last_hidden_state
        logits = self.classifier(sequence_output[:, 0, :])  # Use [CLS] token representation

        loss = None
        if label_id is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), label_id.view(-1))

        return ModelOutput(logits=logits, loss=loss)
    def freeze_layers(self, n_frozen_layers: int):
        """
        Freeze the specified number of layers in the encoder.
        """
        for i, param in enumerate(self.bert.encoder.layer.parameters()):
            if i < n_frozen_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True
    def freeze_embeddings(self):
        """
        Freeze the embedding layer.
        """
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

    
if __name__ == "__main__":
    model = ASAG_CrossEncoder(model_type="ms-marco-12", num_labels=3, freeze_layers=0, freeze_embeddings=False)
    tok = get_tokenizer("ms-marco-12")
    inputs = tok("This is a test sentence.","nimadabi", return_tensors="pt")
    outputs = model(**inputs)
    print(outputs)