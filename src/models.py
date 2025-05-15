import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

import torch
from transformers import RobertaModel, BertModel, AutoModel
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
from typing import Optional, Tuple

def get_tokenizer(model_name: str) -> AutoTokenizer:

    return AutoTokenizer.from_pretrained(model_name)
@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None

class ASAG_CrossEncoder(nn.Module):
    def __init__(self, model_name: str, num_labels: int, freeze_layers: int = 0, freeze_embeddings: bool = False):
    
        super(ASAG_CrossEncoder, self).__init__()

        
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )
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


def mean_pooling(
    model_output: ModelOutput, 
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Perform mean pooling on the model output.
    """
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask            
class ASAG_SentenceEmbeddings(nn.Module):
    def __init__(self,model_name: str, num_labels: int, freeze_layers: int = 0, freeze_embeddings: bool = False, use_multiplication = False):
        super().__init__()
        self.use_multiplication = use_multiplication
        self.se = AutoModel.from_pretrained(model_name)
        hidden_size = self.se.config.hidden_size
        n_multi = 4 if use_multiplication else 3
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * n_multi, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )
        self.num_labels = num_labels
        if freeze_layers > 0:
            self.freeze_layers(freeze_layers)
        if freeze_embeddings:
            self.freeze_embeddings()
    def get_se(self,input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None) -> ModelOutput:
        encoder_outputs = self.se(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = mean_pooling(encoder_outputs, attention_mask)
        pooled_output = torch.nn.functional.normalize(pooled_output, p=2, dim=1)
        return pooled_output 
    def forward(
        self,
        input_ids_1: torch.Tensor,
        attention_mask_1: torch.Tensor,
        input_ids_2: torch.Tensor,
        attention_mask_2: torch.Tensor,
        token_type_ids_1: Optional[torch.Tensor] = None,
        token_type_ids_2: Optional[torch.Tensor] = None,
        label_id: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        emb_1 = self.get_se(input_ids_1, attention_mask_1, token_type_ids_1)
        emb_2 = self.get_se(input_ids_2, attention_mask_2, token_type_ids_2)

        features = [emb_1, emb_2, torch.abs(emb_1 - emb_2)]
        if self.use_multiplication:
            features = features + [emb_1 * emb_2]
        features = torch.cat(features, dim=1)

        logits = self.classifier(features)
        loss = None
        if label_id is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), label_id.view(-1))
        return ModelOutput(logits=logits, loss=loss)
    def freeze_layers(self, n_frozen_layers: int):
        """
        Freeze the specified number of layers in the encoder.
        """
        for i, param in enumerate(self.se.encoder.layer.parameters()):
            if i < n_frozen_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True
    def freeze_embeddings(self):    
        """
        Freeze the embedding layer.
        """
        for param in self.se.embeddings.parameters():
            param.requires_grad = False
        
    
if __name__ == "__main__":
    model = ASAG_SentenceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        num_labels=3,
        freeze_layers=0,
        freeze_embeddings=False,
        use_multiplication=True
    )
    tokenizer = get_tokenizer("sentence-transformers/all-MiniLM-L6-v2")
    text_1 = "This is the first sentence."
    text_2 = "This is the second sentence."
    inputs_1 = tokenizer(text_1, return_tensors="pt", padding=True, truncation=True)
    inputs_2 = tokenizer(text_2, return_tensors="pt", padding=True, truncation=True)
    outputs = model(
        input_ids_1=inputs_1['input_ids'],
        attention_mask_1=inputs_1['attention_mask'],
        input_ids_2=inputs_2['input_ids'],
        attention_mask_2=inputs_2['attention_mask']
    )
    print("Logits:", outputs.logits)