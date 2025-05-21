import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

import torch
from transformers import RobertaModel, BertModel, AutoModel,T5Model
from torch.nn import CrossEntropyLoss
from dataclasses import dataclass
from typing import Optional, Tuple
import re 

def get_tokenizer(model_name: str) -> AutoTokenizer:

    return AutoTokenizer.from_pretrained(model_name)
@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states
class ASAG_Encoder(nn.Module):
    """
    Encoder based ASAG model
    """
    def __init__(self, model_name: str, num_labels: int, freeze_layers: int = 0, freeze_embeddings: bool = False):
    
        super().__init__()

        self.model_name = model_name 
        self.encoder = AutoModel.from_pretrained(model_name)
        if "t5" in model_name:
            self.encoder = self.encoder.get_encoder()
        hidden_size = self.encoder.config.hidden_size
        self.classifier = ClassificationHead(hidden_size, num_labels)
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
        if "t5" in self.model_name:
            return self.forward_t5(input_ids, attention_mask, label_id)
 
        encoder_outputs = self.encoder(
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
    def forward_t5(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_id: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        """
        Forward method for T5 model. Extracts the last EOS token as the sequence representation.
        """
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = encoder_outputs.last_hidden_state

        # Extract the position of </s> token (id = tokenizer.eos_token_id)
        eos_token_id = self.encoder.config.eos_token_id
        eos_mask = input_ids.eq(eos_token_id)
        batch_size, hidden_size = sequence_output.size(0), sequence_output.size(-1)
        sequence_output = sequence_output[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]

        logits = self.classifier(sequence_output)

        loss = None
        if  label_id is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), label_id.view(-1))

        return ModelOutput(logits=logits, loss=loss)
    def freeze_layers(self, n_frozen_layers: int):
        """
        Freeze the specified number of layers in the encoder.
        """

        regex = re.compile(r"(\d+)\.layer")
        for name, param in self.encoder.named_parameters():
            match = regex.search(name)
            if match:
                layer_num = int(match.group(1))
                if layer_num < n_frozen_layers:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
    def freeze_embeddings(self):
        """
        Freeze the embedding layer.
        """
        if hasattr(self.encoder, "embeddings"):
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
        if hasattr(self.encoder, "embed_tokens"):
            for param in self.encoder.embed_tokens.parameters():
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
        self.classifier = ClassificationHead(hidden_size * n_multi, num_labels)
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
    # Import T5 tokenizer
    from transformers import T5Tokenizer
    # Define model name and tokenizer
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    batched_text = [
        ["Hello, how are you?", "I am fine, thank you."],
        ["What is your name?", "My name is T5."],
        ["Where are you from?", "I am from the internet."],
        ["What do you do?", "I help with natural language processing tasks."],
        ["How old are you?", "I am ageless."],
        
    ]
    encoding = tokenizer(
        batched_text,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )
    model = ASAG_Encoder(model_name, num_labels=2)
    output = model(**encoding)
