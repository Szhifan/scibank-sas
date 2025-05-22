from transformers import AutoTokenizer, T5ForConditionalGeneration
from data_prep import SB_Dataset_conditional_generation
import torch
from torch.utils.data import DataLoader
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
print([tokenizer.decode(i) for i in tokenizer("contradictory")])