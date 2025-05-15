from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch

model = AutoModel.from_pretrained('cross-encoder/nli-deberta-v3-large')
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-large')

features = tokenizer(['How many people live in Berlin?', 'How many people live in Berlin?'], ['Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 'New York City is famous for the Metropolitan Museum of Art.'],  padding=True, truncation=True, return_tensors="pt")

model.eval()

with torch.no_grad():
    model_output = model(**features)
print(model_output)