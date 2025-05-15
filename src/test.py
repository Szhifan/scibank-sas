from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# Load base HF model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

text = "This is a test sentence."

# Match SentenceTransformer behavior
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    pooled_output = mean_pooling(outputs, inputs['attention_mask'])
    pooled_output = torch.nn.functional.normalize(pooled_output, p=2, dim=1)

# SentenceTransformer output
model_se = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
with torch.no_grad():
    embeddings = model_se.encode(text, convert_to_tensor=True)

# Print difference
print("Manual output:", pooled_output[0][:5])
print("SentenceTransformer output:", embeddings[:5])
print("Cosine similarity:", torch.nn.functional.cosine_similarity(pooled_output, embeddings.unsqueeze(0)))
