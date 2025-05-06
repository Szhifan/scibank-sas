from transformers import BertModel

# Load a pre-trained BERT model
model = BertModel.from_pretrained("bert-base-uncased")

encoder = model.encoder
for name, param in encoder.named_parameters():
    print(name, param.requires_grad)