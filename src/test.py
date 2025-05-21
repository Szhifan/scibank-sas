from transformers import T5ForSequenceClassification, T5Tokenizer

text = "The quick brown fox jumps over the lazy dog."
model_name = "google-t5/t5-small"
# Load the pre-trained T5 model and tokenizer
model = T5ForSequenceClassification.from_pretrained(model_name)
for name, p in model.named_parameters():

    print(name, p.shape)