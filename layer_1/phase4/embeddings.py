import torch 
from transformers import AutoTokenizer,AutoModel

model_name = "bert-base-uncased"


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "The financial market is volatile."

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

print("Input IDs:")
print(inputs["input_ids"])

print("\nHidden state shape:")
print(outputs.last_hidden_state.shape)

### Embedding layer

embedding_layer = model.embeddings.word_embeddings

print("Embedding matrix shape:")
print(embedding_layer.weight.shape)

token_id = inputs["input_ids"][0][1] ## converting words to token IDs

vector = embedding_layer(token_id)  ## getting vector

print("Token ID:", token_id.item())
print("Vector shape:", vector.shape)
print("First 10 values:", vector[:10])

'''You have just performed:

Token ID
   ↓
Embedding matrix
   ↓
Vector

This is the actual mechanism.'''

words = ["cat", "dog", "car", "banana"]

for word in words:
    token_id = tokenizer.convert_tokens_to_ids(word)
    vector = embedding_layer.weight[token_id]

    print(word)
    print("ID:", token_id)
    print("Dimension:", vector.shape)
    print()


## Similarity in Embedding
import torch.nn.functional as F

def get_embedding(word):
    token_id = tokenizer.convert_tokens_to_ids(word)
    return embedding_layer.weight[token_id]

cat = get_embedding("cat")
dog = get_embedding("dog")
car = get_embedding("car")

print("cat-dog:",
      F.cosine_similarity(cat.unsqueeze(0),
                          dog.unsqueeze(0)).item())

print("cat-car:",
      F.cosine_similarity(cat.unsqueeze(0),
                          car.unsqueeze(0)).item())


## Sentence embedding
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "How can I calculate compound interest?",
    "What is the formula for interest growth?",
    "The football match starts at 8 PM."
]

embeddings = model.encode(sentences)

print(embeddings.shape)                           
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(embeddings)

print(similarities)