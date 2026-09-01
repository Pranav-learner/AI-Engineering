from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

text = "The financial market is volatile"

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print("Tokens: ")
print(tokens)

print("\nToken IDs: ")
print(token_ids)

print("\nToken count:") 
print(len(token_ids))

# Token -> ID -> Token
for token_id in token_ids:
    token = tokenizer.decode([token_id])
    print(token_id, "->", repr(token))


## experiment comparing different inputsl
texts = [
    "hello world",
    "unbelievable",
    "OpenAI builds AI systems.",
    "123456789",
    "def calculate_interest(principal, rate):",
    "भारत एक देश है",
    "🚀 AI engineering"
]

for text in texts:
    tokens = tokenizer.tokenize(text)

    print("\nTEXT:", text)
    print("TOKENS:", tokens)
    print("COUNT:", len(tokens))