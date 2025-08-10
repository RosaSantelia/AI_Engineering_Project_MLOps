def preprocess(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
