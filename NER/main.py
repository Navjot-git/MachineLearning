from datasets import load_dataset

# Load the CoNLL-2003 dataset
dataset = load_dataset("conll2003", trust_remote_code=True)

print(dataset["train"][0])
print("")
print(dataset["train"][1])
print(dataset)

# Access the label names for NER tags
labels = dataset["train"].features['ner_tags'].feature.names
print(labels)

