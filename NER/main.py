from datasets import load_dataset, load_metric
import spacy
from transformers import pipeline

# Load the CoNLL-2003 dataset
dataset = load_dataset("conll2003", trust_remote_code=True)

print(dataset["train"][0])
print("----------")
print(dataset["train"].features)
print("----------")

# Access the label names for NER tags
labels = dataset["train"].features['ner_tags'].feature.names
print(labels)

print("---------------NER MODEL USING SPACY------------------")
# Load the pre-trained small English model
nlp = spacy.load("en_core_web_sm")

# Example text
text = "Apple is looking at buying a U.K. startup for $1 billion"

# Process the text using SpaCy
doc = nlp(text)

# Extract and print named entities
print("Entities in the text:")
for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")
    
    
print("---------------NER MODEL USING HUGGING FACE TRANSFORMERS------------------")

# Create a NER pipeline using a pre-trained BERT model
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."

entities = ner_pipeline(text)

# for entity in entities:
#     print(f"{entity['word']} -> {entity['entity']}")
    
# Initialize variables for merging subword tokens
final_entities = []
current_word = ""
current_label = None

# Iterate over the entities
for entity in entities:
    word = entity['word']
    label = entity['entity']
    
    # Check if the token is a subword (starts with ##)
    if word.startswith('##'):
        current_word += word[2:]  # Remove "##" aand append the subword
    else:
        # If a new word starts, append the previous word and label
        if current_word:
            final_entities.append((current_word, current_label))
        # Start a new word
        current_word = word
        current_label = label
        
# Append the last word
if current_word:
    final_entities.append((current_word, current_label))
    
# Adjust labels to handle B- and I- for multi-token entities
adjusted_entities = []
prev_label = None

for i, (word, label) in enumerate(final_entities):
    if label.startswith("I") and (prev_label is None or prev_label[2:] != label[2:]):
        # Convert the label to B if it's the start of a new entity
        label = "B" + label[1:]
    
    adjusted_entities.append((word, label))
    prev_label = label  # Update the previous label
    
# Print the merged entities
for word, label in adjusted_entities:
    print(f"{word} -> {label}")
    
# Define a sample sentence with expected entities
expected_entities = [
    ('Apple', 'B-ORG'),
    ('Inc', 'I-ORG'),
    ('Steve', 'B-PER'),
    ('Jobs', 'I-PER'),
    ('Cupertino', 'B-LOC'),
    ('California', 'I-LOC')
]

# Compare model's entities with expected ones
correct = 0
for word, label in adjusted_entities:
    if (word, label) in expected_entities:
        correct += 1

# Calculate simple accuracy
accuracy = correct / len(expected_entities)
print(f"Accuracy: {accuracy:.2f}")