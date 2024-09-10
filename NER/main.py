from datasets import load_dataset
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

text = "Elon Musk founded SpaceX and Tesla in the United States"

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
    
# Print the merged entities
for word, label in final_entities:
    print(f"{word} -> {label}")