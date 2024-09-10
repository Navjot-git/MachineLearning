from datasets import load_dataset, load_metric
import spacy
from transformers import pipeline
from seqeval.metrics import classification_report

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

# Example sentences
sentences = [
    "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
    "Google was founded by Larry Page and Sergey Brin.",
    "Barack Obama was born in Hawaii."
]

# Ground truth (expected) entities for each sentence (with "O" labels)
expected_entities_list = [
    [('Apple', 'B-ORG'), ('Inc', 'I-ORG'), ('Steve', 'B-PER'),
     ('Jobs', 'I-PER'), ('Cupertino', 'B-LOC'), ('California', 'I-LOC')],
    [('Google', 'B-ORG'), ('Larry', 'B-PER'), ('Page', 'I-PER'),
     ('Sergey', 'B-PER'), ('Brin', 'I-PER')],
    [('Barack', 'B-PER'), ('Obama', 'I-PER'), ('Hawaii', 'B-LOC')]
]

# Store the predicted and true labels
true_labels = []
pred_labels = []

# Create a NER pipeline using a pre-trained BERT model
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

for sentence, expected_entities in zip(sentences, expected_entities_list):

    entities = ner_pipeline(sentence)

    # for entity in entities:
    #     print(f"{entity['word']} -> {entity['entity']}")
        
    # Initialize variables for merging subword tokens
    final_entities = []
    current_word = ""
    current_label = None

    # Iterate over the entities
    for entity in entities:
        print(entity)
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
        
    # Extract just the labels (ignoring the words) for evaluation
    true_labels.append([label for _, label in expected_entities])
    print(true_labels)
    print(expected_entities)
    pred_labels.append([label for _, label in adjusted_entities])
    print(pred_labels)
    print(adjusted_entities)
        
        
# Evaluate using seqeval's classification report
print(classification_report(true_labels, pred_labels))
# # Define a sample sentence with expected entities
# expected_entities = [
#     ('Apple', 'B-ORG'),
#     ('Inc', 'I-ORG'),
#     ('Steve', 'B-PER'),
#     ('Jobs', 'I-PER'),
#     ('Cupertino', 'B-LOC'),
#     ('California', 'I-LOC')
# ]

# # Compare model's entities with expected ones
# correct = 0
# for word, label in adjusted_entities:
#     if (word, label) in expected_entities:
#         correct += 1

# # Calculate simple accuracy
# accuracy = correct / len(expected_entities)
# print(f"Accuracy: {accuracy:.2f}")