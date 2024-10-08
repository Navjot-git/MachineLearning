from datasets import load_dataset, load_metric
import NER.model as model
from transformers import pipeline, BertTokenizerFast
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
nlp = model.load("en_core_web_sm")

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
    # Sentence 1: "Apple Inc. was founded by Steve Jobs in Cupertino, California."
    [('Apple', 'B-ORG'), ('Inc', 'I-ORG'), ('.', 'O'),
     ('was', 'O'), ('founded', 'O'), 
     ('by', 'O'), ('Steve', 'B-PER'), ('Jobs', 'I-PER'), ('in', 'O'),
     ('Cupertino', 'B-LOC'), (',', 'O'), ('California', 'I-LOC'), ('.', 'O')],
    
    # Sentence 2: "Google was founded by Larry Page and Sergey Brin."
    [('Google', 'B-ORG'), ('was', 'O'), ('founded', 'O'), ('by', 'O'), 
     ('Larry', 'B-PER'), ('Page', 'I-PER'), ('and', 'O'), ('Sergey', 'B-PER'), 
     ('Brin', 'I-PER'), ('.', 'O')],
    
    # Sentence 3: "Barack Obama was born in Hawaii."
    [('Barack', 'B-PER'), ('Obama', 'I-PER'), ('was', 'O'), 
     ('born', 'O'), ('in', 'O'), ('Hawaii', 'B-LOC'), ('.', 'O')]
]

# Store the predicted and true labels
true_labels = []
pred_labels = []

# Load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
# Create a NER pipeline using a pre-trained BERT model
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

for sentence, expected_entities in zip(sentences, expected_entities_list):
    
    # Tokenize the sentence to get all tokens including punctuation and non-entity tokens
    tokenized_sentence = tokenizer(sentence, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(tokenized_sentence["input_ids"])
    offsets = tokenized_sentence["offset_mapping"] # Offset mapping
    entities = ner_pipeline(sentence)
    # Debug: Print the tokenized sentence and offsets
    print("Tokenized Sentence:", tokenized_sentence)
    print("Tokens:", tokens)
    print("Offsets:", offsets)
    for entity in entities:
        print(f"{entity['word']} -> {entity['entity']}")
    
    # Initialize variables for merging subword tokens
    final_entities = []
    current_word = ""
    current_label = None
    entity_index = 0 # To track index of NER entity
    prev_label = None
    # Iterate over the tokens and their offsets
    for i, (token, offset) in enumerate(zip(tokens, offsets)):
        # Skip special tokens
        if token in ['[CLS]', '[SEP]']:
            continue
        
        # Handle subword tokens (starts with ##)
        if token.startswith('##'):
            current_word += token[2:]  # Remove "##" aand append the subword
        else:
            # If a new word starts, append the previous word and label
            if current_word:
                final_entities.append((current_word, current_label))
            # Start a new word
            current_word = token
            current_label = "O" # Default label for non-entities
            
            # Check if this token aligns with any entity (based on offset mapping)
            if entity_index < len(entities):
                entity = entities[entity_index]
                entity_start = entity['start']
                entity_end = entity['end']
                
                 # If token is within entity boundaries, assign the entity label
                if offset[0] >= entity_start and offset[1] <= entity_end:
                    current_label = entity['entity']
                    
                    # Convert I- to B- if it's a new entity
                    if prev_label is None or prev_label[2:] != current_label[2:]:
                        current_label = "B" + current_label[1:]
                    else:
                        current_label = "I" + current_label[1:]
                    
                    prev_label = current_label

                # Move to the next entity when the token offset matches the end of the entity
                if offset[1] == entity_end:
                    entity_index += 1          
    # Append the last word
    if current_word:
        final_entities.append((current_word, current_label))
        
    # Adjust labels to handle B- and I- for multi-token entities
    adjusted_entities = []
    prev_label = None

    for i, (word, label) in enumerate(final_entities):
        # Check if it's a new entity or the first token
        if label.startswith("I") and (prev_label is None or prev_label[2:] != label[2:]):
            # Convert the label to B if it's the start of a new entity
            label = "B" + label[1:]
        
        # Handle the case where we should switch to B- when the label changes
        if prev_label is None or prev_label[2:] != label[2:]:
            if label.startswith("I"):
                label = "B" + label[1:]  # Switch to B- if it's a new entity

        adjusted_entities.append((word, label))
        prev_label = label  # Update the previous label

        
    # Print the merged entities
    for word, label in adjusted_entities:
        print(f"{word} -> {label}")
        
    # Extract just the labels (ignoring the words) for evaluation
    true_labels.append([label for _, label in expected_entities])
    pred_labels.append([label for _, label in adjusted_entities])

print("##########################################")     
print(true_labels)
print(pred_labels)
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