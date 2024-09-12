from datasets import load_dataset
import spacy
from spacy import displacy

# Load the CoNLL-2003 dataset
dataset = load_dataset("conll2003", trust_remote_code=True)

# print(dataset["train"][0])
# print("----------")
# print(dataset["train"].features)
# print("----------")

# Access the label names for NER tags
labels = dataset["train"].features['ner_tags'].feature.names
print(labels)

print("---------------NER MODEL USING SPACY------------------")
# Load the pre-trained small English model
nlp = spacy.load("en_core_web_sm")

# Get the NER component
ner = nlp.get_pipe("ner")

# Print all the entity labels
print("Entities the model can recognize:")
for label in ner.labels:
    print(label)

# Example text
text = "Apple is looking at buying a U.K. startup for $1 billion"

# Process the text using SpaCy
doc = nlp(text)

# Extract and print named entities
print("Entities in the text:")
for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")
print("****************************") 
# Extract entities, nouns, verbs, and other POS tags
for token in doc:
    if token.ent_type_:
        print(f"{token.text}: {token.ent_type_} (Entity)")
    elif token.pos_ == 'NOUN':
        print(f"{token.text}: {token.pos_} (Noun)")
    elif token.pos_ == 'VERB':
        print(f"{token.text}: {token.pos_} (Verb)")
    elif token.pos_ == 'ADJ':
        print(f"{token.text}: {token.pos_} (Adjective)")
    elif token.pos_ == 'PRON':
        print(f"{token.text}: {token.pos_} (Pronoun)")
    elif token.pos_ == 'ADV':
        print(f"{token.text}: {token.pos_} (Adverb)")
    else:
        print(f"{token.text}: (Other)")
        
        
# # Visualize named entities
# displacy.render(doc, style="ent", jupyter=True)


# Save the visualization to an HTML file
html = displacy.render(doc, style="ent")
with open("entities.html", "w") as f:
    f.write(html)