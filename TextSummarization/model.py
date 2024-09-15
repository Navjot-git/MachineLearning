from datasets import load_dataset
from transformers import BartTokenizer
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt_tab') # Tokenizer for sentence splitting

# Load the CNN/Daily Mail dataset from Hugging Face
dataset = load_dataset("cnn_dailymail", '3.0.0')

# View the structure of the dataset
print(dataset)

# Take a look at a sample from the training set
# print(f"Article: {dataset['train']['article'][0]}")
# print(f"Summary: {dataset['train']['highlights'][0]}")

# Example article from the dataset
sample_article = dataset['train']['article'][0]

# Tokenize the article into sentences
tokenized_sentences = sent_tokenize(sample_article)

print(f"Tokenized Sentences: {tokenized_sentences[:5]}")  # Display the first 5 sentences


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove leading/trailing whitespaces
    text = text.strip()
    return text

# Preprocess the article and summary
preprocessed_article = preprocess_text(sample_article)
preprocessed_summary = preprocess_text(dataset['train']['highlights'][0])

# print(f"Preprocessed Article: {preprocessed_article[:500]}")  # Display the first 500 characters

# Load the tokenizer for BART
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
# Get the number of tokens in the tokenizer
num_tokens = len(tokenizer)

print(f"Number of tokens in BART tokenizer: {num_tokens}")
# Tokenize and truncate the article
inputs = tokenizer(preprocessed_article, max_length=1024, return_tensors="pt", truncation=True)

print(f"Tokenized Input Shape: {inputs['input_ids'].shape}")
# print(inputs)

def preprocess_batch(examples):
    # Apply preprocessing on articles and summaries
    articles = [preprocess_text(article) for article in examples['article']]
    summaries = [preprocess_text(summary) for summary in examples['highlights']]
    
    # Tokenize and truncate
    inputs = tokenizer(articles, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
    targets = tokenizer(summaries, max_length=150, truncation=True, padding="max_length", return_tensors="pt")
    
    return {
        "input_ids": inputs['input_ids'],
        "attention_mask": inputs['attention_mask'],
        "labels": targets['input_ids']
    }
    
# Apply the function to the dataset
tokenized_dataset = dataset.map(preprocess_batch, batched=True)
    
print(tokenized_dataset)


print(tokenized_dataset['train']['article'][0])
print(tokenized_dataset['train']['highlights'][0])
print(tokenized_dataset['train']['id'][0])
print(tokenized_dataset['train']['input_ids'][0])
print(tokenized_dataset['train']['attention_mask'][0])
print(tokenized_dataset['train']['labels'][0])