import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from evaluate import load
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import Trainer, TrainingArguments
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

# print(f"Tokenized Sentences: {tokenized_sentences[:5]}")  # Display the first 5 sentences


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


# print(tokenized_dataset['train']['article'][0])
# print(tokenized_dataset['train']['highlights'][0])
# print(tokenized_dataset['train']['id'][0])
# print(tokenized_dataset['train']['input_ids'][0])
# print(tokenized_dataset['train']['attention_mask'][0])
# print(tokenized_dataset['train']['labels'][0])

# Load the pre-trained BART model
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn", attn_implementation="eager")

# Define training arguments 
# training_args = TrainingArguments(
#     output_dir="./results",         # Output directory for model checkpoints and logs
#     eval_strategy="epoch",    # Evaluate at the end of each epoch
#     learning_rate=2e-5,             # Learning rate
#     per_device_train_batch_size=8,  # Batch size for training
#     per_device_eval_batch_size=8,   # Batch size for evaluation
#     num_train_epochs=3,             # Number of training epochs
#     weight_decay=0.01,              # Weight decay for regularization
#     logging_dir="./logs",           # Directory for logging
#     logging_steps=100,              # Log every 100 steps
#     save_total_limit=2,             # Only save the last 2 checkpoints
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset["train"],       # Training dataset
#     eval_dataset=tokenized_dataset["validation"]    # Validation dataset
# )
# print("Training...")
# trainer.train()
# print("Evaluating...")
# trainer.evaluate()

# Generating summaries (Inference)

article = dataset['test']['article'][0]
# Tokenize the article
inputs = tokenizer(article, max_length=1024, return_tensors='pt', truncation=True)
# Generate the summary and capture attention weights
outputs = model(
    inputs['input_ids'], 
    attention_mask=inputs["attention_mask"],
    output_attentions=True,
    return_dict=True,
    )
outputs2 = model.generate(
    inputs['input_ids'], 
    max_length=150
    )
print(outputs[0], outputs2[0])
# Decode the generated summary
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(article)
# print(inputs)
# print(outputs)
print(f"Original Article:\n{article[:500]}...")  # Display part of the article
print(f"Generated Summary:\n{summary}")


# Load the ROUGE metic: Recall-Oriented Understudy for Gisting Evaluation
rouge = load("rouge")

# Funcion to compute ROUGE scores
def compute_rouge(predictions, references):
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(references, str):
        references = [references]
    rouge_output = rouge.compute(predictions=predictions, references=references)
    return rouge_output

generated_summary="The Palestinian Authority becomes the 123rd member of the International Criminal Court. The move gives the court jurisdiction over alleged crimes in Palestinian territories. Israel and the United States opposed the Palestinians' efforts to join the body. But Palestinian Foreign Minister Riad al-Malki said it was a move toward greater justice."
reference_summary=dataset['test']['highlights'][0]
print(reference_summary)
# Compute the ROUGE scores
rouge_scores = compute_rouge(generated_summary,reference_summary)

# Print the results
print(rouge_scores)

# Extract sequences and attentions
attentions = outputs.attentions # This will give you a list of attention matrices from each layer

# Display the shape of attention maps
for idx, attn in enumerate(attentions):
    print(f"Layer {idx + 1} attention shape: {attn.shape}")
    
    
# Select the attention map for the first layer and first head
attention_map = attentions[0][0].detach().numpy()  # Get the attention from the first layer, first head


# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(attention_map, cmap="viridis")
plt.title("Attention Map for Layer 1, Head 1")
plt.show()
