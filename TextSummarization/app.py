from flask import Flask, render_template, request
from transformers import BartTokenizer, BartForConditionalGeneration
from fuzzywuzzy import fuzz
from evaluate import load
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')  # Necessary for the sent_tokenize function

app = Flask(__name__)

# Load pre-trained model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
rouge = load("rouge")

def compute_rouge(predictions, references):
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(references, str):
        references = [references]
    rouge_output = rouge.compute(predictions=predictions, references=references)
    return rouge_output

# Function to highlight matching sentences
def highlight_matches(article, summary):
    article_sentences = sent_tokenize(article)  # Tokenize article into sentences
    summary_sentences = sent_tokenize(summary)  # Tokenize summary into sentences
    
    highlighted_article = article
    highlighted_summary = summary

    # Iterate through each sentence in the summary and match with the article using fuzzy matching
    for summary_sentence in summary_sentences:
        for article_sentence in article_sentences:
            match_ratio = fuzz.ratio(summary_sentence, article_sentence)

            if match_ratio > 40:  # Adjust this threshold for more or less strict matching
                highlighted_article = highlighted_article.replace(article_sentence, f'<span class="highlight">{article_sentence}</span>')
                highlighted_summary = highlighted_summary.replace(summary_sentence, f'<span class="highlight">{summary_sentence}</span>')
    
    return highlighted_article, highlighted_summary



@app.route("/", methods=["GET", "POST"])
def summarize():
    summary = ""
    article_text = ""
    highlighted_article = ""
    highlighted_summary = ""
    
    if request.method == "POST":
        article_text = request.form['article']
        inputs = tokenizer(article_text, max_length=1024, return_tensors="pt", truncation=True)
        summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Highlight matching sentences between the article and summary
        highlighted_article, highlighted_summary = highlight_matches(article_text, summary)
    
    return render_template("index.html", summary=summary, highlighted_article=highlighted_article, highlighted_summary=highlighted_summary, article_text=article_text)

if __name__ == "__main__":
    app.run(debug=True)
