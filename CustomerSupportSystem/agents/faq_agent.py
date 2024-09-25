from transformers import DistilBertTokenizer, DistilBertModel, pipeline
import torch
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class FAQAgent:
    def __init__(self, escalation_agent):
        # Check if MPS is available (for Apple Silicon)
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print(device)

        # Load the DistilBERT model and tokenizer for embedding-based search
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

        # Load the pre-trained pipeline for question-answering if needed
        self.qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", device=0 if device.type == "mps" else -1)
        
        self.escalation_agent = escalation_agent
        self.escalation_flag = False 

        # Load FAQ data from the JSON file
        with open("faq_data.json", "r") as f:
            self.faq_data = json.load(f)["faqs"]

        # Precompute FAQ question embeddings
        self.faq_embeddings = [self.get_embedding(faq['question']) for faq in self.faq_data]

    # Function to generate embeddings for a given question
    def get_embedding(self, question):
        inputs = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    # Function to find the closest FAQ question using cosine similarity
    def find_most_similar_faq(self, question):
        user_embedding = self.get_embedding(question)
        similarities = [cosine_similarity(user_embedding, faq_emb)[0][0] for faq_emb in self.faq_embeddings]
        most_similar_idx = np.argmax(similarities)
        return self.faq_data[most_similar_idx]

    def get_answer(self, question):
        # Find the most similar FAQ question
        similar_faq = self.find_most_similar_faq(question)
        print(f"Most similar FAQ: {similar_faq['question']}")

        # Use the matched FAQ's answer as the context for question-answering if needed
        context = similar_faq['answer']

        # Use the question-answering pipeline to get the best answer
        response = self.qa_pipeline(question=question, context=context)
        print(question, response)

        # If confidence is low, escalate the issue
        if response['score'] < 0.25:  # Threshold for low confidence
            self.escalation_flag = True
            return self.escalation_agent.escalate_issue(question)

        # Return the answer with the highest score
        # return response["answer"]
        # Instead of using the QA model, return the exact answer
        return similar_faq['answer']

    def has_escalated(self):
        return self.escalation_flag
