import json
from transformers import pipeline
from agents.escalation_agent import EscalationAgent

class FAQAgent:
    def __init__(self, escalation_agent):
        # Load the pre-trained model for question-answering
        self.qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
        self.escalation_agent = escalation_agent
        # Load FAQ data from the JSON file
        with open("faq_data.json", "r") as f:
            self.faq_data = json.load(f)["faqs"]
            
    def get_answer(self, question):
        # Combine all FAQ answers as the context for the QA model
        context = " ".join([faq["answer"] for faq in self.faq_data])
        
        # Use the model to find the most relevant answer
        response = self.qa_pipeline(question=question, context=context)
        print(question, context)
        print(response['score'])
        # If confidence is low, escalate the issue
        if response['score'] < 0.6: # Threshold for low confidence
            return self.escalation_agent.escalate_issue(question)
        
        # Return the answer with the highest score
        return response["answer"]