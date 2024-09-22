import json

class FAQAgent:
    def __init__(self):
        # Load FAQ data from the JSON file
        with open("faq_data.json", "r") as f:
            self.faq_data = json.load(f)["faqs"]
            
    def get_answer(self, question):
        # Simple string matching
        for faq in self.faq_data:
            if question.lower() in faq["question"].lower():
                return faq["answer"]
        return "Sorry, I don't know the answer to that yet!"