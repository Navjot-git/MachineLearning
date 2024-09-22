from fastapi import FastAPI
from agents.greeter_agent import GreeterAgent
from agents.faq_agent import FAQAgent

# Create the FastAPI app
app = FastAPI()

# Initialize the Greeter Agent
greeter = GreeterAgent()
faq_agent = FAQAgent()

# Define the /greet endpoint
@app.get("/greet")
def greet_user():
    return greeter.greet()

# Define the /faq endpoint
@app.get("/faq")
def get_faq_answer(question: str):
    return {"answer": faq_agent.get_answer(question)}