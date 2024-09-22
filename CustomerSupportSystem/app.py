from fastapi import FastAPI
from agents.greeter_agent import GreeterAgent
from agents.faq_agent import FAQAgent
from agents.escalation_agent import EscalationAgent

# Create the FastAPI app
app = FastAPI()

# Initialize the agents
greeter = GreeterAgent()
faq_agent = FAQAgent()
escalation_agent = EscalationAgent()

# Define the /greet endpoint
@app.get("/greet")
def greet_user():
    return greeter.greet()

# Define the /faq endpoint
@app.get("/faq")
def get_faq_answer(question: str):
    print(escalation_agent.get_escalation_log())
    return {"answer": faq_agent.get_answer(question)}

@app.get("/escalation-log")
def get_escalation_log():
    print(escalation_agent.get_escalation_log())
    # Endpoint to view all escalated issues (for admin review)
    return {"escalated_issues": escalation_agent.get_escalation_log()}