from fastapi import FastAPI, Request
from agents.greeter_agent import GreeterAgent
from agents.faq_agent import FAQAgent
from agents.escalation_agent import EscalationAgent
from agents.feedback_agent import FeedbackAgent

# Create the FastAPI app
app = FastAPI()

# Initialize the agents
escalation_agent = EscalationAgent()
greeter = GreeterAgent()
faq_agent = FAQAgent(escalation_agent) # FAQ agent passes escalation agent
feedback_agent = FeedbackAgent(faq_agent, escalation_agent) # Feedback agent checks FAQ agent's escalation status

# Define the /greet endpoint
@app.get("/greet")
def greet_user():
    return greeter.greet()

# Define the /faq endpoint
@app.get("/faq")
def get_faq_answer(question: str):
    return {"answer": faq_agent.get_answer(question)}

@app.get("/escalation-log")
def get_escalation_log():
    # Endpoint to view all escalated issues (for admin review)
    return {"escalated_issues": escalation_agent.get_escalation_log()}

# Feedback collection endpoint
@app.post("/feedback")
async def submit_feedback(request: Request):
    data = await request.json()
    feedback = data.get("feedback")
    return {"message": feedback_agent.collect_feedback(feedback)}

# View all feedback (for admin or review purposes)
@app.get("/feedback-log")
def view_feedback():
    return {"all_feedback": feedback_agent.get_all_feedback()}