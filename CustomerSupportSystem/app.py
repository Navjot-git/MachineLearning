from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from agents.greeter_agent import GreeterAgent
from agents.faq_agent import FAQAgent
from agents.escalation_agent import EscalationAgent
from agents.feedback_agent import FeedbackAgent

# Create the FastAPI app
app = FastAPI()

# Serve static files (CSS, JS) from the "/static" path
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the agents
escalation_agent = EscalationAgent()
greeter = GreeterAgent()
faq_agent = FAQAgent(escalation_agent) # FAQ agent passes escalation agent
feedback_agent = FeedbackAgent(faq_agent, escalation_agent) # Feedback agent checks FAQ agent's escalation status

# Path to the directory where index.html is located
static_dir = Path(__file__).parent / "static"

# Serve the main HTML file at the root ("/")
@app.get("/")
async def serve_homepage():
    return FileResponse(static_dir / "index.html")

# Define the /greet endpoint
@app.get("/greet")
def greet_user():
    return greeter.greet()

# Define the /faq endpoint (no prefix, accessible directly from the root)
@app.get("/faq")
def get_faq_answer(question: str):
    answer = faq_agent.get_answer(question)
    status = "resolved" if faq_agent.escalation_flag == False else "escalated"
    return {"answer": answer, "status": status}

@app.get("/escalation-log")
def get_escalation_log():
    # Endpoint to view all escalated issues (for admin review)
    return {"escalated_issues": escalation_agent.get_escalation_log()}

# Feedback collection endpoint
@app.post("/feedback")
async def submit_feedback(request: Request):
    data = await request.json()
    feedback = data.get("feedback")
    feedback_agent.collect_feedback(feedback)
    sentiment = feedback_agent.sentiment_agent.analyze_sentiment(feedback)
    return {"message": "Thank you for your feedback!", "sentiment": sentiment}

# View all feedback (for admin or review purposes)
@app.get("/feedback-log")
def view_feedback():
    return {"all_feedback": feedback_agent.get_all_feedback()}