from fastapi import FastAPI
from agents.greeter_agent import GreeterAgent

# Create the FastAPI app
app = FastAPI()

# Initialize the Greeter Agent
greeter = GreeterAgent()

# Define the /greet endpoint
@app.get("/greet")
def greet_user():
    return greeter.greet()