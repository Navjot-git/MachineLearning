import spacy
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from spacy import displacy

# Load the SpaCy mode
nlp = spacy.load("en_core_web_sm")

# Initialize FastAPI app
app = FastAPI()

# Mount the static folder (for serving CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Define the request body format
class TextRequest(BaseModel):
    text: str
    
# Define the response format
class EntityResponse(BaseModel):
    text: str
    label: str
    start: int
    end: int
    
# Root route to return a simple welcome message
@app.get("/")
def get_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "entities": None})

# Route to handle form submissions from the web interface
@app.post("/extract-text/", response_class=HTMLResponse)
async def post_root(request: Request, text: str = Form(...)):
    doc = nlp(text)

    # Check if entities exist in the document
    if not doc.ents:
        return templates.TemplateResponse("index.html", {"request": request, "entities_html": "No entities found."})

    # Generate HTML using displaCy
    entities_html = displacy.render(doc, style="ent", page=True)
    
    # Pass the displaCy HTML to the template
    return templates.TemplateResponse("index.html", {"request": request, "entities_html": entities_html})

# New Route to handle standard entities and non-entities
@app.post("/extract-entities-and-nonentities/", response_class=HTMLResponse)
async def show_all_tokens(request: Request, text: str = Form(...)):
    doc = nlp(text)
    
    # Start with displaCy-rendered entities HTML
    entities_html = displacy.render(doc, style="ent", page=True)
    
    # Create an HTML string to hold both entities and non-entities in the same sentence
    tokens_html = "<p>"
    
    for token in doc:
        if token.ent_type_:  # If token is an entity, style it like displaCy
            tokens_html += f'<mark style="background-color:#f39c12; padding: 0.2em 0.4em; border-radius: 0.35em;">{token.text} ({token.ent_type_})</mark> '
        else:  # For non-entities, just display them as plain text
            tokens_html += f'{token.text} '
    
    tokens_html += "</p>"
    
    # Return the sentence with both entities and non-entities
    return templates.TemplateResponse("index.html", {"request": request, "entities_html": tokens_html})


# API Endpoint to extract entities from text
@app.post("/extract-entities/", response_model=list[EntityResponse])
def extract_entities(request: TextRequest):
    doc = nlp(request.text)
    
    # Extract entities
    entities = [
        {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
    ]
    
    # Return the entities as a list 
    return entities
    
    
"""
  "Microsoft is planning to open a new office in New York by the end of 2025."
  "Elon Musk sold $5 billion worth of Tesla shares in November 2021."
  "The United Nations will hold a summit in Geneva on March 15, 2023, to discuss climate change."
  "Apple acquired Beats Electronics for $3 billion in 2014."
  "Lionel Messi signed a two-year contract with Paris Saint-Germain in August 2021."
  "The 2020 Tokyo Olympics were postponed due to the COVID-19 pandemic, affecting athletes from all over the world."
  "Amazon launched its new Echo device in Germany, where the product supports multiple languages including English, German, and French."
  "Beyoncé collaborated with Adidas to launch her Ivy Park clothing line."
  "Google is working on expanding its data centers in Singapore and Hong Kong."
  "In 1969, Neil Armstrong became the first person to walk on the moon as part of NASA's Apollo 11 mission."

"""