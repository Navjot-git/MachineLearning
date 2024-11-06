from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
import pickle
import os
from fastapi.responses import FileResponse
from pathlib import Path


# Initialize FastAPI app
app = FastAPI()

# Serve static files (CSS, JS) from the "/static" path
app.mount("/static", StaticFiles(directory="static"), name="static")

df = pd.read_csv("backend/data/tv_shows_preprocessed.csv")

# Load the similarity matrix
with open('backend/data/similarity_matrix.pkl', 'rb') as file:
    similarity_matrix = pickle.load(file)

# Assume `overview_matrix` and `genres_df` are precomputed and loaded from preprocessing
# (This can be done in `dataprep.py`)
# combined_features = hstack([overview_matrix, genres_df.values])
# similarity_matrix = cosine_similarity(combined_features)

# Dummy similarity matrix for structure demonstration
# similarity_matrix = cosine_similarity([[1, 0], [0, 1]])  # Replace with actual similarity matrix

class Show(BaseModel):
    name: str
    overview: str
    genres: List[str]
    vote_average: float
    vote_count: int
    poster_path: str


# Path to the directory where index.html is located
static_dir = Path(__file__).parent / "static"

# Serve the main HTML file at the root ("/")
@app.get("/")
async def serve_homepage():
    return FileResponse(static_dir / "index.html")

# @app.get("/")
# def read_root():
#     return {"Welcome": "TV Show Recommendation System"}


@app.get("/shows")
def get_shows(limit: int = Query(14, description="Number of shows to return")):
    """Endpoint to retrieve a limited number of shows."""
    # Replace NaN values with empty strings or a default value
    df_cleaned = df.fillna('')  # Replace NaN with empty strings or a default value
    
    # Limit the number of shows returned
    limited_shows = df_cleaned.head(limit)

    # Convert the DataFrame to a dictionary and return
    return limited_shows.to_dict(orient='records')

@app.get("/show/{show_id}", response_model=Show)
def get_show(show_id: int):
    """Endpoint to get details of a specific show by ID."""
    if show_id not in df.index:
        raise HTTPException(status_code=404, detail="Show not found")
    show_data = df.loc[show_id].to_dict()
    return Show(**show_data)

@app.get("/recommend", response_model=List[dict])
def recommend(title: str = Query(..., description="Title of the show to get recommendations for")):
    """Endpoint to get recommendations for a specific show based on its title."""
    
    # Check if the show exists in the dataset
    if title not in df['name'].values:
        raise HTTPException(status_code=404, detail="Show not found")

    # Find the index of the show in the DataFrame
    idx = df[df['name'] == title].index[0]

    # Get similarity scores for all shows with the selected show
    sim_scores = list(enumerate(similarity_matrix[idx]))

    # Sort shows based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get indices of the top 5 most similar shows (excluding the show itself)
    top_n = 5
    sim_scores = sim_scores[1:top_n+1]  # Skip the first item as it's the show itself
    show_indices = [i[0] for i in sim_scores]
    
    # Return names of recommended shows
    # recommended_shows = df['name'].iloc[show_indices].tolist()

    recommended_shows = [
        {"name": df.iloc[i]["name"], "poster_path": df.iloc[i]["poster_path"]}
        for i in show_indices
    ]

    return recommended_shows
