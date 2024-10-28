import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()


# TMDb API Key
API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

def get_popular_tv_shows(page=1):
    url = f"{BASE_URL}/tv/popular?api_key={API_KEY}&language=en-US&page={page}"
    response = requests.get(url)
    return response.json().get('results', [])

def get_tv_show_details(tv_id):
    url = f"{BASE_URL}/tv/{tv_id}?api_key={API_KEY}&language=en-US"
    response = requests.get(url)
    return response.json()

# Fetch and store data in batches
all_shows = []
for page in range(1, 51):
    shows = get_popular_tv_shows(page)
    print(f"Fetching page: {page}")
    for show in shows:
        tv_id = show['id']
        details = get_tv_show_details(tv_id)
        all_shows.append(details)
        time.sleep(0.5) # Rate limit handling

# Convert to DataFrame and save
tv_shows_df = pd.DataFrame(all_shows)
tv_shows_df.to_csv("tv_shows_data.csv", index=False)
print("Data saved to tv_shows_data.csv")