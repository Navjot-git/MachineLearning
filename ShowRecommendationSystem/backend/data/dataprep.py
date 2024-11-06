import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# Load and initial cleaning
df = pd.read_csv("backend/data/tv_shows_data.csv")
df = df.drop(['adult', 'backdrop_path', 'created_by', 'homepage', 'in_production', 'last_air_date', 
              'last_episode_to_air', 'next_episode_to_air', 'original_name', 'production_countries', 
              'seasons', 'status', 'tagline', 'type'], axis=1)
df.dropna(subset=['genres', 'overview', 'vote_average', 'vote_count', 'poster_path'], inplace=True)

# Process 'genres' column: convert JSON-like strings to lists of genre names
df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
print(df['genres'])
df['genres'] = df['genres'].apply(lambda x: [genre['name'] for genre in x] if isinstance(x, list) else [])
print(df['genres'])
# Encode genres using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(df['genres'])
genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)

# Concatenate genres encoding with the main DataFrame
df = pd.concat([df, genres_df], axis=1)
print(df['genres'])
df.dropna(subset=['overview'], inplace=True)

# TF-IDF Transformation on 'overview' text
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
overview_matrix = tfidf.fit_transform(df['overview'])

# Combine TF-IDF matrix with genres encoding
combined_features = hstack([overview_matrix, genres_df.values])

# Compute Cosine Similarity
similarity_matrix = cosine_similarity(combined_features)

# Export the necessary matrices for 'model.py'
import pickle

# Save similarity matrix to disk
with open('backend/data/similarity_matrix.pkl', 'wb') as file:
    pickle.dump(similarity_matrix, file)

# Save the updated DataFrame to a new CSV file
df.to_csv("backend/data/tv_shows_preprocessed.csv", index=False)
