import pickle
import numpy as np 
import pandas as pd
import streamlit as st

import gdown
import os

# Google Drive file IDs
movie_dict_file_id = "1nFGT8JdaTCf0ZFr_-GZ7gEmtU0NqM8SP"
similarity_file_id = "1Z0Hb5HxvGavIyNHZ2tHSqojAPGueN7qu"

# File paths
movie_dict_file_path = "movie_dict.pkl"
similarity_file_path = "similarity.pkl"

# Function to download a file if not present
def download_file(file_id, file_path):
    if not os.path.exists(file_path):
        print(f"Downloading {file_path}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_path, quiet=False)
    else:
        print(f"{file_path} already exists.")

# Download files
download_file(movie_dict_file_id, movie_dict_file_path)
download_file(similarity_file_id, similarity_file_path)

# Load pickle files
try:
    with open(movie_dict_file_path, 'rb') as f:
        movies_dict = pickle.load(f)
    movies = pd.DataFrame(movies_dict)

    with open(similarity_file_path, 'rb') as f:
        similarity = pickle.load(f)
    
    print("Files loaded successfully!")
except Exception as e:
    print(f"Error loading files: {e}")

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    for i in movies_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

# Set the title and a brief description
st.title('🎬 Movie Recommender System')
st.write('Discover your next favorite movie based on what you love!')

# Movie selection dropdown
selected_movie_name = st.selectbox(
    'Select a movie to get recommendations:',
    movies['title'].values
)

# Recommend button
if st.button('Recommend'):
    with st.spinner('Generating recommendations...'):
        recommendations = recommend(selected_movie_name)
        st.success('Here are your recommended movies:')
        
        # Display recommended movies
        for movie in recommendations:
            st.markdown(f"- **{movie}**")  # Use markdown for better formatting

# Optional: Add some more features
st.sidebar.header('Explore Movies')
# Display top 10 movies for exploration
st.sidebar.subheader('Top Movies')
top_movies = movies['title'].head(10).values
for movie in top_movies:
    st.sidebar.markdown(f"- {movie}")

# Optionally, add a footer
st.markdown("<footer style='text-align: center;'>Created by : Aryan </footer>", unsafe_allow_html=True)
st.markdown("<footer style='text-align: center;'>  Roll No. : 2301921630014 </footer>", unsafe_allow_html=True)
