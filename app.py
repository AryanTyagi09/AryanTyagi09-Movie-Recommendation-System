import pickle
import pandas as pd
import streamlit as st
import gdown

# Google Drive file links
movie_dict_url = "https://drive.google.com/uc?id=1nFGT8JdaTCf0ZFr_-GZ7gEmtU0NqM8SP"
similarity_url = "https://drive.google.com/uc?id=1Z0Hb5HxvGavIyNHZ2tHSqojAPGueN7qu"

# Download files
gdown.download(movie_dict_url, 'movie_dict.pkl', quiet=False)
gdown.download(similarity_url, 'similarity.pkl', quiet=False)

# Load data
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))


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
