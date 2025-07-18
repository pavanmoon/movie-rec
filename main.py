import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import requests

# Define constants
API_KEY = "f1637e77b7ca1ee7c4ce0db6268be28e"
POSTER_URL_BASE = "https://image.tmdb.org/t/p/w500/"


def read_data():
  """Reads movie data from CSV files."""
  #df1 = pd.read_csv("tmdb_5000_credits.csv")
  df2 = pd.read_csv("tmdb_5000_movies.csv")
  #df1.columns = ['id', 'tittle', 'cast', 'crew']
  #df2 = df2.merge(df1, on='id')
  return df2


def calculate_weighted_ratings(df2):
  """Calculates weighted ratings for movies."""
  C = df2['vote_average'].mean()
  m = df2['vote_count'].quantile(0.9)
  q_movies = df2.copy().loc[df2['vote_count'] >= m]
  q_movies['score'] = q_movies.apply(weighted_rating, axis=1, m=m, C=C)
  q_movies = q_movies.sort_values('score', ascending=False)
  return q_movies


def weighted_rating(x, m, C):
  """Calculates weighted rating for a single movie."""
  v = x['vote_count']
  R = x['vote_average']
  return (v / (v + m) * R) + (m / (m + v) * C)


def fetch_poster(movie_id):
  url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
  data = requests.get(url).json()
  poster_path = data.get('poster_path')
  if poster_path:
    return f"{POSTER_URL_BASE}{poster_path}"
  return None


def get_top_movies_by_score(q_movies):
  """Prints and visualizes top movies by weighted score."""
  st.header("Top 10 Movies by Weighted Score 🚀")
  wr = q_movies[['title', 'score', 'id']].head(10)
  st.dataframe(wr[['title', 'score']])

  fig, ax = plt.subplots()
  ax.barh(wr['title'], wr['score'], align='center', color='#00ffcc')
  ax.invert_yaxis()
  ax.set_xlabel("Weighted Score")
  ax.set_title("Top 10 Movies based on Weighted Score")
  st.pyplot(fig)

  st.header("Top 10 Movie Posters by Weighted Score 🎥")
  for _, row in wr.iterrows():
    poster_url = fetch_poster(row['id'])
    if poster_url:
      st.image(poster_url, caption=row['title'])
    else:
      st.write(f"{row['title']} - Poster not available")


def get_top_movies_by_popularity(df2):
  """Prints and visualizes top movies by popularity."""
  st.header("Top 10 Popular Movies by TMDb 🌟")
  pop = df2[['title', 'popularity',
             'id']].sort_values('popularity', ascending=False).head(10)
  st.dataframe(pop[['title', 'popularity']])

  fig, ax = plt.subplots()
  ax.barh(pop['title'], pop['popularity'], align='center', color='#00ffcc')
  ax.invert_yaxis()
  ax.set_xlabel("Popularity")
  ax.set_title("Popular Movies")
  st.pyplot(fig)

  st.header("Top 10 Movie Posters by Popularity 🍿")
  for _, row in pop.iterrows():
    poster_url = fetch_poster(row['id'])
    if poster_url:
      st.image(poster_url, caption=row['title'])
    else:
      st.write(f"{row['title']} - Poster not available")


def create_tfidf_matrix(df2):
  """Creates TF-IDF matrix and cosine similarity matrix."""
  tfidf = TfidfVectorizer(stop_words='english')
  df2['overview'] = df2['overview'].fillna('')
  tfidf_matrix = tfidf.fit_transform(df2['overview'])
  cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
  indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()
  return tfidf_matrix, cosine_sim, indices


def get_personalized_recommendations(title, cosine_sim, indices, df2):
  """Recommends movies similar to a given title."""
  if title not in indices:
    return None
  idx = indices[title]
  sim_scores = list(enumerate(cosine_sim[idx]))
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
  sim_scores = sim_scores[1:6]  # Get top 5 similar movies
  movie_indices = [i[0] for i in sim_scores]
  return df2['title'].iloc[movie_indices]


# Main code to run Streamlit app
st.title("Movie Recommender System 🎥")
df2 = read_data()
q_movies = calculate_weighted_ratings(df2.copy())
tfidf_matrix, cosine_sim, indices = create_tfidf_matrix(df2)

menu = st.sidebar.radio("Choose a Recommender Type", [
    "Top 10 Movies By Weighted Score", "Top 10 Popular Movies",
    "Get Personalized Recommendations"
])

if menu == "Top 10 Movies By Weighted Score":
  get_top_movies_by_score(q_movies.copy())

elif menu == "Top 10 Popular Movies":
  get_top_movies_by_popularity(df2.copy())

elif menu == "Get Personalized Recommendations":
  st.header("Get Personalized Movie Recommendations 📽️")

  # List of selected movies
  selected_movies = [
      "Harry Potter and the Chamber of Secrets",
      "Harry Potter and the Philosopher's Stone",
      "The Hobbit: The Desolation of Smaug", "Avatar", "Spider-Man 3",
      "Avengers: Age of Ultron", "Iron Man", "Iron Man 2",
      "X-Men: The Last Stand", "Star Trek Beyond", "The Fast and the Furious",
      "How to Train Your Dragon", "Mission: Impossible - Rogue Nation",
      "Minions"
  ]

  selected_movie = st.selectbox("Select a movie 🍿", selected_movies)

  if st.button("Get Recommendations 🎬"):
    recommendations = get_personalized_recommendations(selected_movie,
                                                       cosine_sim, indices,
                                                       df2)
    if recommendations is not None:
      for title in recommendations:
        movie_id = df2[df2['title'] == title]['id'].values[0]
        poster_url = fetch_poster(movie_id)
        if poster_url:
          st.image(poster_url, caption=title)
        else:
          st.write(f"Poster not available for {title}")
    else:
      st.write(f"No recommendations found for '{selected_movie}'")
