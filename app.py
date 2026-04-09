import streamlit as st
import pandas as pd

df = pd.read_csv("/content/movies.csv")
df.head()
     
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
mood_map = {
    "happy": ["happy", "comedy"],
    "sad": ["sad", "romance"],
    "excited": ["action"],
    "scared": ["horror"]
}
def recommend_movies(user_mood):

    if user_mood not in mood_map:
        print(" Mood not recognized")
        return

    genres = mood_map[user_mood]

    filtered_df = df[df["genre"].isin(genres)]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(filtered_df["description"])

    user_vec = vectorizer.transform([user_mood])

    similarity = cosine_similarity(user_vec, tfidf_matrix)

  
    indices = similarity.argsort()[0][-5:][::-1]

    print("\n🎬 Recommended Movies:\n")
    for i in indices:
        print("-", filtered_df.iloc[i]["title"])
      user_input = input("How are you feeling? (happy/sad/excited/scared): ").lower()
recommend_movies(user_input)
