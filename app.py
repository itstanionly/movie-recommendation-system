import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 Movie Recommendation System")

df = pd.read_csv("movies.csv")

mood_map = {
    "happy": ["happy", "comedy"],
    "sad": ["sad", "romance"],
    "excited": ["action"],
    "scared": ["horror"],
    "comedy": ["comedy"],
    "action": ["action"],
    "horror": ["horror"]
}

user_input = st.text_input("Enter mood or genre:")

if user_input:
    user_input = user_input.lower().strip()

    if user_input not in mood_map:
        st.error("❌ Mood not recognized")
    else:
        genres = mood_map[user_input]
        filtered_df = df[df["genre"].isin(genres)]

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(filtered_df["description"])

        user_vec = vectorizer.transform([user_input])
        similarity = cosine_similarity(user_vec, tfidf_matrix)

        indices = similarity.argsort()[0][-5:][::-1]

        st.subheader("🎬 Recommended Movies")
        for i in indices:
            st.write(filtered_df.iloc[i]["title"])
