# Movie Recommendation System

## Overview
This project is a hybrid movie recommendation system that suggests movies based on user mood or genre.

## Approach
- User inputs a mood (like happy, sad, excited, scared)
- The system maps the mood to relevant genres
- Movies are filtered based on genre
- TF-IDF is applied on movie descriptions
- Cosine similarity is used to recommend top 3–5 movies

## Features
- Mood-based recommendation
- Genre filtering
- Uses TF-IDF (NLP technique)
- Returns top 3–5 relevant movies
- Simple and clean implementation

## Tech Stack
- Python
- Pandas
- Scikit-learn

## How to Run
1. Open the notebook in Google Colab
2. Upload `movies.csv`
3. Run all cells
4. Enter your mood (happy/sad/excited/scared)

## Output
- Takes user input (mood)
- Returns 3–5 recommended movies
