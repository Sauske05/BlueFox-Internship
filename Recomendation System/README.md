Netflix Recommendation System
Overview
This project implements a content-based recommendation system for Netflix titles using a dataset of movies and TV shows. The system recommends similar titles based on their genres and types by leveraging feature concatenation and cosine similarity.
Dataset
The dataset (netflix_titles.csv) contains information about Netflix titles, including:

show_id: Unique identifier for each title
type: Movie or TV Show
title: Title of the movie or show
director: Director of the title
cast: Actors involved
country: Country of origin
date_added: Date added to Netflix
release_year: Year of release
rating: Content rating
duration: Duration (minutes or seasons)
listed_in: Genres
description: Brief description

Methodology

Data Loading and Exploration:

The dataset is loaded using Pandas and explored to identify missing values.
Key columns (type and listed_in) are selected for the recommendation system.


Feature Concatenation:

A new column combined_features is created by concatenating listed_in (genres) and type to form a single feature for each title.


Text Vectorization:

The CountVectorizer from scikit-learn is used to convert the combined_features into a matrix of token counts, with genres and types split by commas.


Cosine Similarity:

A cosine similarity matrix is computed to measure the similarity between titles based on their vectorized features.
The matrix is used to identify titles with similar genre and type profiles.


Recommendation Function:

A function recommend_movies takes a movie title as input and returns the top 5 most similar titles based on cosine similarity scores.
The function handles cases where the input title is not found in the dataset.


Visualization:

A heatmap is generated using Seaborn to visualize the cosine similarity matrix for a subset of titles.



Requirements
To run the notebook, install the following Python libraries:
pip install pandas scikit-learn seaborn matplotlib

Usage

Ensure the netflix_titles.csv dataset is in the same directory as the notebook.
Open the Recommendation_system.ipynb notebook in Jupyter Notebook or JupyterLab.
Run the cells sequentially to:
Load and preprocess the data
Create the recommendation system
Test the recommendation function by entering a movie title
Visualize the similarity matrix


Example input for the recommendation function:Enter the movie name: Ganglands

Output:Movies recommended for Ganglands:
11      Bangkok Breaking
543           Undercover
734                Lupin
1223              Dealer
2676               Fauda
Name: title, dtype: object



Files

Recommendation_system.ipynb: Jupyter Notebook containing the implementation.
netflix_titles.csv: Dataset of Netflix titles (not included in this repository; must be sourced separately).
README.md: This file, providing an overview and instructions.

Limitations

The recommendation system relies solely on type and listed_in columns, ignoring other potentially useful features like description or cast.
Missing values in the dataset (e.g., director, cast, country) are not handled, as they are not used in the current implementation.
The system may not capture nuanced similarities due to the simplicity of the feature concatenation approach.

Future Improvements

Incorporate additional features like description using TF-IDF vectorization or NLP techniques.
Handle missing data to improve robustness.
Experiment with other similarity metrics or clustering techniques.
Add user-based filtering to personalize recommendations.

License
This project is for educational purposes and uses the Netflix titles dataset. Ensure you have the right to use the dataset before running the code.
