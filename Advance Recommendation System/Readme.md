# Anime Recommendation using SVD

## Overview

The **Anime Recommendation System** notebook implements a collaborative filtering model using matrix factorization via the SVD (Singular Value Decomposition) algorithm. It uses the `Surprise` library to handle rating data and generate personalized anime recommendations.

The notebook covers:
- Data preprocessing and rating normalization
- Model training using SVD
- Performance evaluation with RMSE and MAE
- Prediction for specific user-anime pairs

---

## Features

- 📄 **Anime Ratings Dataset** – Loads and preprocesses user-anime ratings
- ⚙️ **Rating Normalization** – Adjusts rating scale to a standard 0–5 range
- 🧠 **Model Training** – Implements collaborative filtering using SVD
- 📉 **Performance Evaluation** – Measures accuracy with RMSE and MAE
- 🔍 **Custom Prediction** – Predicts rating for a specific user-anime pair

---

## Prerequisites

- Python 3.7+
- `rating.csv` dataset (with `user_id`, `anime_id`, `rating`)
- Required Python libraries:
  - `pandas`
  - `scikit-surprise`

---

