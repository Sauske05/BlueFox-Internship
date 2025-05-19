# Movie Recommendation System

## Overview

The **Movie Recommendation System** notebook implements a simple yet effective recommendation engine based on collaborative filtering. It processes a dataset of user ratings and leverages item-based similarity (cosine similarity) to suggest similar movies.

This project demonstrates:
- Basic **data loading and preprocessing** with Pandas,
- Construction of a **user-item matrix**,
- **Cosine similarity computation** between movie vectors,
- Interactive function to retrieve **top N similar movie recommendations**.

---

## Features

- 📥 **CSV Data Load** – Reads user-item ratings from `rating.csv`
- 🧮 **User-Item Matrix Creation** – Constructs pivot table for recommendation basis
- 📊 **Cosine Similarity Calculation** – Measures similarity between movies
- 🧠 **Top-N Movie Recommendations** – Recommends similar movies based on a target title
- 📈 **Simple yet Scalable Logic** – Can be adapted to larger collaborative filtering systems

---

## Prerequisites

- Python 3.7+
- `rating.csv` dataset (userId, movieId, rating, title)
- Required Python libraries:
  - `pandas`
  - `sklearn`

---
