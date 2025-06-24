# News and Articles Recommendation System

This project implements a simple content-based news article recommender using TF-IDF and cosine similarity. It suggests similar news articles based on a given article title.

## Features

- Reads and cleans a tab-separated news dataset (`new.tsv`)
- Applies TF-IDF vectorization on article titles
- Computes similarity using cosine distance
- Recommends top N similar articles
- Lightweight and works without user data

##  Dataset Format

The dataset should be a `.tsv` (tab-separated) file with at least 4 columns: Only `news_id`, `category`, and `title` are used.

##  Business Use Case
This tool helps digital news platforms increase user engagement by suggesting related articles, driving higher session duration, lower bounce rates, and better ad performance.

##  TODO / Improvements
Expand to full article content (not just titles)

Add user-based collaborative filtering

Build a web interface or API for deployment



