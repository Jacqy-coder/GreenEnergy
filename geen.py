import pandas as pd

# Step 1: Read the file line by line (safely)
news_data = []
with open("new.tsv", 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 4:  # Check if there's enough data
            news_id = parts[0]
            category = parts[1]
            title = parts[3]
            news_data.append([news_id, category, title])

# Step 2: Load into a DataFrame
news_df = pd.DataFrame(news_data, columns=['news_id', 'category', 'title'])
print(news_df.head())


#clean up the data
print(news_df.isnull().sum())

#remove duplicates
news_df.drop_duplicates(inplace=True)

#clean text data
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

news_df['title'] = news_df['title'].apply(clean_text)

#remove white spece
news_df['title'] = news_df['title'].str.strip()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Step 3: TF-IDF vectorization of the titles
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(news_df['title'])

# Step 4: Define recommendation function (compute similarity only when needed)
from sklearn.metrics.pairwise import cosine_similarity

def recommend_news(news_title, top_n=5):
    try:
        idx = news_df[news_df['title'] == news_title].index[0]
    except IndexError:
        return ["News title not found."]
    
    # Compute similarity between the selected article and all others
    vec = tfidf_matrix[idx]
    sim_scores = cosine_similarity(vec, tfidf_matrix).flatten()

    # Get indices of top_n similar articles, excluding the article itself
    similar_indices = sim_scores.argsort()[::-1][1:top_n + 1]
    
    return news_df['title'].iloc[similar_indices].tolist()


sample_title = news_df['title'].iloc[0]
print(f"Article: my health\n")
print("Recommended articles:")
for rec in recommend_news(sample_title):
    print("-", rec)