import pandas as pd  # For handling tabular data (like Excel) in Python
import re  # For cleaning and processing text (removing symbols, etc.)
import matplotlib.pyplot as plt  # For creating charts and graphs
import seaborn as sns  # Makes matplotlib graphs prettier and more powerful

# Text processing and machine learning tools from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer  # Converts text into numbers based on word importance
from sklearn.metrics.pairwise import cosine_similarity  # Measures how similar two pieces of text are
from sklearn.preprocessing import LabelEncoder  # Converts category names (like "Sports", "Tech") into numbers

from sklearn.model_selection import train_test_split  # Splits data into training and testing parts

# Machine learning models
from sklearn.linear_model import LogisticRegression  # A basic but powerful classifier (used for prediction)
from sklearn.linear_model import SGDClassifier  # Uses a fast method for training on large text data
from sklearn.ensemble import RandomForestClassifier  # Uses multiple decision trees to improve accuracy

# Evaluation tools to check how well the models work
from sklearn.metrics import accuracy_score  # Measures how many predictions were correct
from sklearn.metrics import classification_report  # Gives detailed accuracy report (precision, recall, F1-score)
from sklearn.metrics import ConfusionMatrixDisplay  # Shows a visual summary of prediction errors

import numpy as np  # Handles arrays, lists, and math functions (used for performance)

# ===========================================
# 🧠 1. Problem Statement
# ===========================================

# Business goal: Suggest articles to users based on what they are currently reading.
# We're trying to group or predict article "categories" (like business, tech, sports) using the article's title.
# Later, this can help recommend similar articles in the same category.
# Success is measured by how accurately the model predicts the article category.

# ===========================================
# 📥 2. Data Collection
# ===========================================

# Read the raw data file (TSV format = tab-separated values)
news_data = []

# Go through each line in the file
with open("new.tsv", 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')  # Split line by tab
        if len(parts) >= 4:  # Only keep lines with at least 4 parts
            news_id = parts[0]  # Unique ID for the article
            category = parts[1]  # Category label (e.g. "Sports")
            title = parts[3]  # The title of the article
            news_data.append([news_id, category, title])  # Store the clean row

# Create a table (DataFrame) to work with
news_df = pd.DataFrame(news_data, columns=['news_id', 'category', 'title'])

# ===========================================
# 🧹 3. Data Cleaning
# ===========================================

# Check for missing or empty data
print(news_df.isnull().sum())

# Remove duplicate articles
news_df.drop_duplicates(inplace=True)

# Clean up text in the 'title' column
def clean_text(text):
    text = text.lower()  # Make text lowercase (HELLO → hello)
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters, keep only letters/numbers
    return text

# Apply cleaning function to the article titles
news_df['title'] = news_df['title'].apply(clean_text)
news_df['title'] = news_df['title'].str.strip()  # Remove extra spaces

# ===========================================
# 🔍 4. Exploratory Data Analysis (EDA)
# ===========================================

# Show first few rows to understand data
print(news_df.head())
print(news_df.info())
print(news_df.describe())

# What are the different article categories?
print(news_df['category'].unique())
print("Number of unique categories:", news_df['category'].nunique())

# Count how many articles are in each category
category_counts = news_df['category'].value_counts()
print("\nCategory distribution:")
print(category_counts)

# Make a bar chart to show the article category distribution
plt.figure(figsize=(12, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')
plt.title('Number of Articles per Category')
plt.ylabel('Count')
plt.xlabel('Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ===========================================
# 5. Feature Engineering
# ===========================================

# Convert text labels (like "business") to numbers
label_encoder = LabelEncoder()
news_df['label'] = label_encoder.fit_transform(news_df['category'])

# Separate the inputs (X) and outputs (y)
X = news_df['title']  # The text of the article
y = news_df['label']  # The numeric category label

# Split data into 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Convert text into TF-IDF vectors (numbers that represent words)
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)  # Only keep top 5000 words
X_train_tfidf = tfidf.fit_transform(X_train)  # Fit and transform training data
X_test_tfidf = tfidf.transform(X_test)  # Transform test data using same vectorizer

# ===========================================
# 🤖 6. Model Training & Evaluation
# ===========================================

# Create a dictionary of models to try
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SGD Classifier": SGDClassifier(random_state=42)
}

# Loop through each model
for name, model in models.items():
    print(f"\n🔍 Training: {name}")
    model.fit(X_train_tfidf, y_train)  # Train the model
    y_pred = model.predict(X_test_tfidf)  # Make predictions
    acc = accuracy_score(y_test, y_pred)  # Measure accuracy
    print(f"Accuracy: {acc:.4f}")  # Print result
    print("Classification Report:")
    print(classification_report(y_test, y_pred))  # Show detailed performance

# ===========================================
# 📉 7. Confusion Matrix (for the last model)
# ===========================================

# This matrix shows what the model got right or wrong for each category
present_labels = np.unique(y_test)  # Actual labels used
present_label_names = label_encoder.inverse_transform(present_labels)  # Decode label names

ConfusionMatrixDisplay.from_estimator(
    model, 
    X_test_tfidf, 
    y_test, 
    labels=present_labels, 
    display_labels=present_label_names, 
    xticks_rotation=90,
    cmap='viridis'
)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ===========================================
# 🚀 8. Deployment Prep - Best Model
# ===========================================

# Re-train the best performing model (e.g., Logistic Regression)
best_model = LogisticRegression(max_iter=1000)
best_model.fit(X_train_tfidf, y_train)