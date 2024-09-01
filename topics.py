import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter

# Load the CSV file into a DataFrame
df = pd.read_csv('3_govt_urls_state_only.csv')

# Extract state names from the 'Location' column
states = df['Location'].dropna()

# Clean state names by normalizing spaces and case
states = [state.strip().lower() for state in states]

# Download stopwords from NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Cache to store cleaned text
cleaned_text_cache = {}

# Clean the text by removing stopwords, punctuation, digits, state names, and phrase "Note added" with timestamp
def clean_text(text):
    if text in cleaned_text_cache:
        return cleaned_text_cache[text]

    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\b\w*\d\w*\b', '', text)  # Removes words containing digits like 4th,etc.

    # Remove state names
    for state in states:
        text = re.sub(r'\b' + re.escape(state) + r'\b', '', text)

    # Remove the phrase "Note added" and timestamp following it
    text = re.sub(r'-- note added.*', '', text)
    
    # Remove extra spaces
    text = text.strip()
    
    cleaned_text = ' '.join(word for word in text.split() if word not in stop_words)
    
    cleaned_text_cache[text] = cleaned_text
    return cleaned_text

# Generate n-grams from a list of tokens
def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))

# Extract and sort the most common n-grams from a list of texts
def extract_top_ngrams(texts, n, min_freq=1):
    all_ngrams = []
    for text in texts:
        tokens = clean_text(text).split()
        all_ngrams.extend(generate_ngrams(tokens, n))
    ngram_counts = Counter(all_ngrams)
    sorted_ngrams = sorted(ngram_counts.items(), key=lambda item: item[1], reverse=True)
    return {ngram: count for ngram, count in sorted_ngrams if count >= min_freq}

# Mapping of sectors based on common n-grams
def map_sectors(ngrams_dict):
    return { ' '.join(ngram): ' '.join(ngram) for ngram in ngrams_dict }

# Assign a sector to a given note
def assign_sector(note, mapping):
    cleaned_note = clean_text(note)  # Clean text once
    tokens = cleaned_note.split()
    ngrams_list = generate_ngrams(tokens, 2) + generate_ngrams(tokens, 3)
    
    for ngram in ngrams_list:
        ngram_str = ' '.join(ngram)
        if ngram_str in mapping:
            return mapping[ngram_str]
    return 'Unassigned'

# Extract common bigrams and trigrams
common_bigrams = extract_top_ngrams(df['Note'], n=2)
common_trigrams = extract_top_ngrams(df['Note'], n=3)

# Combine bigrams and trigrams into a single dictionary
combined_ngrams = {**common_bigrams, **common_trigrams}

# Save the most common n-grams to a CSV file
top_ngrams_df = pd.DataFrame(combined_ngrams.items(), columns=['N-Gram', 'Frequency'])
top_ngrams_df.to_csv('top_ngrams.csv', index=False)
print("Most common n-grams have been saved to 'top_ngrams.csv'.")

# Create sector mapping
sector_mapping = map_sectors(combined_ngrams)

# Assign sector to each note
df['Sector'] = df['Note'].apply(lambda x: assign_sector(x, sector_mapping))

# Remove unwanted columns and keep only 'Sector' and 'Note'
df_cleaned = df[['Sector', 'Note']]

# Save the cleaned DataFrame to a new CSV file
df_cleaned.to_csv('topics.csv', index=False)
print("Updated DataFrame with sectors has been saved to 'topics.csv'.")
