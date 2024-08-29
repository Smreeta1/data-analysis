import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter

# Load the CSV file into a DataFrame
df = pd.read_csv('3_govt_urls_state_only.csv')

# Select the relevant column containing text data
text_column = df['Note']

# Download stopwords from NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#Clean the text by removing stopwords, punctuation, and digits
def clean_text(text):
    text = text.lower()  #to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    return ' '.join(word for word in text.split() if word not in stop_words)

# Generate n-grams from a list of tokens
def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))

#Extract and sort the most common n-grams from a list of texts
def extract_top_ngrams(texts, n, min_freq=2):
  
    all_ngrams = []
    for text in texts:
        tokens = clean_text(text).split()
        all_ngrams.extend(generate_ngrams(tokens, n))
    ngram_counts = Counter(all_ngrams)
    sorted_ngrams = sorted(ngram_counts.items(), key=lambda item: item[1], reverse=True)
    return {ngram: count for ngram, count in sorted_ngrams if count >= min_freq}

#mapping of sectors based on common n-grams
def map_sectors(ngrams_dict):
    return { ' '.join(ngram): ' '.join(ngram) for ngram in ngrams_dict }

#Assign a sector to a given note
def assign_sector(note, mapping):
    tokens = clean_text(note).split()
    ngrams_list = generate_ngrams(tokens, 2) + generate_ngrams(tokens, 3)
    for ngram in ngrams_list:
        ngram_str = ' '.join(ngram)
        if ngram_str in mapping:
            return mapping[ngram_str]
    return 'Unassigned'

#common bigrams and trigrams
common_bigrams = extract_top_ngrams(text_column, n=2)
common_trigrams = extract_top_ngrams(text_column, n=3)

# Convert to DataFrame for CSV output
bigram_df = pd.DataFrame(list(common_bigrams.items()), columns=['Bigram', 'Frequency'])
trigram_df = pd.DataFrame(list(common_trigrams.items()), columns=['Trigram', 'Frequency'])

# Combine bigrams and trigrams, padding with NaN where necessary
combined_ngrams_df = pd.concat([bigram_df, trigram_df], axis=1)

# Saving top ngrams to CSV
combined_ngrams_df.to_csv('top_ngrams.csv', index=False)
print("Top bigrams and trigrams have been saved to 'top_ngrams.csv'.")

# Combine bigrams and trigrams into a single dictionary
combined_ngrams = {**common_bigrams, **common_trigrams}

# Create sector mapping
sector_mapping = map_sectors(combined_ngrams)

# Assign sector to each note
df['Sector'] = df['Note'].apply(lambda x: assign_sector(x, sector_mapping))
print("Notes have been categorized and saved to 'topics.csv'.")


# Save the updated DataFrame with sectors to a new CSV file
df.to_csv('topics.csv', index=False)
