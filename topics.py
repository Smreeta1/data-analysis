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

def clean_text(text):
    if text in cleaned_text_cache:
        return cleaned_text_cache[text]

    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\b\w*\d\w*\b', '', text)  # Remove words containing digits

    # Remove state names
    for state in states:
        text = re.sub(r'\b' + re.escape(state) + r"('s)?\b", '', text) #removes state names with apostrophe('s)

    # Regular expression to match URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Replace all URLs with 'agencies url'
    text = re.sub(url_pattern, 'agencies url', text)

    # Remove U.S. and similar patterns
    text = re.sub(r'\bu\.s\.[^,]*,', '', text)

    # Remove .gov patterns
    text = re.sub(r'\b[a-z]+\.gov\b', '', text)

    # Remove specific two-letter state abbreviations (like NC, NY, SC, LA)
    text = re.sub(r'\b\w{2}\b', '', text)
   
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords
    cleaned_text = ' '.join(word for word in text.split() if word not in stop_words)

    cleaned_text_cache[text] = cleaned_text
    return cleaned_text


# Generate n-grams from a list of tokens
def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))

# Extract and sort the most common n-grams from a list of texts
def extract_top_ngrams(texts, n_values, min_freq=1):
    all_ngrams = []
    for text in texts:
        tokens = clean_text(text).split()
        for n in n_values:
            all_ngrams.extend(ngrams(tokens, n))
    ngram_counts = Counter(all_ngrams)
    sorted_ngrams = sorted(ngram_counts.items(), key=lambda item: item[1], reverse=True)
    return {ngram: count for ngram, count in sorted_ngrams if count >= min_freq}

# Extract common n-grams (bigrams and trigrams)
common_ngrams = extract_top_ngrams(df['Note'], n_values=[2, 3])

# Save the most common n-grams to a CSV file
top_ngrams_df = pd.DataFrame(common_ngrams.items(), columns=['N-Gram', 'Frequency'])
top_ngrams_df.to_csv('top_ngrams.csv', index=False)
print("Most common n-grams have been saved to 'top_ngrams.csv'.")

# Mapping of sectors based on top n-grams
def map_sectors(ngrams_dict):
    return { ' '.join(ngram): ' '.join(ngram) for ngram in ngrams_dict }

# Assign a sector note
def assign_sector(note, mapping):
    cleaned_note = clean_text(note) 
    tokens = cleaned_note.split()
    ngrams_list = generate_ngrams(tokens, 2) + generate_ngrams(tokens, 3)
    
    for ngram in ngrams_list:
        ngram_str = ' '.join(ngram)
        if ngram_str in mapping:
            return mapping[ngram_str]
    return 'Unassigned'

# Create sector mapping
sector_mapping = map_sectors(common_ngrams)

# Assign sector to each note
df['Sector'] = df['Note'].apply(lambda x: assign_sector(x, sector_mapping))

# Remove unwanted columns and keep only 'Sector' and 'Note'
df_cleaned = df[['Sector', 'Note']]

# Save the cleaned DataFrame to a new CSV file
df_cleaned.to_csv('topics.csv', index=False)
print("Updated DataFrame with sectors has been saved to 'topics.csv'.")
