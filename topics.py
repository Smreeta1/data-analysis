import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter

# Load the CSV file into a DataFrame
df = pd.read_csv('3_govt_urls_state_only.csv')

# Extract and clean state names from the 'Location' column
states = sorted(set(df['Location'].dropna().str.strip().str.lower()))

# Save cleaned state names to a CSV file
pd.DataFrame(states, columns=['State']).to_csv('states.csv', index=False)

# Load cleaned state names from CSV file
states_set = set(pd.read_csv('states.csv')['State'])

# Create a regex pattern for state names
state_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, states_set)) + r')\b', re.IGNORECASE)

# Download required NLTK resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Cache for storing cleaned text
cleaned_text_cache = {}

def clean_text(text, states_pattern):
    if not text:
        return ''
    
    # Return cached result if available
    cached_text = cleaned_text_cache.get(text)
    if cached_text is not None:
        return cached_text
    
    # Clean and process text
    text = text.split('--', 1)[0].strip().lower()
    text = re.sub(states_pattern, '', text)
    text = re.sub(r'https://\S+', ' ', text)
    text = re.sub(r'\bu\.s\.[^,]*,', '', text)
    text = re.sub(r'\b\w{2}\b', '', text)
    
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    cleaned_text = ' '.join(tokens)
    cleaned_text_cache[text] = cleaned_text
    return cleaned_text

def extract_top_ngrams(texts, n_values, min_freq=2):
    all_ngrams = [
        ngram
        for text in texts
        for n in n_values
        for ngram in ngrams(clean_text(text, state_pattern).split(), n)
        if len(clean_text(text, state_pattern)) >= min_freq
    ]
    ngram_counts = Counter(all_ngrams)
    return {ngram: count for ngram, count in ngram_counts.items() if count >= min_freq}

def save_ngrams_to_csv(ngrams_dict, filename='top_ngrams.csv'):
    pd.DataFrame(ngrams_dict.items(), columns=['N-Gram', 'Frequency']).to_csv(filename, index=False)
    print(f"Most common n-grams have been saved to '{filename}'.")

def map_sectors(ngrams_dict):
    return { ' '.join(ngram): ' '.join(ngram) for ngram in ngrams_dict }

def assign_sector(note, mapping):
    tokens = clean_text(note, state_pattern).split()
    for n in [2, 3]:
        for ngram in ngrams(tokens, n):
            if ' '.join(ngram) in mapping:
                return mapping[' '.join(ngram)]
    return ''

# Extract common n-grams (bigrams and trigrams)
common_ngrams = extract_top_ngrams(df['Note'].dropna(), n_values=[2, 3])

# Save the most common n-grams to a CSV file
save_ngrams_to_csv(common_ngrams)

# Create sector mapping
sector_mapping = map_sectors(common_ngrams)

# Assign sector to each note
df['Sector'] = df['Note'].apply(lambda x: assign_sector(x, sector_mapping))

# Save the cleaned DataFrame to a new CSV file
df[['Sector', 'Note']].to_csv('topics.csv', index=False)
print("Updated DataFrame with sectors has been saved to 'topics.csv'.")
