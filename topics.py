import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter

# Load the CSV file into a DataFrame
df = pd.read_csv('3_govt_urls_state_only.csv')

# Extract state names from the 'Location' column
states = df['Location'].dropna().str.strip().str.lower()

# Clean state names by normalizing spaces and case
states = sorted(set(states))

# Save cleaned state names to a CSV file
states_df = pd.DataFrame(states, columns=['State'])
states_df.to_csv('states.csv', index=False)

# Load cleaned state names from CSV file
states_df = pd.read_csv('states.csv')
states_set = set(states_df['State'])

# Create a regex pattern for state names
state_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, states_set)) + r')\b', re.IGNORECASE)

# Download stopwords from NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Cache to store cleaned text
cleaned_text_cache = {}

def clean_text(text, states_pattern):
    if text in cleaned_text_cache:
        return cleaned_text_cache[text]
    
    # Remove text after '--'
    text = text.split('--', 1)[0].strip()

    # Convert text to lowercase
    text = text.lower()

    # Remove state names
    text = re.sub(states_pattern, '', text)

    # Remove words containing digits
    text = re.sub(r'\b\w*\d\w*\b', '', text)

    # Remove URLs
    url_pattern = r'https://\S+'
    text = re.sub(url_pattern, ' ', text)

    # Remove U.S. and similar patterns
    text = re.sub(r'\bu\.s\.[^,]*,', '', text)

    # Remove .gov patterns
    text = re.sub(r'\b[a-z]+\.gov\b', '', text)

    # Remove specific two-letter state abbreviations (like NC, NY, SC, LA)
    text = re.sub(r'\b\w{2}\b', '', text)
   
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize text
    tokens = re.findall(r'\b\w+\b', text)

    # Filter out stopwords
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # Join tokens back into a single string
    cleaned_text = ' '.join(filtered_tokens)
    
    cleaned_text_cache[text] = cleaned_text
    return cleaned_text


# Apply the function to clean the 'Note' column
df['Note'] = df['Note'].apply(lambda x: clean_text(x, state_pattern))


def extract_top_ngrams(texts, n_values, min_freq=2):
    all_ngrams = []
    for text in texts:
        tokens = clean_text(text,state_pattern).split()
        for n in n_values:
            all_ngrams.extend(ngrams(tokens, n))
    ngram_counts = Counter(all_ngrams)
    sorted_ngrams = sorted(ngram_counts.items(), key=lambda item: item[1], reverse=True)
    return {ngram: count for ngram, count in sorted_ngrams if count >= min_freq}

# Save the most common n-grams to a CSV file
def save_ngrams_to_csv(ngrams_dict, filename='top_ngrams.csv'):
    top_ngrams_df = pd.DataFrame(ngrams_dict.items(), columns=['N-Gram', 'Frequency'])
    top_ngrams_df.to_csv(filename, index=False)
    print(f"Most common n-grams have been saved to '{filename}'.")

# Mapping of sectors based on top n-grams
def map_sectors(ngrams_dict):
    return { ' '.join(ngram): ' '.join(ngram) for ngram in ngrams_dict }

# Assign a sector note
def assign_sector(note, mapping):
    tokens = clean_text(note,state_pattern).split()
    for n in [2, 3]:
        ngrams_list = ngrams(tokens, n)
        for ngram in ngrams_list:
            ngram_str = ' '.join(ngram)
            if ngram_str in mapping:
                return mapping[ngram_str]
    return 'Unassigned'

# Extract common n-grams (bigrams and trigrams)
common_ngrams = extract_top_ngrams(df['Note'], n_values=[2, 3])

# Save the most common n-grams to a CSV file
save_ngrams_to_csv(common_ngrams)

# Create sector mapping
sector_mapping = map_sectors(common_ngrams)

# Assign sector to each note
df['Sector'] = df['Note'].apply(lambda x: assign_sector(x, sector_mapping))

# Remove unwanted columns and keep only 'Sector' and 'Note'
df_cleaned = df[['Sector', 'Note']]

# Save the cleaned DataFrame to a new CSV file
df_cleaned.to_csv('topics.csv', index=False)
print("Updated DataFrame with sectors has been saved to 'topics.csv'.")
