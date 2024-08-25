import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.util import ngrams
import re

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Eg. text
text = "How are you feeling? Are you happy today?"

# Tokenization: Sentence and Word
sentences = sent_tokenize(text)
print("Sentences:", sentences)

# Tokenize each sentence into words
for sentence in sentences:
    words = word_tokenize(sentence)
    print("Words:", words)

# Convert the entire text to lowercase
text_lower = text.lower()
print("Lowercase Text:", text_lower)

# Tokenize the lowercase text
words_lower = word_tokenize(text_lower)
print("Lowercase Words:", words_lower)

# Remove punctuation using regex
words_lower_clean = [word for word in words_lower if re.match(r'\b\w+\b', word)]
print("Words Without Punctuation:", words_lower_clean)

# Remove Stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words_lower_clean if word not in stop_words]
print("Filtered Words:", filtered_words)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in words_lower_clean]
print("Lemmatized Words:", lemmatized_words)

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in words_lower_clean]
print("Stemmed Words:", stemmed_words)

# N-grams
bigrams = list(ngrams(words_lower_clean, 2))
trigrams = list(ngrams(words_lower_clean, 3))
print("Bigrams:", bigrams)
print("Trigrams:", trigrams)

# Function to generate n-grams from raw text
def generate_ngrams(text, n):
    # Remove punctuation and split the text by spaces to create a list of strings (words)
    tokens = re.findall(r'\b\w+\b', text.lower())
    return list(ngrams(tokens, n))

# Generate and print unigrams (n=1)
n = 1
unigrams = generate_ngrams(text, n)
print("\nUnigrams:")
for grams in unigrams:
    print(grams)

# Generate and print bigrams (n=2)
n = 2
bigrams = generate_ngrams(text, n)
print("\nBigrams:")
for grams in bigrams:
    print(grams)

# Generate and print trigrams (n=3)
n = 3
trigrams = generate_ngrams(text, n)
print("\nTrigrams:")
for grams in trigrams:
     print(grams)
