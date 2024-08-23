import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.util import ngrams

# Download necessary NLTK resources
nltk.download('punkt_tab')
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

# Remove Stopwords
stop_words = set(stopwords.words('english'))

# Create an empty list to hold filtered words
filtered_words = []

# Filter out stopwords from the list of words
for word in words_lower:
    if word not in stop_words:
        filtered_words.append(word)
        
print("Filtered Words:", filtered_words)

# Lemmatization
lemmatizer = WordNetLemmatizer()

# Create an empty list to hold lemmatized words
lemmatized_words = []

# Lemmatize each word
for word in words_lower:
    lemmatized_word = lemmatizer.lemmatize(word)
    lemmatized_words.append(lemmatized_word)

print("Lemmatized Words:", lemmatized_words)

# Stemming
stemmer = PorterStemmer()

# Create an empty list to hold stemmed words
stemmed_words = []

# Stem each word
for word in words_lower:
    stemmed_word = stemmer.stem(word)
    stemmed_words.append(stemmed_word)

print("Stemmed Words:", stemmed_words)

# N-grams
# Generate bigrams (2-word combinations)
bigrams = list(ngrams(words_lower, 2))
print("Bigrams:", bigrams)

# Generate trigrams (3-word combinations)
trigrams = list(ngrams(words_lower, 3))
print("Trigrams:", trigrams)

# Function to generate n-grams from raw text
def generate_ngrams(text, n):
    # Split the text by spaces to create a list of strings (words/punctuation)
    tokens = text.split()
    return list(ngrams(tokens, n))

# unigrams (n=1)
n = 1
unigrams = generate_ngrams(text, n)
print("\nUnigrams:")
for grams in unigrams:
    print(grams)

# bigrams (n=2)
n = 2
bigrams = generate_ngrams(text, n)
print("\nBigrams:")
for grams in bigrams:
    print(grams)

# trigrams (n=3)
n = 3
trigrams = generate_ngrams(text, n)
print("\nTrigrams:")
for grams in trigrams:
    print(grams)
