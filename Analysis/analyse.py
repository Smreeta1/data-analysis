import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns  

sns.set_style('darkgrid')

# Download necessary NLTK resources
nltk.download('stopwords')

def load_data(filepath):
    """Load and preprocess the data."""
    df = pd.read_csv(filepath, encoding="ISO-8859-1")
    df.columns = ['Sentiment', 'News Headline']
    print(df.head())
    return df

def split_data(df, test_size=0.4, random_state=123):
    """Split the data into training and testing sets."""
    x = df['News Headline']
    y = df['Sentiment']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    print(f'Numpy Arrays Shape before coverting to Pandas daraframe:{x_train.shape, x_test.shape, y_train.shape, y_test.shape}')
    
    df_train = pd.DataFrame({'news': x_train, 'sentiment': y_train})
    df_test = pd.DataFrame({'news': x_test, 'sentiment': y_test})
    print(f'Before punctuation removal:{df_train.head()}')
    
    return df_train, df_test

def remove_punctuation(text):
    """Remove punctuation from a given text."""
    if isinstance(text, float):
        return text
    return ''.join(char for char in text if char not in string.punctuation)

def preprocess_text(df):
    """Apply text preprocessing steps."""
    df['news'] = df['news'].apply(remove_punctuation)
    print(f'After punctuation removal:{df.head()}')
    return df

def generate_ngrams(text, ngram=1):
    """Generate n-grams from a given text."""
    words = [word for word in text.split(" ") if word not in set(stopwords.words('english'))]
    n_grams = zip(*[words[i:] for i in range(ngram)])
    return [' '.join(ngram) for ngram in n_grams]

def count_ngrams(df):
    """Count ngrams in the dataframe based on sentiment."""
    positiveValues = defaultdict(int)
    negativeValues = defaultdict(int)
    neutralValues = defaultdict(int)
    
    for sentiment, group in df.groupby('sentiment'):
        ngram_counter = positiveValues if sentiment == 'positive' else \
                        negativeValues if sentiment == 'negative' else \
                        neutralValues
        for text in group['news']:
            for ngram in generate_ngrams(text):
                ngram_counter[ngram] += 1
    
    return positiveValues, negativeValues, neutralValues

def create_sentiment_dfs(positiveValues, negativeValues, neutralValues):
    """Create DataFrames from the n-gram counts."""
    df_positive = pd.DataFrame(sorted(positiveValues.items(), key=lambda x: x[1], reverse=True), columns=['Word', 'Count'])
    df_negative = pd.DataFrame(sorted(negativeValues.items(), key=lambda x: x[1], reverse=True), columns=['Word', 'Count'])
    df_neutral = pd.DataFrame(sorted(neutralValues.items(), key=lambda x: x[1], reverse=True), columns=['Word', 'Count'])
    return df_positive, df_negative, df_neutral

def plot_top_words(df_positive, df_negative, df_neutral):
    """Plot the top words for each sentiment."""
    # Extract top 10 words
    top_words_pos = df_positive.head(10)
    top_words_neg = df_negative.head(10)
    top_words_neu = df_neutral.head(10)
    
    # Plot for positive sentiment
    plt.figure(figsize=(16, 4))
    plt.bar(top_words_pos['Word'], top_words_pos['Count'], color='green', width=0.4)
    plt.xlabel("Words in positive dataframe")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.title("Top 10 Positive Words")
    plt.show()

    # Plot for negative sentiment
    plt.figure(figsize=(16, 4))
    plt.bar(top_words_neg['Word'], top_words_neg['Count'], color='red', width=0.4)
    plt.xlabel("Words in negative dataframe")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.title("Top 10 Negative Words")
    plt.show()

    # Plot for neutral sentiment
    plt.figure(figsize=(16, 4))
    plt.bar(top_words_neu['Word'], top_words_neu['Count'], color='blue', width=0.4)
    plt.xlabel("Words in neutral dataframe")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.title("Top 10 Neutral Words")
    plt.show()


def process_text_data(filepath, test_size=0.4, random_state=123):
    """Main function to process the text data."""
    df = load_data(filepath)
    df_train, df_test = split_data(df, test_size, random_state)
    df_train = preprocess_text(df_train)
    df_test = preprocess_text(df_test)
    
    positiveValues, negativeValues, neutralValues = count_ngrams(df_train)
    df_positive, df_negative, df_neutral = create_sentiment_dfs(positiveValues, negativeValues, neutralValues)
    plot_top_words(df_positive, df_negative, df_neutral)
    return df_train, df_test

if __name__ == "__main__":
#Generating results from a dataset:Eg. for:'all-data.csv'
 train_data, test_data = process_text_data('all-data.csv', test_size=0.4, random_state=123)

 
#Unigrams:

train_data['unigrams'] = train_data['news'].apply(lambda x: generate_ngrams(x, 1))
test_data['unigrams'] = test_data['news'].apply(lambda x: generate_ngrams(x, 1))

# train dataset with unigrams
print("Train unigrams dataframe:")
print(train_data.head())

# test dataset with bigrams
print("Test unigrams dataframe:")
print(test_data.head())

#Bigrams:
train_data['bigrams'] = train_data['news'].apply(lambda x: generate_ngrams(x, 2))
test_data['bigrams'] = test_data['news'].apply(lambda x: generate_ngrams(x, 2))

#train dataset bigrams
print("Train bigrams dataframe:")
print(train_data.head())

# test dataset bigrams
print("Test bigrams dataframe:")
print(test_data.head())

