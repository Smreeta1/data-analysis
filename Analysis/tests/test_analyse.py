import pytest
import pandas as pd
from Analysis.analyse import remove_punctuation, generate_ngrams, generate_unigrams_bigrams

def test_remove_punctuation():
    assert remove_punctuation("Hello, World!") == "Hello World"
    assert remove_punctuation("She's good.") == "Shes good"
    assert remove_punctuation("Why isn't it now?") == "Why isnt it now"

def test_generate_ngrams():
    text = "everything is real but temporary in this material world"
    
    # Expected output 
    expected_unigrams = ['everything', 'real', 'temporary', 'material', 'world']
    expected_bigrams = ['everything real', 'real temporary', 'temporary material', 'material world']
    expected_trigrams = ['everything real temporary', 'real temporary material', 'temporary material world']
    
    # Check if ngrams are generated correctly
    assert generate_ngrams(text, ngram=1) == expected_unigrams
    assert generate_ngrams(text, ngram=2) == expected_bigrams
    assert generate_ngrams(text, ngram=3) == expected_trigrams


def test_generate_unigrams_bigrams():
    
    data = {
        'news': ["hello world", "python testing"]
    }
    df = pd.DataFrame(data)
    
    # Training and testing on the same data
    train_data = df
    test_data = df
    
    # Generate unigrams and bigrams
    train_data, test_data = generate_unigrams_bigrams(train_data, test_data)
    
    # Check if unigrams and bigrams columns are generated
    assert 'unigrams' in train_data.columns
    assert 'bigrams' in train_data.columns
    assert 'unigrams' in test_data.columns
    assert 'bigrams' in test_data.columns

    # Check the content of the first row using .at[]
    assert train_data.at[0, 'unigrams'] == ['hello', 'world']
    assert train_data.at[0, 'bigrams'] == ['hello world']

    #second row
    assert train_data.at[1, 'unigrams'] == ['python', 'testing']
    assert train_data.at[1, 'bigrams'] == ['python testing']

    # for test_data
    assert test_data.at[0, 'unigrams'] == ['hello', 'world']
    assert test_data.at[0, 'bigrams'] == ['hello world']

    assert test_data.at[1, 'unigrams'] == ['python', 'testing']
    assert test_data.at[1, 'bigrams'] == ['python testing']
