import pytest
import pandas as pd
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from Analysis.analyse import remove_punctuation, generate_ngrams 


def test_remove_punctuation():
    assert remove_punctuation("Hello, World!") == "Hello World"
    assert remove_punctuation("She's good.") == "Shes good"
    assert remove_punctuation("Why isn't it now?") == "Why isnt it now"

def test_generate_ngramss():
    text = "Everything is real but temporary in this material world!"
    assert  generate_ngrams(text, ngram=1) == ['everything', 'is', 'real','but','temporary', 'in', 'this', 'material', 'world!']
    assert  generate_ngrams(text, ngram=2) == ['everything is', 'is real','but temporary', 'temporary in', 'in this', 'this material', 'material world!']
    assert  generate_ngrams(text, ngram=3) == ['everything is real', 'is real but','real but temporary',' but temporary in', 'temporary in this', 'in this material', 'this material world']



   
