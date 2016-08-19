import string, re
import nltk
import math
from sets import Set
from sklearn.linear_model import LogisticRegression
import time
from sklearn.externals import joblib
import gensim
import operator
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
import time
from nltk.corpus import wordnet as wn
import lda
import lda.datasets
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import math
import warnings
import pickle

features_df = pd.read_table("C:\Users\JareD\Major Project\EvenSem\Data\DataSet_Feature_Extraction_3.tsv")

features = ['UnigramCount', 'BigramCount', 'TrigramCount', 'LemmaCount', 'IDFUnigramCount']
features = features + ['SynonymCount', 'AntonymCount', 'HypernymCount', 'HyponymCount']
features = features + ['LevenshteinEditDistance']


X_Train = features_df[features]
y_Train = features_df['Label']

model = LogisticRegression()
model = model.fit(X_Train, y_Train)

pickle.dump(model, open("C:\Users\JareD\Major Project\EvenSem\Models\QA Classifier\qa_classifier_1.pkl", "wb"))

features = features + ['PositiveSimilarity', 'NegativeSimilarity']

X_Train = features_df[features]
y_Train = features_df['Label']

model = LogisticRegression()
model = model.fit(X_Train, y_Train)

pickle.dump(model, open("C:\Users\JareD\Major Project\EvenSem\Models\QA Classifier\qa_classifier_2.pkl", "wb"))