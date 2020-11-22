import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

TEST_SIZE = .2

raw_path = Path("raw_data") / "imdb.csv"
output_path = Path("data") / "imdb.pkl"

df = pd.read_csv(raw_path)
Y = (df.sentiment.values=="positive")

vocabulary = {
    "like": 1,
    "love": 1,
    "good": 1,
    "great": 1,
    "best": 1,
    "excellent": 1,
    "could": 0,
    "would": 0,
    "better": 0,
    "bad": 0,
    "terrible": 0,
    "worst": 0
}

vectorizer = CountVectorizer(binary=True, vocabulary=vocabulary.keys())
present = vectorizer.fit_transform(df.review).toarray()
feature_names = vectorizer.get_feature_names()
polarities = np.array([vocabulary[feature_name] for feature_name in feature_names])
L = present * polarities + (1 - present) * (1 - polarities)

L_train, L_test, Y_train, Y_test = train_test_split(L, Y, test_size=TEST_SIZE, random_state=123)
pickle.dump(([L_train, L_test], [Y_train, Y_test]), open(output_path, "wb"))
