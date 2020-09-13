import os

import pandas as pd


def load_alexa_2gram_score_data():
    df = pd.read_parquet(os.path.abspath("resources/models/alexa_1m_enriched_ngrams/data.parquet"))
    df = df[df['ngram_size'] == 2]
    df = df.drop('ngram_size', axis=1)
    df['gram'] = df['gram'].str.replace(' ', '')
    gram = df['gram'].tolist()
    amount = df['amount'].tolist()
    alexa_dict = dict(zip(gram, amount))
    return alexa_dict


def calc_alexa_2grams_score(url, alexa_dict):
    sum = 0
    for c1, c2 in zip(url, url[1:]):
        if c1 + c2 in alexa_dict:
            sum += alexa_dict[c1 + c2]
    return sum / (len(url) - 1)


def fill_alexa_2_grams_features(features, alexa_dict, url):
    features['2grams_alexa_score'] = calc_alexa_2grams_score(url, alexa_dict)
    return features
