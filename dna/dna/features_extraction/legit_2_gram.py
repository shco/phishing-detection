import pandas as pd


def load_legit_2gram_score_data():
    df = pd.read_parquet("resources/models/linguistic_features_legit_prod_urls_23_ngrams/data.parquet")
    df = df[df['ngram_size'] == 2]
    df = df.drop('ngram_size', axis=1)
    df['gram'] = df['gram'].str.replace(' ', '')
    gram = df['gram'].tolist()
    amount = df['amount'].tolist()
    legit_dict = dict(zip(gram, amount))
    return legit_dict


def calc_legit_2grams_score(url, legit_dict):
    sum = 0
    for c1, c2 in zip(url, url[1:]):
        if c1 + c2 in legit_dict:
            sum += legit_dict[c1 + c2]
    return sum / (len(url) - 1)


def fill_legit_2_grams_features(features, alexa_dict, url):
    features['2grams_real23_score'] = calc_legit_2grams_score(url, alexa_dict)
    return features
