from dna.features_extraction.columns import words


def fill_words_features(features, domain):

    others = True
    for w in words:
        if w in domain:
            features[w] = True
            others = False
        else:
            features[w] = False

    features['others'] = others
    return features
