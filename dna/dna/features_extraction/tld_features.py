from dna.features_extraction.columns import tlds_features


def fill_tld_features(features, suffix):

    for f in tlds_features:
        features[f] = False

    if suffix in tlds_features:
        features[suffix] = True
        features['others'] = False
    else:
        features['others'] = True

    return features
