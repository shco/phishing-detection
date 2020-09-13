import math
import os
import pickle


# This function returns a tuple for each transition found in the string x
# e.g. abcd -> ('a','b'),('b','c'),('c','d')
def extract_transitions(x, gram_size=1):
    prev = None
    for i in range(0, len(x), gram_size):
        if i + gram_size > len(x):
            break
        gram = x[i:i + gram_size]
        curr = "".join(gram)
        if prev is not None:
            yield (prev, curr)
        prev = curr


# This function associates each transition with a counter of 1.0
# e.g. abcd -> (('a','b'), 1.0),(('b','c'), 1.0),(('c','d'), 1.0)
def map_extract_transitions(x, gram_size=1):
    for transition in extract_transitions(x, gram_size):
        yield (transition, 1.0)


def calc_prob(matrix, func):
    mul_prob = None
    avg_prob = 0.0
    total = 0.0
    avg_prob_sqr = 0.0
    # Calculate the avg probability and regular probability of the transitions in the current row based on the matrix.
    for transition in func:  # Extract the transitions from the relavant column in row.
        if transition in matrix:
            avg_prob += matrix[transition]
            avg_prob_sqr += matrix[transition] * matrix[transition]
            if mul_prob is not None:
                mul_prob *= matrix[transition]
            else:
                mul_prob = matrix[transition]
        else:
            mul_prob = 0.0
        total += 1
    if total == 0:
        mul_prob = 0.0
        avg_prob = 0.0
        std_prob = -999.0
    else:
        avg_prob /= total
        avg_prob_sqr /= total
        sqrt_val = avg_prob_sqr - (avg_prob * avg_prob)
        if sqrt_val > 0:
            std_prob = math.sqrt(sqrt_val)
        else:
            std_prob = 0.0
    return avg_prob, std_prob, mul_prob


# This function calculates the transition probabilities for each row in rows based on the training matrix.
def calc_mc_scores(feature_confs):
    values = dict()
    for feature_conf in feature_confs:
        matrix = feature_conf["matrix"]  # Training matrix url or domain
        col = feature_conf["column"]  # url column or domain column
        gram_size = feature_conf["gram_size"]  # The relevant n-gram size assigned to url or domain.
        string = feature_conf["string"]
        if col == 'url_parts':
            string = string.split('.')
        avg_prob, std_prob, mul_prob = calc_prob(matrix, extract_transitions(string, gram_size))
        values['{}_avg_prob'.format(col)] = avg_prob
        values['{}_std_prob'.format(col)] = std_prob
        values['{}_mul_prob'.format(col)] = mul_prob
    return values


def load_MC():
    with open(os.path.abspath('resources/models/MarkovChainModels/domain_mat.pickle'), 'rb') as handle:
        domain_mat = pickle.load(handle)
    with open(os.path.abspath('resources/models/MarkovChainModels/url_mat.pickle'), 'rb') as handle:
        url_mat = pickle.load(handle)
    with open(os.path.abspath('resources/models/MarkovChainModels/url_parts_mat.pickle'), 'rb') as handle:
        url_parts_mat = pickle.load(handle)

    return domain_mat, url_mat, url_parts_mat


def fill_mc_features(features, domain, domain_mat, url_mat, url_parts_mat):

    mc_features = calc_mc_scores( [
        {"matrix": domain_mat, "column": "domain", "gram_size": 1, "string": domain},
        {"matrix": url_mat, "column": "url", "gram_size": 2, "string": features['url']},
        {"matrix": url_parts_mat, "column": "url_parts", "gram_size": 1, "string": features['url']}])
    features['mc_dmn_avg_prob'] = mc_features['domain_avg_prob']
    features['mc_dmn_std_prob'] = mc_features['domain_std_prob']
    features['mc_dmn_prob'] = mc_features['domain_mul_prob']
    features['mc_url_avg_prob'] = mc_features['url_avg_prob']
    features['mc_url_std_prob'] = mc_features['url_std_prob']
    features['mc_url_prob'] = mc_features['url_mul_prob']
    features['mc_url_parts_avg_prob'] = mc_features['url_parts_avg_prob']
    features['mc_url_parts_std_prob'] = mc_features['url_parts_std_prob']
    features['mc_url_parts_prob'] = mc_features['url_parts_mul_prob']

    return features
