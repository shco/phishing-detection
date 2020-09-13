from Classes.Models.XGBoostFiles.utils import *


def fill_structural_features(df):
    df['is_idn'] = is_idn(df['domain']) if df['domain'] else False
    df['levels'] = level(df['url']) if df['domain'] else 0
    df['url_length'] = length(df['url']) if df['domain'] else 0
    df['domain_entropy'] = entropy(df['domain']) if df['domain'] else 0
    df['domain_length'] = length(df['domain']) if df['domain'] else 0
    df['dn_vowels_count'] = vowels_count(df['domain']) if df['domain'] else 0
    df['dn_hyphens_count'] = hyphens_count(df['domain']) if df['domain'] else 0
    df['dn_digits_count'] = digits_count(df['domain']) if df['domain'] else 0
    df['dn_consonant_count'] = consonant_count(df['domain']) if df['domain'] else 0
    df['dn_digit_as_letter_cnt'] = digit_as_letter_cnt(df['domain']) if df['domain'] else 0
    df['dn_vowels_ratio'] = element_ratio(df['dn_vowels_count'], df['domain_length'])
    df['dn_digits_ratio'] = element_ratio(df['dn_digits_count'], df['domain_length'])
    df['dn_hyphens_ratio'] = element_ratio(df['dn_hyphens_count'], df['domain_length'])
    df['dn_consonant_ratio'] = element_ratio(df['dn_consonant_count'], df['domain_length'])
    df['subdomain_length'] = length(df['subdomain']) if df['subdomain'] else 0
    df['sub_dn_consonant_count'] = consonant_count(df['subdomain']) if df['subdomain'] else 0
    df['sub_dn_vowels_count'] = vowels_count(df['subdomain']) if df['subdomain'] else 0
    df['sub_dn_digits_count'] = digits_count(df['subdomain']) if df['subdomain'] else 0
    df['sub_dn_hyphens_count'] = hyphens_count(df['subdomain']) if df['subdomain'] else 0
    df['sub_dn_vowels_ratio'] = element_ratio(df['sub_dn_vowels_count'], df['subdomain_length'])
    df['sub_dn_consonant_ratio'] = element_ratio(df['sub_dn_consonant_count'], df['subdomain_length'])
    df['sub_dn_hyphens_ratio'] = element_ratio(df['sub_dn_hyphens_count'], df['subdomain_length'])
    df['sub_dn_digits_ratio'] = element_ratio(df['sub_dn_digits_count'], df['subdomain_length'])

    return df


def fill_prob_features(df, domain_mat, url_mat, url_parts_mat, alexa_dict, legit_dict, word_prob_dict, tld_prob_dict):

    mc_features = calc_mc_scores([
        {"matrix": domain_mat, "column": "domain", "gram_size": 1, "string": df['domain']},
        {"matrix": url_mat, "column": "url", "gram_size": 2, "string": df['url']},
        {"matrix": url_parts_mat, "column": "url_parts", "gram_size": 1, "string": df['url']}])
    df['mc_dmn_avg_prob'] = mc_features['domain_avg_prob']
    df['mc_dmn_std_prob'] = mc_features['domain_std_prob']
    df['mc_dmn_prob'] = mc_features['domain_mul_prob']
    df['mc_url_avg_prob'] = mc_features['url_avg_prob']
    df['mc_url_std_prob'] = mc_features['url_std_prob']
    df['mc_url_prob'] = mc_features['url_mul_prob']
    df['mc_url_parts_avg_prob'] = mc_features['url_parts_avg_prob']
    df['mc_url_parts_std_prob'] = mc_features['url_parts_std_prob']
    df['mc_url_parts_prob'] = mc_features['url_parts_mul_prob']
    df['2grams_alexa_score'] = calc_2grams_score(df['url'], alexa_dict)
    df['2grams_real23_score'] = calc_2grams_score(df['url'], legit_dict)
    df['tld_prob'] = tld_prob_dict[df['tld']] if df['tld'] in tld_prob_dict.keys() else 0
    df['word_prob'] = calc_word_prob(df['url'], word_prob_dict)

    return df
