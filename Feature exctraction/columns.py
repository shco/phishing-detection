url = ["url"]

structural_features = ["is_idn", "levels", "url_length", "domain_entropy", "domain_length", "dn_vowels_count",
                       "dn_hyphens_count", "dn_vowels_ratio", "dn_digits_ratio", "dn_hyphens_ratio",
                       "dn_consonant_ratio", "dn_digit_as_letter_cnt", "subdomain_length", "dn_consonant_count",
                       "sub_dn_consonant_count", "sub_dn_vowels_count", "dn_digits_count", "sub_dn_digits_count",
                       "sub_dn_digits_ratio", "sub_dn_hyphens_count", "sub_dn_vowels_ratio", "sub_dn_consonant_ratio",
                       "sub_dn_hyphens_ratio"]

prob_features = ["mc_dmn_avg_prob", "mc_dmn_std_prob", "mc_dmn_prob", "mc_url_avg_prob", "mc_url_std_prob",
                 "mc_url_prob", "mc_url_parts_avg_prob", "mc_url_parts_std_prob", "mc_url_parts_prob",
                 "2grams_alexa_score", "2grams_real23_score", "word_prob", "tld_prob"]

ordered_columns = ['is_idn', 'levels', 'url_length', 'domain_entropy', 'domain_length', 'dn_vowels_count',
                   'dn_hyphens_count', 'dn_vowels_ratio', 'dn_digits_ratio', 'dn_hyphens_ratio', 'dn_consonant_ratio',
                   'dn_digit_as_letter_cnt', 'subdomain_length', 'dn_consonant_count', 'sub_dn_consonant_count',
                   'sub_dn_vowels_count', 'dn_digits_count', 'sub_dn_digits_count', 'sub_dn_hyphens_count',
                   'sub_dn_vowels_ratio', 'sub_dn_consonant_ratio', 'sub_dn_hyphens_ratio', 'mc_dmn_avg_prob',
                   'mc_dmn_std_prob', 'mc_dmn_prob', 'mc_url_avg_prob', 'mc_url_std_prob', 'mc_url_prob',
                   "mc_url_parts_avg_prob", "mc_url_parts_std_prob", "mc_url_parts_prob", 'sub_dn_digits_ratio',
                   '2grams_alexa_score', '2grams_real23_score', "word_prob", "tld_prob"]

all_features = []
all_features.extend(url)
all_features.extend(structural_features)
all_features.extend(prob_features)
