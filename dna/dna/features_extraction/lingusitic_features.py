import math


def is_idn(domain):
    if len(domain) > 4 and domain[0:4] == 'xn--':
        return True
    else:
        return False


def level(url):
    return url.count('.') + 1


def length(url):
    if url is None:
        return 0
    return len(url)


def entropy(s):
    entropy_res = 0
    len_s = len(s)
    for x in s:
        p_x = s.count(x) / float(len_s)
        if p_x > 0:
            entropy_res += - p_x * math.log(p_x, 2)
    return entropy_res


def vowels_count(url):
    if url is None:
        return 0
    return url.count('a') + url.count('e') + url.count('i') + url.count('o') + url.count('u')


def hyphens_count(url):
    if url is None:
        return 0
    return url.count('-')


def digits_count(url):
    if url is None:
        return 0
    counter = 0
    for i in range(10):
        counter += url.count(str(i))
    return counter


def consonant_count(url):
    if url is None:
        return 0
    consonant_list = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z']
    counter = 0
    for c in consonant_list:
        counter += url.count(c)
    return counter


def digit_as_letter_cnt(url):
    return url.count('0') + url.count('1') + url.count('2') + url.count('5')


def element_ratio(elem_num, url_len):
    if url_len == 0:
        return 0
    return elem_num/url_len


def fill_lingusitic_features(features, domain, subdomain):

    features['is_idn'] = is_idn(domain)
    features['levels'] = level(features['url'])
    features['url_length'] = length(features['url'])
    features['domain_entropy'] = entropy(domain)
    features['domain_length'] = length(domain)
    features['dn_vowels_count'] = vowels_count(domain)
    features['dn_hyphens_count'] = hyphens_count(domain)
    features['dn_digits_count'] = digits_count(domain)
    features['dn_consonant_count'] = consonant_count(domain)
    features['dn_digit_as_letter_cnt'] = digit_as_letter_cnt(domain)
    features['dn_vowels_ratio'] = element_ratio(features['dn_vowels_count'], features['domain_length'])
    features['dn_digits_ratio'] = element_ratio(features['dn_digits_count'], features['domain_length'])
    features['dn_hyphens_ratio'] = element_ratio(features['dn_hyphens_count'], features['domain_length'])
    features['dn_consonant_ratio'] = element_ratio(features['dn_consonant_count'], features['domain_length'])
    features['subdomain_length'] = length(subdomain)
    features['sub_dn_consonant_count'] = consonant_count(subdomain)
    features['sub_dn_vowels_count'] = vowels_count(subdomain)
    features['sub_dn_digits_count'] = digits_count(subdomain)
    features['sub_dn_hyphens_count'] = hyphens_count(subdomain)
    features['sub_dn_vowels_ratio'] = element_ratio(features['sub_dn_vowels_count'], features['subdomain_length'])
    features['sub_dn_consonant_ratio'] = element_ratio(features['sub_dn_consonant_count'], features['subdomain_length'])
    features['sub_dn_hyphens_ratio'] = element_ratio(features['sub_dn_hyphens_count'], features['subdomain_length'])
    features['sub_dn_digits_ratio'] = element_ratio(features['sub_dn_digits_count'], features['subdomain_length'])

    return features
