import time
from tqdm import tqdm
from Classes.Models.XGBoostFiles.features import fill_prob_features
from Classes.Models.XGBoostFiles.columns import ordered_columns, url
from Classes.Models.XGBoostFiles.utils import *
import swifter


def extract_xgboost_features(df, domain_mat, url_mat, url_parts_mat, alexa_dict, legit_dict, word_prob_dict,
                             tld_prob_dict, name, structural_features=False):
    start_time = time.time()
    tqdm.pandas()

    if structural_features:
        print("parse domain")
        df[['subdomain', 'domain', 'tld']] = df['url'].progress_apply(parse_domain).progress_apply(pd.Series)
        print('start extracting structural features')
        print('is_idn')
        df['is_idn'] = df['domain'].progress_apply(is_idn)
        print('levels')
        df['levels'] = df['url'].progress_apply(level)
        print('url length')
        df['url_length'] = df['url'].progress_apply(length)
        print('domain entropy')
        df['domain_entropy'] = df['domain'].progress_apply(entropy)
        print('domain length')
        df['domain_length'] = df['domain'].progress_apply(length)
        print('dn vowels count')
        df['dn_vowels_count'] = df['domain'].progress_apply(vowels_count)
        print('dn hyphens count')
        df['dn_hyphens_count'] = df['domain'].progress_apply(hyphens_count)
        print('dn digits count')
        df['dn_digits_count'] = df['domain'].progress_apply(digits_count)
        print('dn consonant count')
        df['dn_consonant_count'] = df['domain'].progress_apply(consonant_count)
        print('dn digit as letter cnt')
        df['dn_digit_as_letter_cnt'] = df['domain'].progress_apply(digit_as_letter_cnt)
        print('dn vowels ratio')
        df['dn_vowels_ratio'] = df.progress_apply(lambda x: element_ratio(x['dn_vowels_count'], x['domain_length']), axis=1)
        print('dn digits ratio')
        df['dn_digits_ratio'] = df.progress_apply(lambda x: element_ratio(x['dn_digits_count'], x['domain_length']), axis=1)
        print('dn hyphens ratio')
        df['dn_hyphens_ratio'] = df.progress_apply(lambda x: element_ratio(x['dn_hyphens_count'], x['domain_length']), axis=1)
        print('dn consonant ratio')
        df['dn_consonant_ratio'] = df.progress_apply(lambda x: element_ratio(x['dn_consonant_count'], x['domain_length']), axis=1)
        print('subdomain length')
        df['subdomain_length'] = df['subdomain'].progress_apply(length)
        print('sub dn consonant count')
        df['sub_dn_consonant_count'] = df['subdomain'].progress_apply(consonant_count)
        print('sub dn vowels count')
        df['sub_dn_vowels_count'] = df['subdomain'].progress_apply(vowels_count)
        print('sub_dn_digits_count')
        df['sub_dn_digits_count'] = df['subdomain'].progress_apply(digits_count)
        print('sub_dn_hyphens_count')
        df['sub_dn_hyphens_count'] = df['subdomain'].progress_apply(hyphens_count)
        print('sub_dn_vowels_ratio')
        df['sub_dn_vowels_ratio'] = df.progress_apply(lambda x: element_ratio(x['sub_dn_vowels_count'], x['subdomain_length']), axis=1)
        print('sub_dn_consonant_ratio')
        df['sub_dn_consonant_ratio'] = df.progress_apply(lambda x: element_ratio(x['sub_dn_consonant_count'], x['subdomain_length']), axis=1)
        print('sub_dn_hyphens_ratio')
        df['sub_dn_hyphens_ratio'] = df.progress_apply(lambda x: element_ratio(x['sub_dn_hyphens_count'], x['subdomain_length']), axis=1)
        print('sub_dn_digits_ratio')
        df['sub_dn_digits_ratio'] = df.progress_apply(lambda x: element_ratio(x['sub_dn_digits_count'], x['subdomain_length']), axis=1)
        df.to_parquet('./{} structural features.parquet'.format(name))

    df = pd.read_parquet('./{} structural features.parquet'.format(name))

    print('start extracting prob features')
    df = df.swifter.apply(fill_prob_features, axis=1, args=(domain_mat, url_mat, url_parts_mat, alexa_dict, legit_dict,
                                                            word_prob_dict, tld_prob_dict))
    time_predict_data = time.time() - start_time
    print(time_predict_data)
    return df


def example():
    print('load main file')
    name = 'DNS2'
    df = pd.read_csv('/home/scho/dns_model_all_url/PredictFiles/DataSets/DNS2/sample_of_200000 edited.csv', index_col=0)['without_www']
    df.columns = ['url']

    print('load dicts')
    domain_mat, url_mat, url_parts_mat = load_MC()
    alexa_dict = load_2gram_score_data("./Classes/Models/XGBoostFiles/resources/models/alexa_1m_enriched_ngrams/"
                                       "data.parquet")
    legit_dict = load_2gram_score_data("./Classes/Models/XGBoostFiles/resources/models/"
                                       "linguistic_features_legit_prod_urls_23_ngrams/data.parquet")
    word_prob_dict = pickle.load(open(f'./Classes/Models/XGBoostFiles/resources/models/MainModel/'
                                      f'Classifying_Phishing_RNN_word_prob_dict.dat', 'rb'))
    tld_prob_dict = pickle.load(open(f'./Classes/Models/XGBoostFiles/resources/models/MainModel/'
                                     f'Classifying_Phishing_RNN_tld_prob_dict.dat', 'rb'))

    df = extract_xgboost_features(df, domain_mat, url_mat, url_parts_mat, alexa_dict, legit_dict, word_prob_dict,
                                  tld_prob_dict, name=name, structural_features=True)
    return df

