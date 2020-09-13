from datetime import datetime
import tldextract


def get_domains_features(domains_list):
    res = {}
    res['cert_num_domains'] = len(domains_list)

    if len(domains_list) > 1:
        all_domains_array = []
        distinct_domains = set()
        wildcard_domains = set()

        for d in domains_list:
            parsed_domain = tldextract.extract(d)
            registered_domain = parsed_domain.registered_domain
            distinct_domains.add(registered_domain)
            if parsed_domain.subdomain.startswith('*'):
                wildcard_domains.add(registered_domain)

        for d in domains_list:
            if d.startswith('*') and (d[2:] in distinct_domains):
                continue
            else:
                all_domains_array.append(d)

        # count number of distinct domains in current certificate
        num_distinct_domains = len(distinct_domains)
        # if cert contains multiple distinct domains flag it as muti_domain_cert
        muti_domain_cert = num_distinct_domains > 1

        res['cert_num_distinct_domains'] = num_distinct_domains
        res['muti_domain_cert'] = muti_domain_cert
        res['cert_wildcard_domains'] = list(wildcard_domains)
        res['all_domains_array'] = all_domains_array

    else:
        res['cert_num_distinct_domains'] = 1
        res['muti_domain_cert'] = False
        res['cert_wildcard_domains'] = domains_list
        res['all_domains_array'] = domains_list

    return res


def fill_cert_features(features, not_after, not_before, all_domains, res=None):

    not_after_dt = datetime.fromtimestamp(not_after)
    not_before_dt = datetime.fromtimestamp(not_before)
    features['cert_validty_period_days'] = (not_after_dt - not_before_dt).days
    if res is None:
        res = get_domains_features(all_domains)
    features['cert_num_distinct_domains'] = res['cert_num_distinct_domains']
    features['cert_muti_domain_cert'] = res['muti_domain_cert']
    features['cert_wildcard_domain'] = len(res['cert_wildcard_domains']) > 0
    features['cert_num_domains'] = res['cert_num_domains']

    return features
