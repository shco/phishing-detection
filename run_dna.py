from dna.dnac import DNAC

def main():

    dnac = DNAC(threshold=0.66)
    example_json = """{"_index": "fdm-2019-07", "_type": "nrd", "_id": "6DCBCWwBZW8WqxLPszJx", "_score": null, "_source": {"nrd": "hagedornpublishing.com", "date": "2019-07-19T09:11:28", "network": [], "match_quality": 0, "ips": [], "asn": [], "feed": "CertStream", "smtpd": "NotChecked", "http": "NotChecked", "https": "NotChecked", "tld": "com", "cn": "cPanel, Inc. Certification Authority", "mx": "NotChecked", "isp": "NotChecked", "num_domains": 7, "is_wildcard": false, "all_domains": ["cpanel.hagedornpublishing.com", "hagedornpublishing.com", "mail.hagedornpublishing.com", "webdisk.hagedornpublishing.com", "webmail.hagedornpublishing.com", "whm.hagedornpublishing.com", "www.hagedornpublishing.com"], "not_before": 1431907200, "not_after": 1747526399}, "sort": [32615328]}"""
    pred_response = dnac.predict(x=example_json)
    print(pred_response)
    return pred_response

if __name__ == '__main__':

    main()
