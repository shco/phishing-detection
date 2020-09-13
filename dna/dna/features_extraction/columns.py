url = ["url"]

cert_features = ["cert_muti_domain_cert", "cert_wildcard_domain", "cert_validty_period_days", "cert_num_domains",
                 "cert_num_distinct_domains"]

lingusitic_features = ["is_idn", "levels", "url_length", "domain_entropy", "domain_length", "dn_vowels_count",
                       "dn_hyphens_count", "dn_vowels_ratio", "dn_digits_ratio", "dn_hyphens_ratio",
                       "dn_consonant_ratio", "dn_digit_as_letter_cnt", "subdomain_length", "dn_consonant_count",
                       "sub_dn_consonant_count", "sub_dn_vowels_count", "dn_digits_count", "sub_dn_digits_count",
                       "sub_dn_digits_ratio", "sub_dn_hyphens_count", "sub_dn_vowels_ratio", "sub_dn_consonant_ratio",
                       "sub_dn_hyphens_ratio"]

markov_chain_features = ["mc_dmn_avg_prob", "mc_dmn_std_prob", "mc_dmn_prob", "mc_url_avg_prob",
                         "mc_url_std_prob", "mc_url_prob", "mc_url_parts_avg_prob", "mc_url_parts_std_prob",
                         "mc_url_parts_prob"]

ngrams_features = ["2grams_alexa_score", "2grams_real23_score"]

tlds_features = ["com", "edu", "net", "jp", "org", "de", "co.uk", "ru", "pl", "ca", "rs", "me", "blog", "fr", "com.au",
                 "it", "gov", "nl", "com.br", "ch", "us", "ac.uk", "in", "on.ca", "ac.id", "go.id", "io", "es", "cz",
                 "eu", "co", "business", "ir", "se", "ua", "ma", "info", "gr", "vn", "edu.au", "ro", "co.za", "dk",
                 "cc", "hu", "co.jp", "be", "com.tr", "co.il", "network", "no", "pt", "fi", "com.ua", "cn", "cl", "tv",
                 "kz", "site", "at", "com.tw", "co.in", "sk", "pro", "domains", "co.id", "nu", "bg", "br", "by",
                 "pe", "ac.jp", "biz", "place", "online", "xn--p1ai", "hr", "co.nz", "gov.au", "tw", "com.vn", "ee",
                 "com.ar", "ae", "mx", "com.my", "edu.tw", "gov.tw", "id", "si", "ie", "lv", "ba", "az", "lt", "com.sg",
                 "gov.in", "com.cn", "com.mx", "co.kr", "mil", "xyz", "club", "to", "sg", "ly", "ne.jp", "org.uk", "vu",
                 "ac.in", "pk", "su", "uz", "co.th", "ac.at", "edu.pl", "asia", "lk", "edu.ar", "gov.uk", "com.hk",
                 "help", "build", "com.pk", "my", "edu.vn", "edu.ng", "com.co", "qc.ca", "int", "org.au", "host", "pw",
                 "com.ng", "is", "top", "gov.br", "ht", "edu.my", "edu.sg", "ac.nz", "cloud", "edu.hk", "edu.pk",
                 "ac.th", "gouv.fr", "ac.il", "ge", "com.ph", "am", "bid", "edu.cn", "com.sa", "ws", "edu.tr", "mk",
                 "or.id", "hk", "ai", "edu.pe", "men", "org.br", "ng", "cat", "ac.kr", "ac.be", "co.ke", "ac.bd", "link",
                 "fo", "ph", "tn", "edu.ua", "stream", "edu.br", "travel", "gob.mx", "tk", "edu.co", "edu.ly", "org.il",
                 "go.jp", "ac.ir", "edu.sa", "mn", "mobi", "co.rs", "govt.nz", "tc", "dz", "gc.ca", "kr", "org.tw",
                 "bc.ca", "gov.my", "org.tr", "edu.in", "direct", "com.pe", "go.kr", "mn.us", "web.id", "news", "sa",
                 "gd", "gov.sa", "nsw.edu.au", "gob.pe", "or.jp", "md", "fm", "ac.za", "org.ua", "ml", "in.ua", "xxx",
                 "gov.ua", "com.bd", "tech", "social", "nhs.uk", "gv.at", "in.th", "shop", "one", "lu", "gov.co",
                 "edu.bd", "space", "bz", "go.th", "qld.gov.au", "com.ec", "aero", "asn.au", "com.ve", "gov.ae", "live",
                 "ac.ae", "al", "uk", "cu", "win", "name", "ac.rs", "gov.vn", "org.in", "com.uy", "net.au", "gov.bd",
                 "com.eg", "today", "kg", "im", "edu.ph", "gov.ph", "guru", "life", "bel.tr", "gg", "mi.us", "tm",
                 "com.pl", "ab.ca", "sx", "vic.gov.au", "com.mk", "edu.ec", "net.tr", "video", "ga", "media", "global",
                 "jus.br", "ps", "cr", "download", "wa.gov.au", "gen.tr", "nic.in", "edu.iq", "cm", "kiev.ua", "gob.ec",
                 "work", "gov.ar", "gob.ar", "org.my", "co.tz", "gratis", "com.np", "ac.cr", "or.th", "taipei",
                 "gov.tr", "gov.pk", "ac.cy", "cf", "net.ua", "gov.sg", "edu.mx", "fun", "website", "edu.rs", "onl",
                 "academy", "ac.ug", "net.nz", "com.do", "coop", "center", "world", "sch.id", "tx.us", "porn", "store",
                 "li", "ninja", "jobs", "or.kr", "lg.jp", "cv", "tj", "cx", "eus", "qa", "org.nz", "edu.gh", "ar",
                 "com.bo", "na", "mu", "la", "net.vn", "gov.mn", "com.cy", "gov.hk", "sp.gov.br", "org.za", "app", "st",
                 "org.sa", "co.us", "iq", "ac.ke", "gov.cn", "org.sg", "gov.il", "edu.ge", "edu.eg", "com.py", "ac.tz",
                 "gq", "plus", "org.rs", "kommune.no", "net.br", "sc", "org.pk", "tools", "nj.us", "moe", "rocks",
                 "edu.az", "so", "com.tn", "gov.ng", "com.kh", "zone", "hosting", "ec", "or.us", "edu.qa", "k12.or.us",
                 "gov.by", "com.gt", "fl.us", "gl", "ma.us", "org.pl", "events", "run", "com.kw", "go.ke", "com.qa",
                 "nz", "edu.bo", "gov.za", "ag", "gov.pl", "edu.mk", "services", "net.in", "gov.it", "org.hk", "edu.lb",
                 "mp.br", "bo", "click", "org.bd", "ac.ma", "ac", "re", "net.sa", "design", "wiki", "vc", "org.mx",
                 "edu.ru", "gov.kw", "digital", "co.mz", "dating", "com.sv", "com.gr", "in.rs", "tips", "agency",
                 "net.id", "om", "cd", "org.ar", "chat", "com.lb", "gov.lk", "gov.qa", "ci", "vip", "sn", "gob.bo",
                 "press", "church", "black", "market",  "expert", "company", "co.ao", "blog.br", "net.tw", "com.hr",
                 "gov.om", "com.gh", "ms", "md.us", "go.tz", "ovh", "bi", "com.pt", "k12.ca.us", "city", "software",
                 "mg", "com.mt", "co.zw", "sh", "co.zm", "games", "nm.us", "email", "leg.br", "technology", "sd.us",
                 "co.ug", "tokyo", "web.tr", "gov.ru", "net.my", "solutions", "id.au", "org.ng", "exchange",
                 "sa.gov.au", "zp.ua", "edu.uy", "bw", "do", "nc.us", "res.in", "edu.ni", "com.mm", "ao", "cash", "nc",
                 "team", "net.pl", "mb.ca", "af", "net.pk", "edu.om", "ca.us", "edu.np", "gov.pt", "mt", "pa.us",
                 "edu.cu", "dp.ua", "yt", "com.pa", "cool", "police.uk"]

ordered_columns = ['url', 'is_idn', 'levels', 'url_length', 'domain_entropy', 'domain_length',
             'dn_vowels_count', 'dn_hyphens_count', 'dn_vowels_ratio', 'dn_digits_ratio', 'dn_hyphens_ratio',
             'dn_consonant_ratio', 'dn_digit_as_letter_cnt', 'subdomain_length', 'dn_consonant_count',
             'sub_dn_consonant_count', 'sub_dn_vowels_count', 'dn_digits_count', 'sub_dn_digits_count',
             'sub_dn_hyphens_count', 'sub_dn_vowels_ratio', 'sub_dn_consonant_ratio', 'sub_dn_hyphens_ratio',
             'mc_dmn_avg_prob', 'mc_dmn_std_prob', 'mc_dmn_prob', 'mc_url_avg_prob', 'mc_url_std_prob', 'mc_url_prob',
             'mc_url_parts_avg_prob', 'mc_url_parts_std_prob', 'mc_url_parts_prob', 'sub_dn_digits_ratio',
             '2grams_alexa_score', '2grams_real23_score', 'com', 'edu', 'net', 'jp', 'org', 'de', 'co.uk', 'ru', 'pl',
             'ca', 'rs', 'me', 'blog', 'fr', 'com.au', 'it', 'gov', 'nl', 'com.br', 'ch', 'us', 'ac.uk', 'in', 'on.ca',
             'ac.id', 'go.id', 'io', 'es', 'cz', 'eu', 'co', 'business', 'ir', 'se', 'ua', 'ma', 'info', 'gr', 'vn',
             'edu.au', 'ro', 'co.za', 'dk', 'cc', 'hu', 'co.jp', 'be', 'com.tr', 'co.il', 'network', 'no', 'pt', 'fi',
             'com.ua', 'cn', 'cl', 'tv', 'kz', 'site', 'at', 'com.tw', 'co.in', 'sk', 'pro', 'domains', 'co.id', 'nu',
             'bg', 'br', 'by', 'pe', 'ac.jp', 'biz', 'place', 'online', 'xn--p1ai', 'hr', 'co.nz', 'gov.au', 'tw',
             'com.vn', 'ee', 'com.ar', 'ae', 'mx', 'com.my', 'edu.tw', 'gov.tw', 'id', 'si', 'ie', 'lv', 'ba', 'az',
             'lt', 'com.sg', 'gov.in', 'com.cn', 'com.mx', 'co.kr', 'mil', 'xyz', 'club', 'to', 'sg', 'ly', 'ne.jp',
             'org.uk', 'vu', 'ac.in', 'pk', 'su', 'uz', 'co.th', 'ac.at', 'edu.pl', 'asia', 'lk', 'edu.ar', 'gov.uk',
             'com.hk', 'help', 'build', 'com.pk', 'my', 'edu.vn', 'edu.ng', 'com.co', 'qc.ca', 'int', 'org.au', 'host',
             'pw', 'com.ng', 'is', 'top', 'gov.br', 'ht', 'edu.my', 'edu.sg', 'ac.nz', 'cloud', 'edu.hk', 'edu.pk',
             'ac.th', 'gouv.fr', 'ac.il', 'ge', 'com.ph', 'am', 'bid', 'edu.cn', 'com.sa', 'ws', 'edu.tr', 'mk',
             'or.id', 'hk', 'ai', 'edu.pe', 'men', 'org.br', 'ng', 'cat', 'ac.kr', 'ac.be', 'co.ke', 'ac.bd', 'link',
             'fo', 'ph', 'tn', 'edu.ua', 'stream', 'edu.br', 'travel', 'gob.mx', 'tk', 'edu.co', 'edu.ly', 'org.il',
             'go.jp', 'ac.ir', 'edu.sa', 'mn', 'mobi', 'co.rs', 'govt.nz', 'tc', 'dz', 'gc.ca', 'kr', 'org.tw', 'bc.ca',
             'gov.my', 'org.tr', 'edu.in', 'direct', 'com.pe', 'go.kr', 'mn.us', 'web.id', 'news', 'sa', 'gd', 'gov.sa',
             'nsw.edu.au', 'gob.pe', 'or.jp', 'md', 'fm', 'ac.za', 'org.ua', 'ml', 'in.ua', 'xxx', 'gov.ua', 'com.bd',
             'tech', 'social', 'nhs.uk', 'gv.at', 'in.th', 'shop', 'one', 'lu', 'gov.co', 'edu.bd', 'space', 'bz',
             'go.th', 'qld.gov.au', 'com.ec', 'aero', 'asn.au', 'com.ve', 'gov.ae', 'live', 'ac.ae', 'al', 'uk', 'cu',
             'win', 'name', 'ac.rs', 'gov.vn', 'org.in', 'com.uy', 'net.au', 'gov.bd', 'com.eg', 'today', 'kg', 'im',
             'edu.ph', 'gov.ph', 'guru', 'life', 'bel.tr', 'gg', 'mi.us', 'tm', 'com.pl', 'ab.ca', 'sx', 'vic.gov.au',
             'com.mk', 'edu.ec', 'net.tr', 'video', 'ga', 'media', 'global', 'jus.br', 'ps', 'cr', 'download',
             'wa.gov.au', 'gen.tr', 'nic.in', 'edu.iq', 'cm', 'kiev.ua', 'gob.ec', 'work', 'gov.ar', 'gob.ar', 'org.my',
             'co.tz', 'gratis', 'com.np', 'ac.cr', 'or.th', 'taipei', 'gov.tr', 'gov.pk', 'ac.cy', 'cf', 'net.ua',
             'gov.sg', 'edu.mx', 'fun', 'website', 'edu.rs', 'onl', 'academy', 'ac.ug', 'net.nz', 'com.do', 'coop',
             'center', 'world', 'sch.id', 'tx.us', 'porn', 'store', 'li', 'ninja', 'jobs', 'or.kr', 'lg.jp', 'cv', 'tj',
             'cx', 'eus', 'qa', 'edu.gh', 'com.bo', 'ar', 'org.nz', 'na', 'mu', 'la', 'net.vn', 'gov.mn', 'com.cy',
             'gov.hk', 'sp.gov.br', 'org.za', 'app', 'st', 'org.sa', 'co.us', 'iq', 'ac.ke', 'gov.cn', 'org.sg',
             'gov.il', 'edu.ge', 'edu.eg', 'com.py', 'ac.tz', 'gq', 'plus', 'org.rs', 'kommune.no', 'net.br', 'sc',
             'org.pk', 'tools', 'nj.us', 'moe', 'rocks', 'edu.az', 'so', 'com.tn', 'gov.ng', 'com.kh', 'zone',
             'hosting', 'ec', 'or.us', 'edu.qa', 'k12.or.us', 'gov.by', 'com.gt', 'fl.us', 'gl', 'ma.us', 'org.pl',
             'events', 'run', 'com.kw', 'go.ke', 'com.qa', 'nz', 'edu.bo', 'gov.za', 'ag', 'gov.pl', 'edu.mk',
             'services', 'gov.it', 'net.in', 'org.hk', 'mp.br', 'edu.lb', 'bo', 'click', 'org.bd', 'ac.ma', 'ac', 're',
             'net.sa', 'design', 'wiki', 'vc', 'org.mx', 'edu.ru', 'gov.kw', 'digital', 'co.mz', 'dating', 'com.sv',
             'com.gr', 'in.rs', 'tips', 'agency', 'om', 'net.id', 'cd', 'org.ar', 'chat', 'com.lb', 'gov.lk', 'gov.qa',
             'ci', 'vip', 'sn', 'gob.bo', 'press', 'church', 'black', 'market', 'expert', 'company', 'co.ao', 'blog.br',
             'net.tw', 'com.hr', 'gov.om', 'com.gh', 'ms', 'md.us', 'go.tz', 'ovh', 'bi', 'com.pt', 'k12.ca.us', 'city',
             'software', 'mg', 'com.mt', 'co.zw', 'co.zm', 'sh', 'games', 'nm.us', 'email', 'leg.br', 'technology',
             'sd.us', 'co.ug', 'tokyo', 'web.tr', 'gov.ru', 'net.my', 'solutions', 'id.au', 'org.ng', 'exchange',
             'sa.gov.au', 'zp.ua', 'edu.uy', 'do', 'res.in', 'bw', 'nc.us', 'edu.ni', 'com.mm', 'ao', 'cash', 'nc',
             'team', 'net.pl', 'mb.ca', 'af', 'net.pk', 'edu.om', 'ca.us', 'edu.np', 'gov.pt', 'mt', 'pa.us', 'edu.cu',
             'dp.ua', 'yt', 'com.pa', 'cool', 'police.uk', 'others', 'web', 'mail',
             'webmail']

# ordered_columns = ['url', 'cert_muti_domain_cert', 'cert_wildcard_domain', 'cert_validty_period_days', 'cert_num_domains',
#              'cert_num_distinct_domains', 'is_idn', 'levels', 'url_length', 'domain_entropy', 'domain_length',
#              'dn_vowels_count', 'dn_hyphens_count', 'dn_vowels_ratio', 'dn_digits_ratio', 'dn_hyphens_ratio',
#              'dn_consonant_ratio', 'dn_digit_as_letter_cnt', 'subdomain_length', 'dn_consonant_count',
#              'sub_dn_consonant_count', 'sub_dn_vowels_count', 'dn_digits_count', 'sub_dn_digits_count',
#              'sub_dn_hyphens_count', 'sub_dn_vowels_ratio', 'sub_dn_consonant_ratio', 'sub_dn_hyphens_ratio',
#              'mc_dmn_avg_prob', 'mc_dmn_std_prob', 'mc_dmn_prob', 'mc_url_avg_prob', 'mc_url_std_prob', 'mc_url_prob',
#              'mc_url_parts_avg_prob', 'mc_url_parts_std_prob', 'mc_url_parts_prob', 'sub_dn_digits_ratio',
#              '2grams_alexa_score', '2grams_real23_score', 'com', 'edu', 'net', 'jp', 'org', 'de', 'co.uk', 'ru', 'pl',
#              'ca', 'rs', 'me', 'blog', 'fr', 'com.au', 'it', 'gov', 'nl', 'com.br', 'ch', 'us', 'ac.uk', 'in', 'on.ca',
#              'ac.id', 'go.id', 'io', 'es', 'cz', 'eu', 'co', 'business', 'ir', 'se', 'ua', 'ma', 'info', 'gr', 'vn',
#              'edu.au', 'ro', 'co.za', 'dk', 'cc', 'hu', 'co.jp', 'be', 'com.tr', 'co.il', 'network', 'no', 'pt', 'fi',
#              'com.ua', 'cn', 'cl', 'tv', 'kz', 'site', 'at', 'com.tw', 'co.in', 'sk', 'pro', 'domains', 'co.id', 'nu',
#              'bg', 'br', 'by', 'pe', 'ac.jp', 'biz', 'place', 'online', 'xn--p1ai', 'hr', 'co.nz', 'gov.au', 'tw',
#              'com.vn', 'ee', 'com.ar', 'ae', 'mx', 'com.my', 'edu.tw', 'gov.tw', 'id', 'si', 'ie', 'lv', 'ba', 'az',
#              'lt', 'com.sg', 'gov.in', 'com.cn', 'com.mx', 'co.kr', 'mil', 'xyz', 'club', 'to', 'sg', 'ly', 'ne.jp',
#              'org.uk', 'vu', 'ac.in', 'pk', 'su', 'uz', 'co.th', 'ac.at', 'edu.pl', 'asia', 'lk', 'edu.ar', 'gov.uk',
#              'com.hk', 'help', 'build', 'com.pk', 'my', 'edu.vn', 'edu.ng', 'com.co', 'qc.ca', 'int', 'org.au', 'host',
#              'pw', 'com.ng', 'is', 'top', 'gov.br', 'ht', 'edu.my', 'edu.sg', 'ac.nz', 'cloud', 'edu.hk', 'edu.pk',
#              'ac.th', 'gouv.fr', 'ac.il', 'ge', 'com.ph', 'am', 'bid', 'edu.cn', 'com.sa', 'ws', 'edu.tr', 'mk',
#              'or.id', 'hk', 'ai', 'edu.pe', 'men', 'org.br', 'ng', 'cat', 'ac.kr', 'ac.be', 'co.ke', 'ac.bd', 'link',
#              'fo', 'ph', 'tn', 'edu.ua', 'stream', 'edu.br', 'travel', 'gob.mx', 'tk', 'edu.co', 'edu.ly', 'org.il',
#              'go.jp', 'ac.ir', 'edu.sa', 'mn', 'mobi', 'co.rs', 'govt.nz', 'tc', 'dz', 'gc.ca', 'kr', 'org.tw', 'bc.ca',
#              'gov.my', 'org.tr', 'edu.in', 'direct', 'com.pe', 'go.kr', 'mn.us', 'web.id', 'news', 'sa', 'gd', 'gov.sa',
#              'nsw.edu.au', 'gob.pe', 'or.jp', 'md', 'fm', 'ac.za', 'org.ua', 'ml', 'in.ua', 'xxx', 'gov.ua', 'com.bd',
#              'tech', 'social', 'nhs.uk', 'gv.at', 'in.th', 'shop', 'one', 'lu', 'gov.co', 'edu.bd', 'space', 'bz',
#              'go.th', 'qld.gov.au', 'com.ec', 'aero', 'asn.au', 'com.ve', 'gov.ae', 'live', 'ac.ae', 'al', 'uk', 'cu',
#              'win', 'name', 'ac.rs', 'gov.vn', 'org.in', 'com.uy', 'net.au', 'gov.bd', 'com.eg', 'today', 'kg', 'im',
#              'edu.ph', 'gov.ph', 'guru', 'life', 'bel.tr', 'gg', 'mi.us', 'tm', 'com.pl', 'ab.ca', 'sx', 'vic.gov.au',
#              'com.mk', 'edu.ec', 'net.tr', 'video', 'ga', 'media', 'global', 'jus.br', 'ps', 'cr', 'download',
#              'wa.gov.au', 'gen.tr', 'nic.in', 'edu.iq', 'cm', 'kiev.ua', 'gob.ec', 'work', 'gov.ar', 'gob.ar', 'org.my',
#              'co.tz', 'gratis', 'com.np', 'ac.cr', 'or.th', 'taipei', 'gov.tr', 'gov.pk', 'ac.cy', 'cf', 'net.ua',
#              'gov.sg', 'edu.mx', 'fun', 'website', 'edu.rs', 'onl', 'academy', 'ac.ug', 'net.nz', 'com.do', 'coop',
#              'center', 'world', 'sch.id', 'tx.us', 'porn', 'store', 'li', 'ninja', 'jobs', 'or.kr', 'lg.jp', 'cv', 'tj',
#              'cx', 'eus', 'qa', 'edu.gh', 'com.bo', 'ar', 'org.nz', 'na', 'mu', 'la', 'net.vn', 'gov.mn', 'com.cy',
#              'gov.hk', 'sp.gov.br', 'org.za', 'app', 'st', 'org.sa', 'co.us', 'iq', 'ac.ke', 'gov.cn', 'org.sg',
#              'gov.il', 'edu.ge', 'edu.eg', 'com.py', 'ac.tz', 'gq', 'plus', 'org.rs', 'kommune.no', 'net.br', 'sc',
#              'org.pk', 'tools', 'nj.us', 'moe', 'rocks', 'edu.az', 'so', 'com.tn', 'gov.ng', 'com.kh', 'zone',
#              'hosting', 'ec', 'or.us', 'edu.qa', 'k12.or.us', 'gov.by', 'com.gt', 'fl.us', 'gl', 'ma.us', 'org.pl',
#              'events', 'run', 'com.kw', 'go.ke', 'com.qa', 'nz', 'edu.bo', 'gov.za', 'ag', 'gov.pl', 'edu.mk',
#              'services', 'gov.it', 'net.in', 'org.hk', 'mp.br', 'edu.lb', 'bo', 'click', 'org.bd', 'ac.ma', 'ac', 're',
#              'net.sa', 'design', 'wiki', 'vc', 'org.mx', 'edu.ru', 'gov.kw', 'digital', 'co.mz', 'dating', 'com.sv',
#              'com.gr', 'in.rs', 'tips', 'agency', 'om', 'net.id', 'cd', 'org.ar', 'chat', 'com.lb', 'gov.lk', 'gov.qa',
#              'ci', 'vip', 'sn', 'gob.bo', 'press', 'church', 'black', 'market', 'expert', 'company', 'co.ao', 'blog.br',
#              'net.tw', 'com.hr', 'gov.om', 'com.gh', 'ms', 'md.us', 'go.tz', 'ovh', 'bi', 'com.pt', 'k12.ca.us', 'city',
#              'software', 'mg', 'com.mt', 'co.zw', 'co.zm', 'sh', 'games', 'nm.us', 'email', 'leg.br', 'technology',
#              'sd.us', 'co.ug', 'tokyo', 'web.tr', 'gov.ru', 'net.my', 'solutions', 'id.au', 'org.ng', 'exchange',
#              'sa.gov.au', 'zp.ua', 'edu.uy', 'do', 'res.in', 'bw', 'nc.us', 'edu.ni', 'com.mm', 'ao', 'cash', 'nc',
#              'team', 'net.pl', 'mb.ca', 'af', 'net.pk', 'edu.om', 'ca.us', 'edu.np', 'gov.pt', 'mt', 'pa.us', 'edu.cu',
#              'dp.ua', 'yt', 'com.pa', 'cool', 'police.uk', 'others', 'web', 'mail',
#              'webmail']

other_tld_feature = ['others']

words = ['web', 'mail', 'webmail']

all_features = []
all_features.extend(url)
all_features.extend(cert_features)
all_features.extend(lingusitic_features)
all_features.extend(markov_chain_features)
all_features.extend(ngrams_features)
all_features.extend(tlds_features)
all_features.extend(other_tld_feature)
all_features.extend(words)