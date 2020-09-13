import argparse
import csv
import json
import math
import mmap
import os
import threading
import multiprocessing as mp
from functools import partial
from multiprocessing.pool import ThreadPool

import tldextract
import pandas as pd
import pickle

import xgboost
from dna.colors import bcolors
from dna.features_extraction.cert_features import fill_cert_features
from dna.features_extraction.lingusitic_features import fill_lingusitic_features
from dna.features_extraction.markov_chain import fill_mc_features, load_MC
from dna.features_extraction.alexa_2_gram import fill_alexa_2_grams_features, load_alexa_2gram_score_data
from dna.features_extraction.legit_2_gram import fill_legit_2_grams_features, load_legit_2gram_score_data
from dna.features_extraction.tld_features import fill_tld_features
from dna.features_extraction.words_features import fill_words_features

from dna.features_extraction.columns import *


class DNAC:
    def __init__(self, model_path="resources/models/MainModel/offline_model.dat", threshold=0.66):
        self.set_model(model_path)
        self.threshold = threshold
        self.alexa_dict = load_alexa_2gram_score_data()
        self.legit_dict = load_legit_2gram_score_data()
        self.domain_mat, self.url_mat, self.url_parts_mat = load_MC()

    def predict_json(self, file, save_csv='__none__'):
        """
        method returns a list of predictions for a Json file
        Example:
        pred_list = predict_json('path1.Json')
        :param file: a Json contains a records
        :param save_csv: Default = '__none__', True if you wish to save the predictions to CVS, False otherwise
        :return: List of predictions, List of classifications
        :exception: FileNotFoundError: if one of the Json files does not exist
        """
        if not self.check_model():
            return None

        predictions = []
        with open(file) as fp:
            line = ''
            try:
                line = fp.readline()
            except Exception:
                print(bcolors.FAIL+'Line dropped due to corrupted data'+bcolors.ENDC)
            while line:
                predictions.append(self.predict(line))
                line = fp.readline()

        if save_csv is not '__none__':
            self.save_to_csv(predictions, save_csv)
        return predictions

    def save_to_csv(self, predictions, outout_csv):
        """
        creates a CSV file contains the predictions in (URL, Prediction, Class) format
        :param predictions: List of prediction tuples
        :param outout_csv: Path to the CSV result file
        """
        with open(outout_csv, 'w', newline='') as resultFile:
            writer = csv.DictWriter(resultFile, fieldnames=["URL", "Prediction", "Classification"])
            writer.writeheader()
            for data in predictions:
                writer.writerow(data)

    def predict(self, x=""):  # x (f1,f2,f3,f4)
        """
        Gets a single JSON line and predict its class
        :param x: JSON line: data to predict
        :return: dict contain URL, Prediction and Classification
        """
        data = json.loads(x)
        features = self.create_features(data['_source']['nrd'], data['_source']['not_after'],
                                        data['_source']['not_before'],
                                        data['_source']['all_domains'], self.alexa_dict, self.legit_dict)
        features = features[ordered_columns]
        url = features['url']
        features = features.drop('url')
        df = pd.DataFrame(features).transpose().infer_objects()
        df = xgboost.DMatrix(df)
        pred = None
        try:
            pred = self.model.predict(df)
        except Exception:
            return {'URL': None, 'Prediction': None, 'Classification': -1}
        classification = self.classify(pred)
        return {'URL': url, 'Prediction': float(pred), 'Classification': classification}

    def check_model(self):
        if self.model is None:
            print(bcolors.FAIL + 'No model found, please set a model by calling set_model(modelPath)' + bcolors.ENDC)
            return False
        return True

    def classify(self, pred):
        """
        Classifying the predictions, uses the given object threshold to classify
        prediction's index correspond to classify index
        :param pred: prediction value
        :return: 0 if lower then threshold otherwise 1.
        """
        if pred > self.threshold:
            return 1
        else:
            return 0

    def set_model(self, modelPath):
        if os.path.isfile(modelPath):
            self.model = pickle.load(open(modelPath, 'rb'))
        else:
            print(bcolors.WARNING + bcolors.UNDERLINE + 'Warning:' + bcolors.ENDC
                  + bcolors.WARNING + ' Model file not found, you may need to set it later in order to use it'
                  + bcolors.ENDC)
            self.model = None

    def parse_domain(self, d):
        res = {}
        ex_res = tldextract.extract(d)
        res['subdomain'] = ex_res.subdomain if ex_res.subdomain != '' else None
        res['domain'] = ex_res.domain if ex_res.domain != '' else None
        res['suffix'] = ex_res.suffix if ex_res.suffix != '' else None
        # TODO: check why suffix == None
        res['registered_domain'] = ex_res.registered_domain
        return res

    def create_features(self, url, not_after, not_before, all_domains, alexa_dict, legit_dict, f=None):
        features = pd.Series(index=all_features)
        features['url'] = self.drop_www(url)
        res = self.parse_domain(features['url'])
        # features = fill_cert_features(features, not_after, not_before, all_domains, f)
        features = fill_lingusitic_features(features, res['domain'], res['subdomain'])
        features = fill_mc_features(features, res['domain'],
                                    domain_mat=self.domain_mat,
                                    url_mat=self.url_mat,
                                    url_parts_mat=self.url_parts_mat)
        features = fill_alexa_2_grams_features(features, alexa_dict, features['url'])
        features = fill_legit_2_grams_features(features, legit_dict, features['url'])
        features = fill_tld_features(features, res['suffix'])
        features = fill_words_features(features, res['domain'])
        return features

    def drop_www(self, url):
        if url[0:4] == 'www.':
            return url[4:]
        else:
            return url

    def print_results(self, predictions):
        for t in predictions:
            print('\turl = {0:30}\t\tprediction = {1:.10f}\t\tclassification = {2:.0f}'
                  .format(t['URL'], t['Prediction'], t['Classification']))


def main(args):
    dnac = DNAC(model_path="resources/models/MainModel/offline_model.dat", threshold=args.threshold)
    predictions = dnac.predict_json(args.json_file, save_csv=args.save_to_csv)
    if args.prompt:
        dnac.print_results(predictions)
    with open(args.json_file) as json:
        i = 1
        for line in json:
            single_pred = dnac.predict(line)
            if args.prompt:
                print(bcolors.UNDERLINE+'line '+str(i)+':'+bcolors.ENDC)
                dnac.print_results([single_pred])
                i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', action='store_true', dest='prompt',
                        help='Print results to screen')
    parser.add_argument('-j', default='__none__', action='store',
                        dest='json_file', help='Json file to predict')
    parser.add_argument('-t', action='store', default='0.066',
                        dest='threshold', type=float,
                        help='prediction threshold')
    parser.add_argument('-s', default='__none__', action='store',
                        dest='save_to_csv',
                        help='path to the CSV file to save the outout in, output wont be saved if flag is\'nt used')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    arguments = parser.parse_args()
    main(arguments)
    print('json_file       =   ', arguments.json_file)
    print('prompt           =   ', arguments.prompt)
    print('threshold        =   ', arguments.threshold)
    print('save_to_csv      =   ', arguments.save_to_csv)


# TODO: save_to_csv arg: get a directory path from the user for the output
# TODO: save_to_csv arg: return just single file with scores and predictions
# TODO: save_to_csv arg: the csv output format should be the following columns:
#  url, score, prediction

