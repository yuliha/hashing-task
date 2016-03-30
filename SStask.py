__author__ = 'yuli'

import urllib2
import json
import csv

#from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

#from sklearn.neighbors import DistanceMetric
#from scipy.sparse import coo_matrix
import numpy as np

def fetch_desc(trackId):
    url = "https://itunes.apple.com/lookup?id=" + trackId

    request = urllib2.Request(url)
    response = urllib2.urlopen(request)

    json_obj = json.load(response)
    description = json_obj["results"][0]["description"]
    return description

def file2list(filename):
    lines = [line.rstrip('\n') for line in open(filename)]
    lines = lines[:len(lines)-1]
    return lines

def hash_data(data, n_feat):
    hasher = HashingVectorizer(n_features= n_feat,
                                       stop_words='english', non_negative=True,
                                       norm=None, binary=False)

    vectorizer = make_pipeline(hasher, TfidfTransformer())
    D = vectorizer.fit_transform(data)
    return D

def dist_metric(X, method):

    #dist = DistanceMetric.get_metric(method)
    import sklearn.metrics.pairwise as DistanceMetric
    dist = DistanceMetric.linear_kernel((X))    # Thats cosine distance for already normalized vectors

    #Xcoo = coo_matrix(X, dtype=float)
    #Xdens = coo_matrix(Xcoo, dtype=float).todense()
    #dist_metric = dist.pairwise(Xdens)
    return dist

def output_csv(Ids, dist_metric):
    with open('table_file212linKern.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        writer.writerow(Ids)
        [writer.writerow(list(r)+ [Ids[idx]]) for idx,r in enumerate(dist_metric)]
    return

def my_func():
    ids = file2list('ids.txt')
    descs = [fetch_desc(id) for id in ids]

    D = hash_data(descs, n_feat= 2**12)

    output_csv(ids, dist_metric(D, ''))
    return


if __name__ == "__main__":
    my_func()