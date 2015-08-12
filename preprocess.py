#!/usr/bin/python
# -*- coding: utf-8 -*-

# add here func, to optimize in Cython

# futur compatibility

from __future__ import absolute_import
from __future__ import print_function

from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import sys
import codecs
import re
import multiprocessing
import pickle
import copy
import time
import random


# create the dictionary of labels to use to train the classifier
def create_dic(subset,nb_index):
    t0 = time.time()
    print("Copying corpus...", end="")
    sys.stdout.flush()
    # remove the copy when a better solution is found
    stt = copy.deepcopy(subset)
    print("Done. [%.02fs]"%(time.time()-t0))
    sys.stdout.flush()
    t0 = time.time()
    print("Creating dictionary of labels...")
    sys.stdout.flush()
    dic = {}
    for idx, i in enumerate(stt):
        cpt=0.0
        for x in i:
            cpt+=len(x[1])
        cc=0
        for j in i:
            tmp=[]
            for smt in j[1]:
                tmp.append("##LEN%02d ##POS%02d ##IDX%02d %s"%(cpt,cc,int((cc/cpt)*nb_index),smt))
                cc+=1
            try:
                dic[j[0]][0] += 1
                dic[j[0]][1] += tmp
            except KeyError:
                dic[j[0]] = [1,tmp]
    print("Done. [%.02fs]"%(time.time()-t0))
    return dic

# split the data into a training and testing set
def split_data(dic,classes_names,split_train_test = 0.8,truncate=-1):
    classes=[]
    tmp_x=[]
    tmp_y=[]

    def _vect_label(n,nb):
        tmp = np.zeros(nb)
        tmp[n] = 1
        return tmp

    if truncate<0:
        truncate = min(len(dic[e][1]) for e in classes_names)

    for idx,c in enumerate(classes_names):
        classes.append(idx)
        tmp_x.extend(dic[c][1][:truncate])
        tmp_y.extend(np.tile(_vect_label(idx,len(classes_names)),(truncate,1)))

    x_shuf = []
    y_shuf = []

    index_shuf = range(len(tmp_y))
    random.shuffle(index_shuf)
    for i in index_shuf:
        x_shuf.append(tmp_x[i])
        y_shuf.append(tmp_y[i])

    sp = int(len(y_shuf)*split_train_test)
    raw_x_train,raw_x_test = x_shuf[:sp],x_shuf[sp:]
    raw_y_train,raw_y_test = y_shuf[:sp],y_shuf[sp:]
    return raw_x_train, raw_y_train, raw_x_test, raw_y_test

# vectorize the data to make them usable for the classifier
def vectorize_data(raw_x_train, raw_y_train, raw_x_test, raw_y_test):
    t0 = time.time()

    print("Vectorizing the training set...",end=""); sys.stdout.flush()
    vectorizer = TfidfVectorizer(sublinear_tf=True,
                             min_df=0.0000005,
                             ngram_range=(1,1),
                             stop_words='english',
                             token_pattern=r'[a-z0-9#]{2,}')
    vectorizer.fit(raw_x_train)
    print("Done. [%.02fs]"%(time.time()-t0)); sys.stdout.flush()

    t2 = time.time()
    print("Getting features...",end=""); sys.stdout.flush()
    feature_names = vectorizer.get_feature_names()
    print("Done. [%.02fs]"%(time.time()-t2)); sys.stdout.flush()

    tmp_y_train = raw_y_train
    tmp_y_test = raw_y_test

    t3 = time.time()
    print("Creating order...",end=""); sys.stdout.flush()
    max_features = len(vectorizer.vocabulary_) + 1

    tmp_dic=vectorizer.vocabulary_

    tmp_X_train = []
    c = re.compile(r'[a-z0-9#]{2,}')
    for i in range(len(raw_x_train)):
        l = c.findall(raw_x_train[i].lower())
        x = []
        for i in l:
            try:
                x.append(tmp_dic[i])
            except KeyError:
                pass
        tmp_X_train.append(x)

    tmp_X_test = []
    for i in range(len(raw_x_test)):
        l = c.findall(raw_x_test[i].lower())
        x = []
        for i in l:
            try:
                x.append(tmp_dic[i])
            except KeyError:
                pass
        tmp_X_test.append(x)
    print("Done. [%.02fs]"%(time.time()-t3)); sys.stdout.flush()

    print("Done. [%.02fs]"%(time.time()-t0))
    return tmp_X_train, tmp_y_train, tmp_X_test, tmp_y_test, feature_names, max_features, vectorizer
