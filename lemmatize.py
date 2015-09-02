#!/usr/bin/python
# -*- coding: utf-8 -*-

# add here func, to optimize in Cython

# futur compatibility

from __future__ import absolute_import
from __future__ import print_function

#from textblob import Word
#from textblob import TextBlob
#from textblob import Blobber
#from textblob_aptagger import PerceptronTagger
import spacy
from spacy.en import English

import multiprocessing
import numpy as np
import sys
import codecs
import re

import time

# lemmatize a sentence
def sub_lemm(subset):
    suba = []
    nlp =  English()
    for abst in subset:
        t_a=[]
        for labl in abst:
            t_l=[]
            for t in labl[1]:
                doc = nlp(t)
                t_l.append(' '.join([i.lemma_.lower() for i in doc if not i.is_punct and not i.is_space]))
            t_a.append([labl[0],t_l])
        suba.append(t_a)
    return suba

# multiprocessing to slightly speed up the process
def lemm(subset,show_time=True,nb_core=multiprocessing.cpu_count()):
    print("Working on %d core..."%(nb_core))
    print("Splitting datas...",end=" ")
    sys.stdout.flush()
    out=[]
    d={}
    results=[]
    pool = multiprocessing.Pool(nb_core)

    for x in range(0,nb_core):
        d["tab{0}".format(x)]=[]
        #d["tab%d"%x]=[]

    for idx,i in enumerate(subset):
        d['tab%d'%(idx%nb_core)].append(i)

    t0 = time.time()
    for x in range(0,nb_core):
        ress = pool.apply_async(sub_lemm,[d['tab%d'%(x%nb_core)]])
        results.append(ress)
    print("Done. [%.02fs]"%(time.time()-t0))
    print("Lemmatizing...")
    sys.stdout.flush()
    t0 = time.time()
    pool.close()
    pool.join()
    for i in results:
        out.extend(i.get())
    t=time.time()-t0
    print("Done. [%dmin %ds]"%(t/60,t%60))
    return out
