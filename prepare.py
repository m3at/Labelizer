#!/usr/bin/python
# -*- coding: utf-8 -*-

# add here func, to optimize in Cython

# futur compatibility

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import sys
import codecs
import re
import multiprocessing
import pickle

import time



def sizeof_h(num, suffix='B'):
    for unit in ['','K','M','G','T']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    while num>10:
        num/=10
        cpt+=1
    return "%de%d" % (num,cpt)

# exctract a bunch of texts following the Medline format
def extract_txt(fname):
    t0 = time.time()
    print("Exctracting from '%s'..."%(fname.split('/')[-1].split('.')[0]))
    sys.stdout.flush()
    s=codecs.open(fname, 'r','utf-8')
    txt = s.read()
    # Use the properties of the Medline abstracts to separate documents
    # Specific to the corpus
    abstracts = re.split(r'PMID.*\n{2,}', txt)
    del abstracts[-1]
    s.close()

    duration = time.time()-t0
    print("%d documents exctracted - %s  [%s/s]" % (len(abstracts), sizeof_h(sys.getsizeof(abstracts)),
                                         sizeof_h(sys.getsizeof(abstracts) / duration)))
    print("Done. [%.02fs]"%(time.time()-t0))

    return abstracts

# simple regex to exctract the labels
def isolate_labels(subset):
    find = r'(AIM:|AIMS:|[A-Z][-A-Z /\\]{4,}:)'
    labels = re.findall(find,subset,re.MULTILINE)
    return labels

# split the text into abstracts, checking each structure
# dirty, but efficient for this task
def sub_get_abstracts(subset):
    out=[]
    cmpt=[]
    find = r'.+?(?=\n\n)'
    for idx,pick in enumerate(subset):
        labels = re.finditer(find,pick,re.MULTILINE)
        classes=[]
        while True:
            try:
                res = labels.next()
                classes.append([
                        res.string[res.start():res.end()],
                        res.start(),
                        res.end()
                    ])
            except StopIteration:
                break

        leng = len(classes)
        cmpt.append(leng)
        try:
            if leng<5:
                out.append(pick[classes[2][2]+2:classes[3][2]])
                continue
            labels = isolate_labels(pick[classes[3][2]+2:classes[4][2]])
            if len(labels)==0:
                labels = isolate_labels(pick[classes[2][2]+2:classes[3][2]])
            if len(labels)==0:
                labels = isolate_labels(pick[classes[4][2]+2:classes[5][2]])
                if len(labels)==0:
                    labels = isolate_labels(pick[classes[5][2]+2:classes[6][2]])
                    out.append(pick[classes[5][2]+2:classes[6][2]])
                else:
                    out.append(pick[classes[4][2]+2:classes[5][2]])
            else:
                out.append(pick[classes[3][2]+2:classes[4][2]])
        except:
            out.append(pick[classes[2][2]+2:classes[3][2]])
    return out

def get_abstracts(subset,nb_core=multiprocessing.cpu_count()):
    print("Working on %d core..."%(nb_core))
    sys.stdout.flush()
    t0 = time.time()
    out=[]
    d={}
    results=[]
    pool = multiprocessing.Pool(nb_core)

    for x in range(0,nb_core):
        d["tab{0}".format(x)]=[]

    for idx,i in enumerate(subset):
        d['tab%d'%(idx%nb_core)].append(i)

    for x in range(0,nb_core):
        ress = pool.apply_async(sub_get_abstracts,[d['tab%d'%(x%nb_core)]])
        results.append(ress)

    pool.close()
    pool.join()

    for i in results:
        out.extend(i.get())

    print("%s/s on each of the [%d] core" % (sizeof_h((sys.getsizeof(subset)/nb_core)/(time.time()-t0)),nb_core))
    print("Done. [%.02fs]"%(time.time()-t0))
    return out

# filter a set of abstract and verify the number of labeled sections
def get_errors(subset):
    cpt=0
    cpt2=0
    list_at_zero=[]
    list_at_one=[]
    list_at_two=[]
    for idx,i in enumerate(subset):
        try:
            iso = isolate_labels(i)
            if len(iso)<3:
                cpt+=1
                if len(iso)==0:
                    cpt2+=1
                    list_at_zero.append(idx)
                elif len(iso)<2:
                    list_at_one.append(idx)
                else:
                    list_at_two.append(idx)
        except:
            list_at_zero.append(idx)
    l1,l2,l3 = len(list_at_zero),len(list_at_one),len(list_at_two)
    return list_at_zero,list_at_one,list_at_two

# simple regex to remove numbers and replace them with "##NB"
def filter_numbers(subset):
    t0 = time.time()
    print("Filtering numbers...")
    sys.stdout.flush()
    find = r'[0-9]+\.[0-9]+|[0-9]+,[0-9]+|[0-9]+'
    for idx,i in enumerate(subset):
        subset[idx] = re.sub(find,r'##NB',i)
    print("Done. [%.02fs]"%(time.time()-t0))
    return subset

# train and save a sentence tokenizer to correctly parse sentences
def create_sentence_tokenizer(abstracts):
    try:
        t0 = time.time()
        print("Loading sentence tokenizer...")
        sys.stdout.flush()
        tokenizer = pickle.load(open("data/tokenizer.pickle", "rb"))
        print("Done. [%.02fs]"%(time.time()-t0))
        return tokenizer
    except (OSError, IOError) as e:
        print("No existing tokenizer, creating one...")
        sys.stdout.flush()
        try:
            y = int(raw_input("Number of articles for the parser training? [Max: %d]\n>"%(len(abstracts))))
            if y<=0 or y>(len(abstracts)):
                raise ValueError
        except ValueError as ve:
            y = len(abstracts)
            print("%s was not understood, using max [%d] instead."%y)
        print("Training tokenizer...")
        sys.stdout.flush()
        t0=time.time()
        tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
        text = "\n\n".join(abstracts[0:y])
        tokenizer.train(text)
        print("Trained. [%.02fs]"%(time.time()-t0))
        tools.dump_pickle(tokenizer,"data/tokenizer.pickle")
        return tokenizer

# exctract labels from an abstract and structure the sentences under it
def get_sentences_labels(subset,tokenizer):
    find = r'(AIM:|[A-Z][-A-Z /\\]{4,}:)'
    labels = re.finditer(find,subset,re.MULTILINE)
    classes=[]
    while True:
        try:
            res = labels.next()
            classes.append([
                    res.string[res.start():res.end()-1],
                    res.start(),
                    res.end()
                ])
        except StopIteration:
            break
    out = []
    leng = len(subset)
    if len(classes)<2:
        return
    for idx,lab in enumerate(classes):
        if idx == len(classes)-1:
            break
        out.append([
                lab[0],
                tokenizer.sentences_from_text(re.sub(r'\n',' ',subset[lab[2]+1:classes[idx+1][1]]))
                ])
    out.append([
            lab[0],
            tokenizer.sentences_from_text(re.sub(r'\n',' ',subset[classes[-1][2]+1:leng]))
            ])
    return out

# only used for multiprocessing in the next function
def interm(subset,tokenizer):
        res = []
        for i in subset:
            tmp = get_sentences_labels(i,tokenizer)
            res.append(tmp)
        return res

# exctract labels from all the abstracts
def ex_all_labels(subset,tokenizer,nb_core=multiprocessing.cpu_count()):
    print("Working on %d core..."%(nb_core))
    sys.stdout.flush()
    t0 = time.time()
    out=[]
    d={}
    results=[]
    pool = multiprocessing.Pool(nb_core)

    for x in range(0,nb_core):
        d["tab{0}".format(x)]=[]
        #d["tab%d"%x]=[]

    for idx,i in enumerate(subset):
        d['tab%d'%(idx%nb_core)].append(i)

    for x in range(0,nb_core):
        ress = pool.apply_async(interm,[d['tab%d'%(x%nb_core)],tokenizer])
        results.append(ress)

    pool.close()
    pool.join()

    for i in results:
        out.extend(i.get())

    print("%s/s on each of the [%d] core" % (sizeof_h((sys.getsizeof(subset)/nb_core)/(time.time()-t0)),nb_core))
    print("Done. [%.02fs]"%(time.time()-t0))
    return out
