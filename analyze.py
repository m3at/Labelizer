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
import copy

import time

# create a dictionary of labels
def create_dic_simple(subset_o):
    t0 = time.time()
    print("Copying corpus...", end="")
    sys.stdout.flush()
    subset = copy.deepcopy(subset_o)
    print("Done. [%.02fs]"%(time.time()-t0))
    sys.stdout.flush()
    t0 = time.time()
    print("Creating dictionary of labels...")
    sys.stdout.flush()
    dic = {}
    def add_dic(s,t):
        try:
            dic[s][0] += 1
            dic[s][1] += t
        except KeyError:
            dic[s] = [1,t]
    for idx, i in enumerate(subset):
        for j in i:
            add_dic(j[0],j[1])
    print("Done. [%.02fs]"%(time.time()-t0))
    return dic

def show_keys(dic,threshold=5000):
    w = []
    for i in dic.keys():
        w.append([i,dic[i][0]])
    w.sort(key=lambda x: x[1],reverse=True)
    for idx,i in enumerate(w):
        if i[1]>=threshold:
            print(str(i[1]).ljust(9,"_")+i[0])
        else:
            print("...\n(%d other labels with less than %d occurences)\n..."%(len(w)-idx,threshold))
            break

def regroup_keys(dic,primary_keyword):
    cpt=0
    for j in primary_keyword:
        for i in dic.keys():
            if i != j and (str(i).find(j) != -1):
                try:
                    cpt+=1
                    dic[j][0]+=dic[i][0]
                    dic[j][1]+=dic[i][1]
                    del dic[i]
                except KeyError:
                    dic[j]=[dic[i][0],dic[i][1]]
                    del dic[i]
                    pass
    print("Keys regrouped:",cpt)

def replace_keys(dic, keys_to_replace, replace_with):
    cpt=0
    for idx,jj in enumerate(keys_to_replace):
        for j in jj:
            try:
                dic[replace_with[idx]][0]+=dic[j][0]
                dic[replace_with[idx]][1]+=dic[j][1]
                del dic[j]
                cpt+=1
            except KeyError:
                try:
                    dic[replace_with[idx]]=[dic[j][0],dic[j][1]]
                    del dic[j]
                    cpt+=1
                except KeyError:
                    pass
    print("Keys regplaced:",cpt)

def get_exactly(subset,
             pattern=[['BACKGROUND','BACKGROUNDS'],
                             ['METHOD','METHODS'],
                             ['RESULT','RESULTS'],
                             ['CONCLUSION','CONCLUSIONS']],
             no_truncate=False):
    t0 = time.time()
    print("Selecting abstracts...")
    sys.stdout.flush()
    sub_perfect=[]
    for i in subset:
        match=True
        try:
            for idx,j in enumerate(pattern):
                if idx>=len(i):
                    break
                if not i[idx][0] in j:
                    match=False
                    break
            if match:
                if not no_truncate:
                    sub_perfect.append(i[:len(pattern)])
                else:
                    if len(i)==len(pattern):
                        sub_perfect.append(i[:len(pattern)])
        except:
            pass
    cpt=float(len(sub_perfect))
    print("%d/%d match the pattern (%d%%)" %(cpt,len(subset),int((cpt/len(subset))*100)))
    print("Done. [%.02fs]"%(time.time()-t0))
    return sub_perfect
