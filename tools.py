#!/usr/bin/python
# -*- coding: utf-8 -*-

# add here func, to optimize in Cython

# futur compatibility

from __future__ import absolute_import
from __future__ import print_function

import pickle, zipfile
import os.path, glob
import sys
from time import time
import codecs
from sklearn.externals import joblib





def load_pickle(s):
    t0=time()
    print("Loading '%s'..."%(s))
    sys.stdout.flush()
    try:
        f = open(s,'rb')
        returnObject = pickle.load(f)
        f.close()
    except (OSError, IOError) as e:
            print("'%s' not found, trying '%s.pickle'"%(s,s))
            sys.stdout.flush()
            f = open(s+'.pickle','rb')
            returnObject = pickle.load(f)
            f.close()
    print("Done. [%.02fs]"%(time()-t0))
    return returnObject

def dump_pickle(dumpObject,s,check_name=True):
    if os.path.isfile(s) or os.path.isfile(s+'.pickle'):
        if check_name:
            answ = str(raw_input("File already exist. Overwrite? [Y/N]\n>"))
            if answ == 'N' or answ == 'n' or answ == 'NO' or answ == 'no' or answ == 'No':
                answ = str(raw_input("Please enter a file name (.pickle will be added)\n>"))
                dump_pickle(dumpObject,answ,check_name=check_name)
    t0=time()
    print("Dumping...")
    sys.stdout.flush()
    if s.endswith('.pickle'):
        f = open(s,'wb')
        pickle.dump(dumpObject,f, protocol=2)
        f.close()
    else:
        f = open(s+'.pickle','wb')
        pickle.dump(dumpObject,f, protocol=2)
        f.close()
    print("Done. [%.02fs]"%(time()-t0))

def sizeof_h(num, suffix='B'):
    for unit in ['','K','M','G','T']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    while num>10:
        num/=10
        cpt+=1
    return "%de%d" % (num,cpt)


def unzip(source_filename, dest_dir):
    with zipfile.ZipFile(source_filename) as zf:
        for member in zf.infolist():
            words = member.filename.split('/')
            path = dest_dir
            for word in words[:-1]:
                drive, word = os.path.splitdrive(word)
                head, word = os.path.split(word)
                if word in (os.curdir, os.pardir, ''): continue
                #path = os.path.join(path, word)
            zf.extract(member, path)

def exctract_models(path="models"):
    x = os.listdir(path)
    model_list = []
    for i in x:
        if not i.startswith('.') and not i.startswith('_') and i.endswith('.zip'):
            if os.path.exists(os.path.join(path, i[:-4])):
                print("Model ready: %s"%(i[:-4]))
                sys.stdout.flush()
            else:
                print("Exctracting '%s'..."%(i))
                sys.stdout.flush()
                t0=time()
                unzip(os.path.join(path, i),path)
                print("Done. [%.02fs]"%(time()-t0))
                sys.stdout.flush()
            model_list.append(i[:-4])
    return model_list

def load_models(path="models",models={}):
    x = os.listdir(path)
    models = models
    for i in x:
        try:
            if not i.startswith('.') and not i.startswith('_') and os.path.isdir(os.path.join(path, i)):
                way = os.path.join(path, i)
                clf = glob.glob(os.path.join(way,"clf_*.pkl"))
                vec = glob.glob(os.path.join(way,"vectorizer_*.pkl"))
                print(". %s"%(way))
                if len(clf)!=1 or len(vec)!=1:
                    print("└── No model found in '%s'. Skipped."%(i))
                    continue
                t0=time()
                sys.stdout.flush()
                print("├── Loading classifier '%s'..."%(i))
                sys.stdout.flush()
                if "clf_%s"%(i) not in models:
                    models["clf_%s"%(i)] = joblib.load(clf[0])
                    print("├── Done. [%.02fs]"%(time()-t0))
                    sys.stdout.flush()
                t0=time()
                print("├── Loading vectorizer '%s'..."%(i))
                sys.stdout.flush()
                if "vectorizer_%s"%(i) not in models:
                    models["vectorizer_%s"%(i)] = joblib.load(vec[0])
                    print("└── Done. [%.02fs]"%(time()-t0))
                    sys.stdout.flush()
                t0=time()
        except:
            print(">> Error on '%s', skipped."%(i))
    return models

def show_colors(dic,justification=16):
    d = {}
    _color=[1,2,3,5,6,7]
    for idx,i in enumerate(dic):
        d[i]=_color[idx%len(_color)]
    jus=justification
    for k in dic.keys():
        print("\x1b[4%dm%s\x1b[0m"%(d[k],k.ljust(jus)) , end="")
    print()
    for k in dic.keys():
        print("%s"%(k.ljust(jus)) , end="")

"""
def show_colors(justification=16):
    jus=justification
    print(("\x1b[4%dm%s\x1b[0m\x1b[4%dm%s\x1b[0m\x1b[4%dm%s\x1b[0m\x1b[4%dm%s\x1b[0m"%(5,'BACKGROUND'.ljust(jus),2,'METHOD'.ljust(jus),3,'RESULT'.ljust(jus),6,'CONCLUSION'.ljust(jus))))
    print(("%s%s%s%s"%('BACKGROUND'.ljust(jus),'METHOD'.ljust(jus),'RESULT'.ljust(jus),'CONCLUSION'.ljust(jus))))


def timed(f):
  @wraps(f)
  def wrapper(*args, **kwds):
    start = time()
    result = f(*args, **kwds)
    print("Time: %.04fs" % (time() - start))
    return result
  return wrapper
"""
