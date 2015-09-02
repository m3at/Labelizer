#!/usr/bin/python
# -*- coding: utf-8 -*-

# add here func, to optimize in Cython

# futur compatibility

from __future__ import absolute_import
from __future__ import print_function

from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import *

from textblob import Blobber
from textblob_aptagger import PerceptronTagger
from textblob import Word

import seaborn.apionly as sns

from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt

import numpy as np
import sys
import codecs
import re
import multiprocessing
import pickle
import copy
import time
import random
import os


# ensure each sequence has the same length
def pad_sequence(X_train,X_test,y_train,y_test,maxlen=40):
    t0 = time.time()
    print("Pading sequences...")
    sys.stdout.flush()
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print("Done. [%.02fs]"%(time.time()-t0))
    return X_train, X_test, y_train, y_test

class LossKeepIncreasing(Exception):
    pass

# train the neural network
def train_network(network,
                  X_train,y_train,X_test,y_test,
                  epoch,
                  batch_size=64,path_save="weights",patience=1):

    t0 = time.time()
    print("Training...")
    sys.stdout.flush()

    if not os.path.exists(path_save):
        os.makedirs(path_save)


    def _local_evaluate(n_plain_t,n_plain_p):
        c = 0
        for idx,i in enumerate(n_plain_p):
            isit=False
            for idx2,x in enumerate(i):
                if x==1 and x==n_plain_t[idx][idx2]:
                    isit=True
            if isit:
                c+=1
        acc = float(c)/len(n_plain_p)
        rps = metrics.label_ranking_average_precision_score(n_plain_t,n_plain_p)
        print("\x1b[33mAccuracy: %.02f%%\x1b[0m [%d/%d], \x1b[33mRPS: %.03f\x1b[0m"%(acc*100,c,len(n_plain_p),
                                        rps),end="")
        return acc,rps

    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.accuracy = []
            self.val_losses = []
            self.val_accuracy = []
            self.metric = []
            self.file_name = []
            self.pos = 0

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            self.accuracy.append(logs.get('acc'))

        def on_epoch_end(self, epoch, logs={}):
            self.val_losses.append(logs.get('val_loss'))
            self.val_accuracy.append(logs.get('val_acc'))
            self.metric.append(self.val_accuracy[-1]-(self.val_losses[-1]/10.))
            if len(self.val_accuracy)<2:
                #network.save_weights("weights/weight_%d.hdf5"%self.pos,overwrite=True)
                print("\x1b[35mSaving at first epoch\x1b[0m")
            else:
                tmp = True
                for i in self.metric[:-1]:
                    if self.metric[-1]<i:
                        tmp = False
                        break
                if tmp:
                    network.save_weights("%s/best.hdf5"%path_save,overwrite=True)
                    print("\x1b[35mModel improved, saving weight\x1b[0m")
            network.save_weights("%s/epoch_%d.hdf5"%(path_save,self.pos),overwrite=True)

            self.pos=self.pos+1

            cpt = 0
            if len(self.val_accuracy)>1:
                for i in self.metric[-(patience+1):-1]:
                    if self.metric[-1]<i:
                        cpt += 1
            if patience == cpt:
                raise LossKeepIncreasing

    history = LossHistory()

    try:
        network.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epoch,
                    validation_data=(X_test, y_test), show_accuracy=True, callbacks=[history])#,earlystop])
    except LossKeepIncreasing:
        print("Accuracy keep decreasing, stopping early.")
    #network.load_weights("%s/best.hdf5"%path_save)
    print("Done. [%.02fs]"%(time.time()-t0))
    return history

# show the evolution of the loss and accuracy over time
def show_history(history):
    sns.set_style("darkgrid")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 1.6})
    sns.mpl.rc("figure", figsize=(16,5))
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(history.losses, color="#47d024")
    ax1.plot(history.accuracy, color="#0672b6")
    ax1.legend(["loss","accuracy"])
    ax1.set_title("Batch validation")

    ax2.plot(history.val_losses, color="#47d024")
    ax2.plot(history.val_accuracy, color="#0672b6")
    ax2.legend(["loss","accuracy"])
    ax2.set_title("Epoch validation")

    plt.show()


#
def evaluate_network(network, X_test, y_test, classes_names, length=1000, batch_size=64):
    resp = network.predict_proba(X_test[:length],batch_size=batch_size,verbose=False)
    resc = network.predict_classes(X_test[:length],batch_size=batch_size,verbose=False)

    a1=[]
    a2=[]
    cpt=0
    cpt_on=[]
    cpt_real=[]
    cpt_should=0
    should=[]
    cpt_shouldnt=0
    shouldnt=[]
    for idx, i in enumerate(resc):
        a1.append(i)
        a2.append(np.array(y_test[idx]).argmax())
        if i.tolist()==[0,0,0,0]:
            cpt+=1
            cpt_on.append(resp[idx].argmax())
            cpt_real.append(np.array(y_test[idx]).argmax())
            if cpt_on[-1]==cpt_real[-1]:
                cpt_should+=1
                should.append(resp[idx].argmax())
            else:
                cpt_shouldnt+=1
                shouldnt.append(resp[idx].argmax())
            #print(resp[idx])
    print("No decision: %d / %d  [%.02f%%]"%(cpt,len(resc),
                                          (cpt/float(len(resc)))*100 ) , end="")
    print(cpt_should,cpt_shouldnt)

    print("Accuracy: %.06f"%metrics.label_ranking_average_precision_score(y_test[:length],resp))

    cpt_on = np.array(cpt_on)
    print(metrics.classification_report(a1,a2,target_names=classes_names))

    print("Confusion matrix:")
    cm = confusion_matrix(a1, a2)
    print(cm)
    sns.set_style("ticks")
    sns.mpl.rc("figure", figsize=(8,4))

    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure()
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes_names))
    plt.xticks(tick_marks, classes_names, rotation=45)
    plt.yticks(tick_marks, classes_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tick_params(which='both', direction='in',length=0)
    plt.show()


# small utility
def get_colors(dic):
    d = {}
    _color=[1,2,3,5,6,7]
    for idx,i in enumerate(dic):
        d[i]=_color[idx%len(_color)]
    return d


class LSTMPredictor:

    def __init__(self,network,vectorizer,classes_names,nb_index=100,len_graph=20):
        self.tagger = Blobber(pos_tagger=PerceptronTagger())
        self.classes_names = classes_names
        self.vect = vectorizer
        self.network = network
        d = {}
        _color=[1,2,3,5,6,7]
        for idx,i in enumerate(classes_names):
            d[i]=_color[idx%len(_color)]
        self.colors = d
        self.nb_index = nb_index
        self.len_graph = len_graph
        Word("test")

    def _draw_pred(self,predicted,true_label,proba):
        m_len = np.array([len(i) for i in self.classes_names]).max()
        for idx,i in enumerate(self.classes_names):
            s = i.ljust(m_len+1)
            p = proba[idx]
            pad = '*'*int(round(p*self.len_graph))
            if self.classes_names[idx] == predicted == true_label:
                print("\x1b[42m%s\x1b[0m|\x1b[37m%.03f\x1b[0m|%s \x1b[32m[true label][predicted label]\x1b[0m"%(s,p,pad))
            elif self.classes_names[idx] == predicted:
                print("\x1b[45m%s\x1b[0m|\x1b[37m%.03f\x1b[0m|%s \x1b[35m[predicted label]\x1b[0m"%(s,p,pad))
            elif self.classes_names[idx] == true_label:
                print("\x1b[43m%s\x1b[0m|\x1b[37m%.03f\x1b[0m|%s \x1b[33m[true label]\x1b[0m"%(s,p,pad))
            else:
                print("%s|\x1b[37m%.03f\x1b[0m|\x1b[37m%s\x1b[0m"%(s,p,pad))

    def get_prediction(self,text,true_label):
        corr = self.classes_names

        sent=text
        tag = self.tagger(sent[24:].lower()).tags
        #print(tag)
        ph_out=[]
        for i in tag:
            if i[1][0]=='V':
                ph_out.append(Word(i[0]).lemmatize('v'))
            elif i[1][0]=='N':
                ph_out.append(Word(i[0]).lemmatize('n'))
            else:
                ph_out.append(i[0])
        res = " ".join(ph_out)

        t1 = [sent[0:7].lower(),sent[8:15].lower(),sent[16:23].lower()]
        t1.extend(ph_out)
        l = t1
        x = []
        for i in l:
            try:
                x.append(self.vect.vocabulary_[i])
            except KeyError:
                pass
        x = np.array([x])

        tmp=self.network.predict_classes(x, batch_size=32,verbose=False)[0]
        tmp2=self.network.predict_proba(x, batch_size=32,verbose=False)[0]
        am= np.array(tmp2).argmax()
        if tmp.tolist()==[0,0,0,0]:
            print("No decision")
        if np.count_nonzero(tmp2) < 3:
            print("More than 1:",tmp2)
        self._draw_pred(corr[am],true_label,tmp2)
        return np.array(tmp2).max(),corr[am]

    def predict_labeled(self,lab):
        cc=0
        for i in lab:
            cpt=0.0
            for x in lab:
                cpt+=len(x[1])
            tmp=[]
            for smt in i[1]:
                tmp.append("##LEN%02d ##POS%02d ##IDX%02d %s"%(cpt,cc,int((cc/cpt)*self.nb_index),smt))
                cc+=1
            for j in tmp:
                print("%s\n"%(j[24:]))
                confidence,predd = self.get_prediction(j,i[0])
                print('_'*80)
                print()

    def predict_on_csv(self,lab):
        cc=0.0
        cpt=len(lab)
        for i in lab:
            tmp = "##LEN%02d ##POS%02d ##IDX%02d %s"%(cpt,cc,int((cc/cpt)*100),i[1])
            cc+=1
            print("%s\n"%(tmp[24:]))
            #print("%s\n"%(j[24:]))
            confidence,predd = self.get_prediction(tmp,i[0][0])
            print('_'*80)
            print()

# local
def get_oth_lab(subset,tokenizer):
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
        print("Less than two classes. Returning void...")
        print([classes[0][0],re.sub(r'\n',' ',subset[classes[0][2]+1:])])
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
