# coding: utf-8

from __future__ import absolute_import
from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam

import pickle
import time
import sys
import spacy
import re
import argparse
from spacy.en import English
import numpy as np

# !!!
import warnings
warnings.filterwarnings("ignore")


s_epilog=""

parser = argparse.ArgumentParser(
	prog='labelizer.py',
	usage='%(prog)s [options] file [file ...]',
	epilog=s_epilog,
	formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
#parser.add_argument('bar', nargs='*', help='BAR!')
parser.add_argument('-q','--quiet',action="store_true", help='specify a name to save the command')
parser.add_argument('file', nargs='*', help='aone or multiple file(s) to labelise')
parser.add_argument('-v','--version', action='version', version='%(prog)s version 0.1, build 56231')
args = parser.parse_args()

#print(args.file)

#if args.quiet:
#    print("quiet")



# In[48]:
t0 = time.time()
print("Constructing neural network... ".ljust(40),end="")
sys.stdout.flush()

dim_out = 5
net = Sequential()
net.add(Embedding(194354, 16))
net.add(LSTM(16, 16))
net.add(Dense(16, dim_out))
net.add(Dropout(0.5))
net.add(Activation('softmax'))
net.compile(loss='categorical_crossentropy', optimizer='adam', class_mode="categorical")

print("Done. [%.02fs]"%(time.time()-t0))

# In[49]:
t0 = time.time()
print("Loading weights... ".ljust(40),end="")
sys.stdout.flush()

net.load_weights("weight_script.hdf5")

print("Done. [%.02fs]"%(time.time()-t0))


# In[51]:
classes_names = ['BACKGROUND','OBJECTIVE','METHOD','RESULT','CONCLUSION']

# In[30]:

#find = r'([^a-zA-Z0-9 .(),#\n:-]+)'
#print(re.sub(find,'\x1b[41m \x1b[0m',res))

# ---
t0 = time.time()
print("Loading vectorizer and tokenizer... ".ljust(40),end="")
sys.stdout.flush()

f = open("vectorizer_tokenizer.pkl",'rb')
vectorizer,tokenizer = pickle.load(f)
f.close()

print("Done. [%.02fs]"%(time.time()-t0))

# In[142]:

class LSTMPredictor:
    def __init__(self,network,vectorizer,classes_names,tokenizer,nb_index=100,len_graph=40):
        self.tagger = English()
        self.classes_names = classes_names
        self.vect = vectorizer
        self.network = network
        self.nb_index = nb_index
        self.tokenizer = tokenizer
        self.len_graph = len_graph

    def _vect_label(self,n):
        tmp = np.zeros(len(self.classes_names))
        tmp[n] = 1
        return tmp

    def _draw_pred(self,predicted,proba):
        m_len = np.array([len(i) for i in self.classes_names]).max()
        for idx,i in enumerate(self.classes_names):
            s = i.ljust(m_len+1)
            p = proba[idx]
            pad = '*'*int(round(p*self.len_graph))
            if len(pad)>(self.len_graph*2):
                pad='*'*(self.len_graph*2)
            if self.classes_names[idx] == predicted:
                print("\x1b[1;32m%s\x1b[0m|%.03f|\x1b[32m%s\x1b[0m [confidence]"%(s,p,pad))
            else:
                print("%s|\x1b[37m%.03f\x1b[0m|\x1b[37m%s\x1b[0m"%(s,p,pad))

    def silence_get_prediction(self,sent):
        doc = self.tagger(sent.decode())
        ph_out=[]

        find = r'[0-9]+\.[0-9]+|[0-9]+,[0-9]+|[0-9]+'
        res = re.sub(find,r'##NB',' '.join([i.lemma_.lower() for i in doc if not i.is_punct and not i.is_space]))
        ph_out=res.split()
        l = [sent[0:7].lower(),sent[8:15].lower(),sent[16:23].lower()]
        l.extend(ph_out)
        x = []
        for i in l:
            try:
                x.append(self.vect.vocabulary_[i])
            except KeyError:
                pass
        x = np.array([x])

        tmp=self.network.predict_classes(x, batch_size=100,verbose=False)[0]
        tmp2=self.network.predict_proba(x, batch_size=100,verbose=False)[0]
        #am= np.array(tmp2).argmax()
        #print(tmp,tmp2)
        self._draw_pred(self.classes_names[tmp],tmp2)
        return self.classes_names[tmp]

    def silence_predict(self,txt):
        cc=0.0
        sentences = tokenizer.tokenize(txt)
        cpt=len(sentences)
        out = []
        for idx,i in enumerate(sentences):
            tmp = "##LEN%02d ##POS%02d ##IDX%02d %s"%(cpt,cc,int((cc/cpt)*self.nb_index),i[1])
            cc+=1
            print("\x1b[1;35m%d|\x1b[0m %s\n"%(idx+1,i))
            predd = self.silence_get_prediction(tmp)
            out.append(predd)
            print('_'*80)
            print()
        return out


# In[143]:
t0 = time.time()
print("Initialising predictor... ".ljust(40),end="")
sys.stdout.flush()

pred = LSTMPredictor(net,vectorizer,classes_names,tokenizer)

print("Done. [%.02fs]"%(time.time()-t0))
# In[149]:

print("Starting to labelize.")
print("_"*80)
print()
sys.stdout.flush()


# In[146]:

for a in args.file:
    f = open(a,'rb')
    txt = f.read()
    find = r'[\t\r\n\f]+'
    txt = re.sub(find,' ',txt)

    f.close()

    out = pred.silence_predict(txt)

    # In[132]:

    for idx,i in enumerate(out):
        print("\x1b[1;35m%d|\x1b[0m %s"%(idx+1,i))
    print()



# Simple interractive mode: enter a new file name to parse
