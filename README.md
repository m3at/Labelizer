*Work in progress*

# Labelizer

This code build a [LSTM Neural Network](https://en.wikipedia.org/wiki/Long_short_term_memory) to classify sentences in a scientific abstract into a set of labels.

_([Example](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) of Long Short Term Memory neural network usage)_


## Requirements
######Tryed on MacOS X 10.10 and Ubuntu 15.04, work with Python 2.7 and Python 3+.  

This code is written in Python and require packages mostly included in distribution such as [Anaconda](http://continuum.io/downloads) or [Canopy](https://www.enthought.com/products/canopy/).

Additionally, you need to install:  
`TextBlob` for lemmatization (_getting rid of that soon!_)  
`Spacy` for fast lemmatization  
`Keras` for neural network implementation (based on theano)  
`Seaborn` for visualization.  

You can get these packages through pip (prepend sudo if required):
```bash
pip install textblob
pip install textblob-aptagger
pip install spacy
pip install keras
pip install seaborn
```

With anaconda, if you encounter `ImportError: No module named packages` while installing textblob-aptagger:  
In `/home/username/anaconda/lib/python2.7/site-packages/textblob_aptagger/taggers.py` change line 10:  
`from textblob.packages import nltk`  
to  
`import nltk`



## Usage

Try the notebook [Labelizer_part1](Labelizer_part1) for a step by step process of the data extraction, preprocessing and label analysis.  
Try the notebook [Labelizer_part2](Labelizer_part2) for a LSTM training and evaluation.  

You can use the toy data included in the [data](data) folder for a quick experimentation, or use the downloading process included in the [Labelizer_part1](Labelizer_part1) notebook to obtain bigger dataset.



## License

MIT
