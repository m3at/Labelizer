*Work in progress*

# Labelizer

This code build a [LSTM Neural Network](https://en.wikipedia.org/wiki/Long_short_term_memory) to classify scientific abstract into a set of labels.

(Example of Long Short Term Memory Neural Network usage: [link](http://karpathy.github.io/2015/05/21/rnn-effectiveness/))

## Requirements

*TODO: simple one line package installation*

This code is written in Python and require packages mostly included in distribution such as [Anaconda](http://continuum.io/downloads) or [Canopy](https://www.enthought.com/products/canopy/).

Additionally, you need to install:
`TextBlob` for fast lemmatization
`Keras` for neural network implementation (based on theano)
`Seaborn` for visualization.

You can get this packages through pip (prepend sudo if required):
```bash
pip install textblob
pip install textblob-aptagger
pip install keras
pip install seaborn
```

With anaconda, if you encounter `ImportError: No module named packages` while installing textblob-aptagger:

In `/home/username/anaconda/lib/python2.7/site-packages/textblob_aptagger/taggers.py` change line 10:
`from textblob.packages import nltk`
to
`import nltk`

## Usage

Try the notebook 'Complete_workflow_V2' for a step by step process.
Use the toy data included in the data folder for a quick experimentation>

*TODO*

## License

MIT
