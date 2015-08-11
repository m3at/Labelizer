# Labelizer

This code build a LSTM Neural Network to classify scientific abstract into a set of labels.

## Requirements

*TODO: reduce the amount of required packages and specify it per file*

This code is written in Python and require a consequent amount of additional package, mostly included in distribution such as [Anaconda](http://continuum.io/downloads) or [Canopy](https://www.enthought.com/products/canopy/).
Additionally, you need to install `TextBlob` and `Keras`, which you can get with pip (prepend sudo if required):
```bash
pip install textblob
pip install textblob-aptagger
pip install keras
```
 
If you encounter `ImportError: No module named packages` while installing textblob-aptagger with anaconda:
In `/home/username/anaconda/lib/python2.7/site-packages/textblob_aptagger/taggers.py` change line 10:
`from textblob.packages import nltk`
to
`import nltk`

## Usage

TODO

## License

MIT
