# Labelizer  


This code build a [LSTM Neural Network](https://en.wikipedia.org/wiki/Long_short_term_memory) to classify sentences from a scientific abstract into a set of labels.



## Requirements

###### Tryed on MacOS X 10.10 and Ubuntu 15.04, work with Python 2.7 and Python 3+.  
---  

This code is written in Python and require packages mostly included in distribution such as [Anaconda](http://continuum.io/downloads) or [Canopy](https://www.enthought.com/products/canopy/).

Additionally, you need to install:  
`Spacy` for fast lemmatization  
`Keras` for neural network implementation (based on theano)  
`Seaborn` for visualization.  

You can get these packages through pip (prepend sudo if required):
```bash
pip install spacy
pip install keras
pip install seaborn
```



## Usage

Try the notebook [Labelizer_part1](Labelizer_part1.ipynb) for a step by step process of the data extraction, preprocessing and label analysis.  
Try the notebook [Labelizer_part2](Labelizer_part2.ipynb) for a LSTM training and evaluation.  

You can use the toy data included in the [data](data) folder for a quick experimentation, or use the downloading process included in the [Labelizer_part1](Labelizer_part1.ipynb) notebook to obtain bigger dataset.



## License

MIT
