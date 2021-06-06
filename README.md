
![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)

# Entropy in Legal Language

 
 <!-- [![Total Downloads](https://poser.pugx.org/phpunit/phpunit/downloads)](//packagist.org/packages/phpunit/phpunit) -->
 <!-- [![Build Status](https://travis.ibm.com/Mauro-Luzzatto/-Data-Modelling.svg?token=zmafNzx54WQZmTrFgEaV&branch=master)](https://travis.ibm.com/Mauro-Luzzatto/-Data-Modelling) -->

<!-- 
e.g

Add paper name in the small github repo box 

Tragin, M. & Vaulot, D. 2019. Novel diversity within marine Mamiellophyceae (Chlorophyta) unveiled by metabarcoding. Sci. Rep. 9:5190. https://www.nature.com/articles/s41598-019-41680-6
 -->


<!-- Code for use in measuring the sophistication of political text

“Measuring and Explaining Political Sophistication Through Textual Complexity” by Kenneth Benoit, Kevin Munger, and Arthur Spirling. This package is built on quanteda.
 -->

This repository contains the code developed for the paper:

> ["Entropy in Legal Language"](/paper/Friedrich-Luzzatto-Ash-Legal-Entropy.pdf) by Roland Friedrich, Mauro Luzzatto, Elliott Ash (2020), Proceedings of the 2020 Natural Legal Language Processing (NLLP) Workshop, 24 August 2020

A novel method has been introduced to measure the word ambiguity, i.e. local word entropy, in the corpora, based on a word2vec model. 

The code has been developed to investigate the word ambiguity in the written text of opinions by the **U.S. Supreme Court (SCOTUS)** and the **German Bundesgerichtshof (BGH)**, which are representative courts of the **common-law** and **civil-law** court systems. 


<!-- Word embeddings are used to calculate the conditional probability of the context words of all corpus words, based on this distribution the entropy is calculated per word level and then aggregated to corpus level -->

<!-- We use the measure to investigate entropy in the written text of opinions published by the U.S. Supreme Court
(SCOTUS) and the German Bundesgerichtshof (BGH), representative courts of the common-law and civil-law
court systems respectively. -->

<!-- To account for language difference between German (BGH) and English (U.S. Supreme Court), the entropy of two non-legal corpus translations (EuroParl) have been calculated for comparison as well.  -->


# Getting Started 

## Installation
Download the github repository:
```bash
git clone https://github.com/MauroLuzzatto/legal-entropy
```
Run the `makefile` to install all python modules needed to run the code:
```bash
make init
```
---
Or install python requirements and spacy module manually:
```bash
pip install -r requirements.txt
```
```bash
python -m spacy download en_core_web_sm
```
```bash
python -m spacy download de_core_news_sm
```

## Run Code

After the installation run the code as follows:

1) Define the corpora to be processed in `corpus_setup.py`
2) Define the corpora to be evaluated in `experiment_setup.py`
3) run `TextPreprocessing.py`
4) run `ModelTraining.py`
5) run `EntropyEvaluation.py`
6) run `EntropyVisualization.py`


# Code Overview

The code is structured in five parts:

<!-- (make links) -->
1) [Experiment Setup](#1-experiment-setup)
2) [Text Preprocessing](#2-text-preprocessing)
3) [Model Training](#3-model-training)
4) [Entropy Calculation](#4-entropy-calculation)
5) [Entropy Visualization](#5-entropy-visualization)


## 1) Experiment Setup
In the experiment setup the relevant corpus are loaded and the type of experiment is defined.
- `corpus_setup.py`: defined the corpora that should be loaded and preprocessed
<!-- * `name_of_folder = 'EuroParl' # a folder with this name will be created`
* `filename = 'EuroParl_filename.csv'  # documents should be separated by row`
* `pathLoad = r'path\to\file'`
* `language = 'de'`
* `delimiter = ','  # used for loading the text`
* `column_name = 'content'` -->
- `experiment_setup.py`: define the experiments that should be conducted
<!-- [update] -->
- `config.ini`: define the main path, where the results should be saved

<!-- *  `main = path\to\folder` -->
<!-- *  test 1
*  {"threshold_bigram": 10, "threshold_trigram": 40, "min_count": 20}
 -->



<!-- * "epochs": 30
* "min_count": 20 
* "window": 2
* "size": 300 
* "workers": 3 
* "hs": 1 
* "sg": 1 
* "negative": 0 -->



## 2) Text Preprocessing
In a first step the text is preprocessed and cleaned. The corpus is split into a set of cleaned (e.g. lowercase, lemmatize) sentences. This also includes the creation of bigrams and trigrams using `gensim`.

- `TextPreprocessing.py`: main class for the text preprocessing and cleaning
- `preprocessing.py`: contains helper functions for the data preparation.
- `n_grams.json`: define the the threshold and min_count of words for the bigram and trigram creation


<!-- Bootstrapping:
* `create_index_list` = True   
* `bootstrapping` = False # bootstrapp the sentences    
* `num_corpus` = 1 # set the numer of corpora (relevant for bootstrapping)
* `sampling_size` = None  # downsample the number of sentences extracted from the corpus

Text cleaning:
* `lemmatize` = False
* `lowercase` = False    

Further options:
* `save` = True
* `plot` = True
* `load` = False # load sentences -->


## 3) Model Training
After the text preprocessing, the `word2vec` model is trained using a defined set of hyperparameter.
* `ModelTraining.py`: main class for word2vec model training, the hyperparameters are defined in the json file
* `hyperparemeters.json`: define the hyperparameters for the word2vec model training

<!-- Experiment:
* `experiment_name` = 'final_roland'

Futher options:
* `plot` = True
* `evaluate` = True
* `shuffle` = True
* `save` = True
     -->


<!--  -->

## 4) Entropy Calculation 
The trained `word2vec` models are used to calculate the conditional probability of each center words context words. Based on this probability distribution the entropy on word level is calculated (local word entropy). <!-- Next to the pre-build `
predict_output()` method, an adapted method based on `negative sampling` has been implemented -->

- `EntropyEvaluation.py`: main class used for the entropy calculation based on the predicted context word probability

<!--  -->

## 5) Entropy Visualization 
Finally, the calculated word entropies are visualized on a corpus level.

- `Visualization.py`: functions for visualizing the results


## Authors
* **Mauro Luzzatto** - [Maurol](https://github.com/MauroLuzzatto)

