# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 21:26:29 2019

@author: mauro
"""
import os
import sys
import csv
import re
from time import time
import itertools

import spacy
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import numpy as np
from gensim.models.phrases import Phrases, Phraser

# Count the number of cores in a computer
np.random.seed(0)


def create_folder(path):
    """
    create folder, if it doesn't already exist
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def read_sentences(folder_name, filename):
    """
    load the data and remove whitespace characters
    like `\n` at the end of each line
    """
    with open(os.path.join(folder_name, filename), encoding="utf8") as f:
        sentences = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    return [x.strip() for x in sentences]


def bootstrap(
    num_corpus, num_sentences, sampling_size, pathPlots, bootstrapping=True, plot=True
):
    """
    create a list of bootstrapped sentence indexes
    """

    if sampling_size == None:
        sampling_size = num_sentences

    if bootstrapping:
        index_list = []

        for _ in range(num_corpus):
            # range of sentences to pick from
            index = np.arange(0, num_sentences)
            # number of indexes to pick
            index_list.append(np.random.choice(index, size=sampling_size, replace=True))

    else:
        # add non bootstrapping option,
        index_list = [range(num_sentences)]

    if plot:

        fig = plt.figure(figsize=(6, 6))
        plt.title("sanity check for bootstrap index")
        for count, idx_list in enumerate(index_list):
            plt.plot(sorted(idx_list), label=str(count))
        plt.legend(loc="lower right")
        plt.show()
        plt.tight_layout()
        fig.savefig(os.path.join(pathPlots, "bootstrap_type_index_sanity_check.png"))

    return index_list


def get_spacy_model(language):
    """
    load the spacy model dependent on the language
    """
    # load the spacy for lemmatization and stopword removal
    if language == "en":
        spacy_model = "en_core_web_sm"
    elif language == "de":
        spacy_model = "de_core_news_sm"
    elif language == "fr":
        spacy_model = "fr_core_news_sm"
    else:
        print("languagecouldnt be found: ", language)
        sys.exit()
    nlp = spacy.load(
        spacy_model, disable=["ner", "parser"]
    )  # disabling Named Entity Recognition for speed
    print("Load Spacy Model: {}".format(spacy_model))
    return nlp


def text_cleaning(doc, lemmatize=True):
    """
    lemmatize and remove stopwords
    """
    if lemmatize:
        txt = [token.lemma_ for token in doc if not token.is_stop]
    else:
        txt = [token.text for token in doc if not token.is_stop]
    return re.sub(" +", " ", " ".join(txt)).split()


def sentence_preprocessing(sentences, nlp, lowercase=True, lemmatize=True):
    """
    sentence cleaning:
    - remove punctuation
    - remove stop words
    - lowercase (opt.)
    - lemmatize (opt.)
    """
    if lowercase:
        remove_non_characters = (
            re.sub("[^a-zA-Z äöüßÄÖÜôÔèÈéÉàÀÇéâêîôûàèùëïü]+", " ", row.lower())
            for row in sentences
        )
    else:
        remove_non_characters = (
            re.sub("[^a-zA-Z äöüßÄÖÜôÔèÈéÉàÀÇéâêîôûàèùëïü]+", " ", row)
            for row in sentences
        )
    sent = [
        text_cleaning(doc, lemmatize)
        for doc in nlp.pipe(remove_non_characters, n_threads=-1, batch_size=1000)
    ]
    return sent


def plot_word_frequency(sent, pathFolder, filename):
    """
    plot the word frequency
    """
    fig = plt.figure(figsize=(10, 4))
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.title("FreqDist")

    fdist1 = FreqDist(list(itertools.chain.from_iterable(sent)))
    fdist1.plot(30, cumulative=False)
    fig.savefig(os.path.join(pathFolder, filename + ".png"), bbox_inches="tight")

    # save fdist to csv
    with open(
        os.path.join(pathFolder, filename + "_fdist.csv"),
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as fp:
        writer = csv.writer(fp, delimiter=";")
        writer.writerows(fdist1.most_common())
    return fdist1


def get_bigram_and_trigram_model(sentences, n_gram, language, pathSave):
    """
    create bigram and trigram models
    """
    t = time()
    # Train a bigram model.
    bigram = Phrases(
        sentences,
        min_count=n_gram["min_count"],
        threshold=n_gram["threshold_bigram"],
        progress_per=100000,
    )

    # Train a trigram model.
    trigram = Phrases(
        bigram[sentences],
        n_gram["min_count"],
        threshold=n_gram["threshold_trigram"],
        progress_per=100000,
    )
    print(
        "Time to create bigram and trigram models: {} mins".format(
            round((time() - t) / 60, 2)
        )
    )
    return bigram, trigram


def rename_models(folder_name, language="en", language_new="en2", num_corpus=25):
    """
    rename the english word2vec models for parallel word comparison
    (several language pairs) in the gensim_evaluation.py
    """

    new_names = []
    names = []
    for count in range(num_corpus):
        model_name = "{}_{}_full_model.model".format(language, count)
        model_name_new = "{}_{}_full_model.model".format(language_new, count)

        vec_name = "{}.wv.vectors.npy".format(model_name)
        vec_name_new = "{}.wv.vectors.npy".format(model_name_new)

        train_name = "{}.trainables.syn1.npy".format(model_name)
        train_name_new = "{}.trainables.syn1.npy".format(model_name_new)

        new_names.extend([model_name_new, vec_name_new, train_name_new])
        names.extend([model_name, vec_name, train_name])

    for new_name, name in zip(new_names, names):
        if new_name in os.listdir(folder_name):
            continue

        if name in os.listdir(folder_name):
            from_name = os.path.join(folder_name, name)
            to_name = os.path.join(folder_name, new_name)
            os.rename(from_name, to_name)
