# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 14:21:02 2019

@author: mauro
"""

import os
import sys
import pickle
import configparser
from time import time

from scipy.stats import entropy
from gensim.models import Word2Vec
from ModelTraining import callback
import numpy as np
import pandas as pd

from utils.preprocessing import create_folder
from utils.experiment_setup import select_experiment
from utils.log import initalize_logger


def get_list_of_relevant_words(df_words, language, single=True):
    """
    based on dataframe return list of words
    """
    # get the list of relevant words per language
    if "en" in language and single:
        relevant_words = [
            word.lower() for word in df_words["en"].tolist() if not pd.isnull(word)
        ]
    elif "de" in language and single:
        relevant_words = [
            word.lower() for word in df_words["de"].tolist() if not pd.isnull(word)
        ]
    elif "fr" in language and single:
        relevant_words = [
            word.lower() for word in df_words["fr"].tolist() if not pd.isnull(word)
        ]

    elif not single:
        relevant_words = [
            word.lower() for word in df_words[language].tolist() if not pd.isnull(word)
        ]
    else:
        print("language not found: {}".format(language))
        sys.exit()
    return relevant_words


def predict_context_word_proba(w2v_model, w, gensim_proba_calc):
    """
    calculate entropy based on word conetext probabilities
    """

    if gensim_proba_calc:
        scores = w2v_model.predict_output_word([w], topn=None)
        # exclude the word itself, if in the word list
        prob_values = [a[1] for a in scores]  # if a[0] != w

    else:
        vec = w2v_model.wv.get_vector(w)
        # hierarchical softmax
        if w2v_model.hs == 1:
            prob_values = np.exp(np.dot(vec, w2v_model.trainables.syn1.T))

        # negativs sampling
        elif w2v_model.hs == 0 and w2v_model.negative == 0:
            prob_values = np.exp(np.dot(vec, w2v_model.trainables.syn1neg.T))
        else:
            print("probabiltiy values could not be calculated")
            sys.exit()
        prob_values /= sum(prob_values)

    return prob_values


# todo needs to be updated
class EntropyEvaluation(object):
    def __init__(self):
        pass


if __name__ == "__main__":

    # load main path from config file
    config = configparser.ConfigParser()
    config.read("config.ini")
    pathMain = config["paths"]["main"]

    gensim_proba_calc = False
    base_entropy = 2

    experiment_name = "legal_code"
    pathFolders, pathSave, word_list_csv, num_corpus, single = select_experiment(
        experiment_name
    )

    # define paths
    # pathStatistics = create_folder(os.path.join(pathSave, 'Statistics'))
    pathLogs = create_folder(os.path.join(pathSave, "Logs"))
    pathPlot = create_folder(os.path.join(pathSave, "Plots"))

    if word_list_csv is not None:
        use_relevant_words = True
        # load the word list
        df_words = pd.read_csv(
            os.path.join(r"relevant_words", word_list_csv),
            encoding="utf-8",  # "utf-8",
            delimiter=";",
            dtype=str,
        )
    else:
        use_relevant_words = False

    entropy_distribution = {}
    word_distribution = {}

    # loop over the different folders containing the models
    for pathFolder, languages in pathFolders:
        print("\nfolder_name: {}\n-- languages: {}\n".format(pathFolder, languages))
        pathModels = os.path.join(pathFolder, "Models")
        pathPlots = create_folder(os.path.join(pathFolder, "Plots"))

        name_of_folder = os.path.split(pathFolder)[-1]

        # loop over the different languages per folder
        # for language in languages:
        language = languages[0]

        # initalize the logger, per language
        logger = initalize_logger(language, pathLogs, "evaluation")

        # calculate the word ambiguity
        ambiguity = {}
        for count in range(num_corpus):

            # load model method
            model_name = "{}_{}_full_model.model".format(language, count)
            if model_name not in os.listdir(pathModels):
                print("Model not found!: {}".format(model_name))
                continue

            # load the models
            w2v_model = Word2Vec.load(os.path.join(pathModels, model_name))
            print("{} loaded".format(model_name))

            # get word list
            if use_relevant_words:
                # get the list of relevant words based on dataframe
                word_list = get_list_of_relevant_words(df_words, language, single)
            else:
                word_list = w2v_model.wv.vocab
                print("use_relevant_words: {}".format(use_relevant_words))

            print("gensim_proba_calc: {}".format(gensim_proba_calc))

            t = time()
            # get word entropy
            for _count, w in enumerate(word_list):
                if w not in ambiguity:
                    ambiguity[w] = []

                # calculate probabilities
                probabilities = predict_context_word_proba(
                    w2v_model, w, gensim_proba_calc
                )
                # calculate entropy
                ambiguity[w].append(
                    (
                        entropy(probabilities, base=base_entropy),
                        w2v_model.wv.vocab[w].count,
                    )
                )

            print(
                "Time to calculate entropy: {} mins\n".format(
                    round((time() - t) / 60, 2)
                )
            )

        entropy_distribution[name_of_folder] = ambiguity

    # save the ambigutiy distribution
    with open(os.path.join(pathSave, "entropy_distribution.pickle"), "wb") as fp:
        pickle.dump(entropy_distribution, fp)
