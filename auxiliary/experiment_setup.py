# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 18:04:10 2020

@author: mauro
"""

import configparser
import os
import sys
import json
import warnings

from preprocessing import create_folder, rename_models

# load main path from config file
config = configparser.ConfigParser()
config.read("config.ini")
pathMain = config["paths"]["main"]


def select_experiment(experiment_name):
    """
    define and select the corpus experiments to be conducted
    """

    num_corpus = 25
    single = True
    pathSave = create_folder(os.path.join(pathMain, experiment_name))

    if experiment_name == "final":

        num_corpus = 1
        single = False
        word_list_csv = (
            "relevant_words_v7_bigram.csv"  #'relevant_words_v6_20k_per_corpus.csv'
        )

        rename_models(os.path.join(pathMain, "de_StR_r", "Models"), "de", "de5")
        rename_models(os.path.join(pathMain, "de_Zivil_r", "Models"), "de", "de6")
        rename_models(
            os.path.join(pathMain, "en_supreme_court_r", "Models"), "en", "en5"
        )
        rename_models(os.path.join(pathMain, "de_BGH_r", "Models"), "de", "de7")
        rename_models(os.path.join(pathMain, "de_en", "Models"), "de", "de8")
        rename_models(os.path.join(pathMain, "de_en", "Models"), "en", "en8")

        rename_models(os.path.join(pathMain, "de_StR_shuffled", "Models"), "de", "de15")
        rename_models(
            os.path.join(pathMain, "de_Zivil_shuffled", "Models"), "de", "de16"
        )
        rename_models(
            os.path.join(pathMain, "en_supreme_court_shuffled", "Models"), "en", "en15"
        )
        rename_models(os.path.join(pathMain, "de_BGH_shuffled", "Models"), "de", "de17")
        rename_models(os.path.join(pathMain, "de_en_shuffled", "Models"), "de", "de18")
        rename_models(os.path.join(pathMain, "de_en_shuffled", "Models"), "en", "en18")

        pathFolders = [
            (os.path.join(pathMain, r"de_StR_r"), ["de5"]),
            (os.path.join(pathMain, r"de_Zivil_r"), ["de6"]),
            (os.path.join(pathMain, r"en_supreme_court_r"), ["en5"]),
            (os.path.join(pathMain, r"de_BGH_r"), ["de7"]),
            (os.path.join(pathMain, r"de_en"), ["de8"]),
            (os.path.join(pathMain, r"de_en"), ["en8"]),
            (os.path.join(pathMain, r"de_StR_shuffled"), ["de15"]),
            (os.path.join(pathMain, r"de_Zivil_shuffled"), ["de16"]),
            (os.path.join(pathMain, r"en_supreme_court_shuffled"), ["en15"]),
            (os.path.join(pathMain, r"de_BGH_shuffled"), ["de17"]),
            (os.path.join(pathMain, r"de_en_shuffled"), ["de18"]),
            (os.path.join(pathMain, r"de_en_shuffled"), ["en18"]),
        ]

    elif experiment_name == "skip_final":

        num_corpus = 5
        single = False
        word_list_csv = (
            "relevant_words_v7_bigram.csv"  #'relevant_words_v6_20k_per_corpus.csv'
        )

        rename_models(
            os.path.join(pathMain, "en_sc_sentence_level_skip", "Models"), "en", "en4"
        )
        rename_models(os.path.join(pathMain, "de_en_skip", "Models"), "en", "en2")
        rename_models(os.path.join(pathMain, "de_BGH_v4_skip", "Models"), "de", "de4")
        rename_models(os.path.join(pathMain, "de_en_skip", "Models"), "de", "de2")

        pathFolders = [
            (os.path.join(pathMain, "en_sc_sentence_level_skip"), ["en4"]),
            (os.path.join(pathMain, "de_en_skip"), ["en2"]),
            (os.path.join(pathMain, "de_BGH_v4_skip"), ["de4"]),
            (os.path.join(pathMain, "de_en_skip"), ["de2"]),
        ]

    else:
        warnings.warn("analysis_type not found!")
        sys.exit()

    # remove legend
    legend = {}
    for path, language in pathFolders:
        legend[language[0]] = os.path.split(path)[1]

    # save the hyperparameter values
    with open(os.path.join(pathSave, "legend.json"), mode="w", encoding="utf-8") as f:
        json.dump(legend, f, ensure_ascii=False, indent=4)

    return pathFolders, pathSave, word_list_csv, num_corpus, single
