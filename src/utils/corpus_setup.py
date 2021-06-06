# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 15:26:08 2020

@author: mauro
"""


def select_corpus(number):
    """
    define the different corpus that can be used in the analysis
    """
    # default
    name_of_folder = ""
    filename = ""
    pathLoad = ""
    language = "en"
    delimiter = ","
    column_name = "text"
    document_level = True

    if number == 0:
        name_of_folder = "de_BGH"
        filename = "BGH_df_2019-12-13.csv"
        pathLoad = r"data\open_legal_data"
        language = "de"
        delimiter = ","
        column_name = "content"

    elif number == 1:
        name_of_folder = "en_supreme_court_r_v2"
        filename = "Test_Judge.csv"
        pathLoad = r"C:\Users\mauro\Desktop\LawProject"
        language = "en"
        delimiter = "\t"
        column_name = "text"

    elif number == 4:
        name_of_folder = "de_StR_r"
        filename = "BGH_df_2019-12-13_strafrecht.csv"
        pathLoad = (
            r"C:\Users\mauro\OneDrive\Dokumente\Python_Scripts\LawProject\openlegaldata"
        )
        language = "de"
        delimiter = ","
        column_name = "content"

    elif number == 5:
        name_of_folder = "de_Zivil_r"
        filename = "BGH_df_2019-12-13_zivilrecht.csv"
        pathLoad = (
            r"C:\Users\mauro\OneDrive\Dokumente\Python_Scripts\LawProject\openlegaldata"
        )
        language = "de"
        delimiter = ","
        column_name = "content"

    elif number == 6:
        name_of_folder = "de_en"
        filename = "europarl-v7.de-en.de"
        pathLoad = r"C:\Users\mauro\Desktop\LawProject"
        language = "de"

    elif number == 7:
        name_of_folder = "de_en"
        filename = "europarl-v7.de-en.en"
        pathLoad = r"C:\Users\mauro\Desktop\LawProject"
        language = "en"

    elif number == 2:
        name_of_folder = "de_RCv2_skip"
        filename = "german_RCv2.csv"
        pathLoad = (
            r"C:\Users\mauro\OneDrive\Dokumente\Python_Scripts\LawProject\reuters"
        )
        language = "de"
        delimiter = ";"
        column_name = "text"

    elif number == 3:
        name_of_folder = "en_RCv1_skip"
        filename = "reuters_RCv1.csv"
        pathLoad = (
            r"C:\Users\mauro\OneDrive\Dokumente\Python_Scripts\LawProject\reuters"
        )
        language = "en"
        delimiter = ";"
        column_name = "text"

    elif number == 10:
        name_of_folder = "de_BGH_r"
        filename = "BGH_df_2019-12-13.csv"
        pathLoad = r"C:\Users\maurol\OneDrive\Dokumente\Python_Scripts\LawProject\openlegaldata"
        language = "de"
        delimiter = ","
        column_name = "content"

    corpus_info = {}
    corpus_info["name_of_folder"] = name_of_folder
    corpus_info["filename"] = filename
    corpus_info["pathLoad"] = pathLoad
    corpus_info["language"] = language
    corpus_info["delimiter"] = delimiter
    corpus_info["column_name"] = column_name
    corpus_info["document_level"] = document_level
    return corpus_info
