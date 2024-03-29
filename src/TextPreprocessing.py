# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 21:51:22 2019

@author: mauro
"""

import os
import json
import re
import pickle
from itertools import chain
from time import time
import configparser

import pandas as pd
import numpy as np

np.random.seed(0)


from utils.corpus_setup import select_corpus
from utils.log import initalize_logger

from utils.preprocessing import (
    sentence_preprocessing,
    plot_word_frequency,
    bootstrap,
    get_bigram_and_trigram_model,
    create_folder,
    get_spacy_model,
    sentence_selection
)


class TextPreprocessing(object):
    def __init__(self, pathMain, corpus_info):
        self.pathMain = pathMain
        self.name_of_folder = corpus_info["name_of_folder"]
        self.pathLoad = corpus_info["pathLoad"]
        self.filename = corpus_info["filename"]
        self.language = corpus_info["language"]
        

    def set_paths(self):
        """
        create the subfolder dependent on the data 
        """
        print(self.pathMain, self.name_of_folder)
        self.pathFolder = create_folder(
            os.path.join(self.pathMain, self.name_of_folder)
        )
        self.pathPlots = create_folder(os.path.join(self.pathFolder, "Plots"))
        self.pathLogs = create_folder(os.path.join(self.pathFolder, "Logs"))
        self.pathCSV = create_folder(os.path.join(self.pathFolder, "CSV"))

    def read_csv(self, delimiter, column_name="text"):
        """
        load the csv
        """
        df = pd.read_csv(
            os.path.join(self.pathLoad, self.filename), delimiter=delimiter, index_col=0
        )

        self.documents = [
            _text for _text in df[column_name].tolist() if isinstance(_text, str)
        ]

        # remove nan value
        self.documents = [
            document for document in self.documents if not isinstance(document, float)
        ]

        # get the number of sentences
        self.logger.info("documents: {}".format(len(self.documents)))
        return self.documents

    def read_sentences(self):
        """
        load the data and remove whitespace characters
        like `\n` at the end of each line
        use for EuroParl
        """
        self.documents = None
        with open(os.path.join(self.pathLoad, self.filename), encoding="utf8") as f:
            sentences = f.readlines()
        # remove whitespace characters like `\n` at the end of each line
        return [x.strip() for x in sentences]

    def clean_abbreviation(self):
        """
        remove dots from abbreviations, such that the sentence will be correctly split
        """
        with open(os.path.join('src', 'resources', 'abbrevation.json')) as json_file:
            abbrevation_dict = json.load(json_file)

        clean_docs = []
        for doc in self.documents:

            for abbrevation in abbrevation_dict['general']:
                doc = re.sub(abbrevation, lambda x: x.group().replace(".", ""), doc)

            if self.language == "en":
                doc = re.sub(
                    '|'.join(abbrevation_dict['en']),
                    lambda x: x.group().replace(".", ""),
                    doc,
                )
   
            elif self.language == "de":
                # find abbrevations and remove dot
                doc = re.sub(
                    '|'.join(abbrevation_dict['de']),
                    lambda x: x.group().replace(".", ""),
                    doc,
                )
                m = re.findall(
                    r"([ |(].{1,3}\. |\..{1}\. |\.[A-Z]{1}\.|[A-Z]{2,5}[a-z]\. )", doc
                )
                regex = [re.escape(value) for value in m]
                doc = re.sub("|".join(regex), lambda x: x.group().replace(".", ""), doc)
            clean_docs.append(doc)
        self.documents = clean_docs

    def get_paragraphs_from_documents(self):
        """
        split documents into paragraphs
        """
        paragraphs_per_doc = [document.split("\n") for document in self.documents]
        self.paragraphs = list(chain.from_iterable(paragraphs_per_doc))
        self.logger.info("paragraphs: {}".format(len(self.paragraphs)))

    def get_sentences_from_paragraphs(self, nlp):
        """
        split paragraphs into sentences
        """
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        # split parapraphs into sentences
        sentences = []
        for _count, paragraph in enumerate(
            nlp.pipe(self.paragraphs, batch_size=1000)
        ):
            # tokenize the sentences

            # sentences.append([sent.text for sent in paragraph.sents])
            # TODO: decide change?
            for sent in paragraph.sents:
                if ";" in sent.text:
                    for sent in sent.text.split(";"):
                        sentences.append([sent])

                else:
                    sentences.append([sent.text])

        # flatten the list of sentences
        sentences = list(chain.from_iterable(sentences))
        self.logger.info("sentences: {}".format(len(sentences)))
        return sentences

    def start_logger(self, stage="preprocessing"):
        """
        init the logger
        """
        self.logger = initalize_logger(self.language, self.pathLogs, stage=stage)
        self.logger.info("[Start Logging]")
        return self.logger

    def create_index_list(self, num_corpus, sampling_size, bootstrapping):
        """
        create a list of sentence indexes using resampling, such that
        the sentences can be bootstraped
        """
        index_list = bootstrap(
            num_corpus, len(self.sent), sampling_size, self.pathPlots, bootstrapping
        )
        # save the index list
        with open(os.path.join(self.pathFolder, "index_list.pickle"), "wb") as fp:
            pickle.dump(index_list, fp)

    def save_files(self, corpus_info):
        """
        save the processed sentences and the raw text
        """

        with open(
            os.path.join(
                self.pathFolder, "{}.pickle".format(corpus_info["name_of_folder"])
            ),
            "wb",
        ) as fp:
            pickle.dump(self.sent, fp)

    def load_files(self, corpus_info):
        """
        load the sentences
        """
        with open(
            os.path.join(
                self.pathFolder, "{}.pickle".format(corpus_info["name_of_folder"])
            ),
            "rb",
        ) as f:
            self.sent = pickle.load(f)

        self.logger.info(
            "Clean sentences loaded: {}".format("{}.pickle".format(self.language))
        )
        self.logger.info("Number of Sentences: {}".format(len(self.sent)))
        return self.sent

    def get_sentences_from_documents(self):
        """
        split documents into paragraphs and extract the sentences
        """
        nlp = get_spacy_model(self.language)
        # load the spacy model
        self.get_paragraphs_from_documents()
        sentences_original = self.get_sentences_from_paragraphs(nlp)
        return sentence_selection(sentences_original)

    def load_ngram(self, ngram_file="ngram.json"):
        """
        load n-gram parameters
        """
        with open(ngram_file) as json_file:
            self.n_gram = json.load(json_file)
        self.logger.info("n_gram: {}".format(self.n_gram))
        return self.n_gram

    def execute_preprocessing(
        self, sentences_original, lowercase, lemmatize, remove_stopwords
    ):
        """
        conduct text preprocessing
        """
        nlp = get_spacy_model(self.language)
        self.logger.info("Start sentence cleaning! (this can take a while...)")
        # clean the sentences
        sentences = sentence_preprocessing(
            sentences_original,
            nlp,
            lowercase=lowercase,
            lemmatize=lemmatize,
            remove_stopwords=remove_stopwords,
        )
        # remove empty list
        sentences = [
            sentence for sentence in sentences if sentence and len(sentence) > 2
        ]

        self.logger.info("Number of Sentences: {}".format(len(sentences)))
        self.logger.debug("Start ngram calculation!")

        # load ngram parameters from json file
        self.load_ngram()
        # calculate bigram
        bigram, trigram = get_bigram_and_trigram_model(
            sentences, self.n_gram, self.language, self.pathFolder
        )
        t = time()
        # sentence level analysis
        self.sent = [
            sentence
            for sentence in list(trigram[bigram[sentences]])
            if len(sentence) > 2
        ]

        self.logger.info("Number of Sentences: {}".format(len(self.sent)))
        self.logger.info(
            "Time to create bigrams and trigrams: {} mins".format(
                round((time() - t) / 60, 2)
            )
        )

        return self.sent


    def create_corpus_summary(self, fdist, sentences, sentences_original, documents):
        """
        Create corpus summary statistics
        """
        summary_dict = {}
        summary_dict[corpus_info["filename"]] = {}
        summary_dict[corpus_info["filename"]]["number of total tokens"] = sum(
            fdist.values()
        )
        summary_dict[corpus_info["filename"]]["vocabulary size"] = len(fdist)
        summary_dict[corpus_info["filename"]]["number of sentences"] = len(sentences)
        summary_dict[corpus_info["filename"]]["number of original sentences"] = len(
            sentences_original
        )
        summary_dict[corpus_info["filename"]]["number of documents"] = len(documents)
        pd.DataFrame(summary_dict).to_csv(
            os.path.join(self.pathCSV, "corpus_summary.csv")
        )
        return summary_dict
    
    def main(self, corpus_info):
        """
        conduct copurs text processing
        """
        corpus.set_paths()
        # start logger
        logger = corpus.start_logger()
        logger.info("\n[TEXT PROCESSING]")
        logger.info("corpus_info: {}".format(corpus_info))
        logger.info("bootstrapping: {}".format(bootstrapping))
        logger.info("sampling_size: {}".format(sampling_size))
        logger.info("num_corpus: {}".format(num_corpus))
        logger.info("sampling_size: {}".format(sampling_size))
    
        # document level - split into sentences
        if corpus_info["document_level"]:
    
            documents = corpus.read_csv(
                corpus_info["delimiter"], corpus_info["column_name"]
            )
            # text preprocessing:
            corpus.clean_abbreviation()
            # get original setnences
            sentences_original = corpus.get_sentences_from_documents()
            # get cleaned sentences
            sentences = corpus.execute_preprocessing(
                sentences_original, lowercase, lemmatize, remove_stopwords
            )
    
        # sentence level
        elif not corpus_info["document_level"]:
            documents = []
            # get original setnences
    
            sentences_original = corpus.read_sentences()
            # get cleaned sentences
            sentences = corpus.execute_preprocessing(
                sentences_original, lowercase, lemmatize, remove_stopwords
            )
        # load sentences
        elif load:
            sentences = corpus.load_files(corpus_info)
    
        # create list of sentences indexs, used for bootstrapping
        if create_index_list:
            corpus.create_index_list(num_corpus, sampling_size, bootstrapping)
    
        # plot word frequency
        fdist = plot_word_frequency(sentences, corpus.pathPlots, corpus_info["filename"])
    
        summary_dict = self.create_corpus_summary(fdist, sentences, sentences_original, documents)
        logger.info(summary_dict)
    
        if save:
            corpus.save_files(corpus_info)
    
        return documents, sentences, sentences_original


if __name__ == "__main__":
    
    
    # load main path from config file
    config = configparser.ConfigParser()
    config.read(
        os.path.join(os.getcwd(),"src",  "resources", "config.ini")
    )
    pathMain = config["paths"]["main"]
    

    # set the sentence sampling size
    create_index_list = True
    # downsample the number of sentences extracted from the corpus
    sampling_size = None
    # bootstrapp the sentences
    bootstrapping = False
    # set the numer of corpora (relevant for bootstrapping)
    num_corpus = 1 if bootstrapping else 1

    lemmatize = True
    lowercase = True
    remove_stopwords = True

    # save sentences and index list
    save = True
    plot = True
    load = False

    documents = {}
    sentences = {}
    sentences_original = {}

    for corpus_number in [0]:
        corpus_info = select_corpus(corpus_number)
        
        corpus = TextPreprocessing(
            pathMain, corpus_info
            
        )
        corpus.main(corpus_info)
