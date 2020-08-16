# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 21:51:22 2019

@author: mauro
"""

import pandas as pd  # For data handling
from time import time  # To time our operations
import os
import multiprocessing
import numpy as np
import sys
import json
#import logging  # Setting up the loggings to monitor gensim
import pickle
from itertools import chain
import re
import configparser

np.random.seed(0)
 

from auxiliary.corpus_setup import select_corpus
from auxiliary.log import initalize_logger

from auxiliary.preprocessing import sentence_preprocessing, plot_word_frequency, \
                          bootstrap, get_bigram_and_trigram_model, \
                          create_folder, get_spacy_model



def sentence_selection(sentences):
    """
    select sentences that are not only space and have more than two tokens
    """
    return [sent.strip() for sent in sentences if (sent or not sent.isspace()) \
                                                   and len(sent.split()) > 2]

# renmae to TextPreprocessing
class TextPreprocessing(object):
    
    def __init__(self, filename, language, name_of_folder, pathMain, pathLoad):
        self.name_of_folder = name_of_folder
        self.pathMain = pathMain
        self.pathLoad = pathLoad
        self.filename = filename
        self.language = language
        
    def set_paths(self):
        """ create the subfolder dependent on the data """
        
        print(self.pathMain, self.name_of_folder)
        
        self.pathFolder = create_folder(os.path.join(self.pathMain, self.name_of_folder))
        self.pathPlots = create_folder(os.path.join(self.pathFolder, 'Plots'))
        self.pathLogs = create_folder(os.path.join(self.pathFolder, 'Logs'))
    
    def read_csv(self, delimiter, column_name='text'):
        """ 
        load the csv 
        """
        df = pd.read_csv(os.path.join(self.pathLoad, self.filename),
                         delimiter=delimiter, 
                         index_col=0)
        
        self.documents = df[column_name].tolist()
        # remove nan value
        self.documents = [document for document in self.documents 
                          if not isinstance(document, float)]
        
        # get the number of sentences
        self.logger.info('documents: {}'.format(len(self.documents)))
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
        clean_docs = []
        for doc in self.documents:
            if self.language == 'en':
                doc = re.sub('Misc.|u.|u.a.|1.|2.|3.|4.|5.|6.|7.|8.|9.|0.|Art.|Nr.|vgl.|ff.|Aufl.|Rn.|Fn.|Dr.|Prof.|II.|III.|f.', lambda x: x.group().replace('.',''), doc)
                doc = re.sub('i.e.|ed.|Art.|L.Ed.|U.S.C.A.|seq.|C.C.A.|D.C.|S.Ct.|c.|i.|f.|Ed.|Ct.|v.|Inc.|Div.|U.S.|Sup.|Co.|10.|11.|12.|13.|14.|15.|16.|17.|18.|19.|20.|21.|22.|23.|24.|25.|26.|27.|28.|29.|30.|31.|II.|III.|Sci.|loci.|Int.',lambda x: x.group().replace('.',''), doc)
                doc = re.sub('e.g.|Pub.L.|No.|id.|S.Rep.|S.Res.|Sess.|Cong.Rec.|cl.|Art.|Pt.A|U.S.S.G.|Ch.|Circ.|App.A.|Cf.|Ct.Cl.|C.C.A.|Ins.|Stat.|Ry.|Wall.|Rep.|Pa.|Bi.|e.V.|cent.|36.|37.|39.|40.|41.|42.|43.|44.|45.|50.|51.|52.|53.|54.|55.|56.|57.|58.|59.|60.|61.|62.|63.|64.|65.|66.|67.|68.|69.|App.|OD.|m.|Mi.|Ti.|Ka.|Diss.|Ed.|ders.|jug.|Öz.|Co.|j.|Özk.|St.|Sz.|Ö.|A.|B.|C.|D.|E.|F.|G.|H.|I.|J.|K.|L.|M.|N.|O.|P.|Q.|R.|S.|T.|U.|V.|W.|X.|Y.|Z.|Sch.',lambda x: x.group().replace('.',''),doc)
            elif self.language == 'de':
                 # find abbrevations and remove dot
                 doc = re.sub('Misc.|u.|u.a.|1.|2.|3.|4.|5.|6.|7.|8.|9.|0.|Art.|Nr.|vgl.|ff.|Aufl.|Rn.|Fn.|Dr.|Prof.|II.|III.|f.', lambda x: x.group().replace('.',''), doc)
                 m = re.findall(r'([ |(].{1,3}\. |\..{1}\. |\.[A-Z]{1}\.|[A-Z]{2,5}[a-z]\. )', doc)
                 regex = [re.escape(value) for value in m]
                 doc = re.sub('|'.join(regex), lambda x: x.group().replace('.',''), doc)
            clean_docs.append(doc)
        self.documents = clean_docs
        
    def get_paragraphs_from_documents(self):
        """ 
        split documents into paragraphs 
        """
        # get paragraphs per document
        paragraphs_per_doc = [document.split('\n') for document in self.documents]
        # flatten paragraph list
        self.paragraphs = list(chain.from_iterable(paragraphs_per_doc))
        self.logger.info('paragraphs: {}'.format(len(self.paragraphs)))
    
    def get_sentences_from_paragraphs(self, nlp):
        """ 
        split paragraphs into sentences 
        """
        
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        # split parapraphs into sentences
        sentences = []
        for _count, paragraph in enumerate(nlp.pipe(self.paragraphs, n_threads=-1, batch_size=1000)):
            # tokenize the sentences
            sentences.append([sent.text for sent in paragraph.sents])
        
        # flatten the list of sentences
        sentences = list(chain.from_iterable(sentences))
        self.logger.info('sentences: {}'.format(len(sentences)))        
        return sentences
        
    def start_logger(self, stage = 'preprocessing'):
        """ 
        init the logger 
        """
        self.logger = initalize_logger(self.language, self.pathLogs, stage = stage)
        self.logger.info('[Start Logging]')
        return self.logger
        
    def create_index_list(self, num_corpus, sampling_size, bootstrapping):
        """ 
        create a list of sentence indexes using resampling, such that
        the sentences can be bootstraped
        """
        # create bootstrap lists
        index_list = bootstrap(num_corpus, len(self.sent), sampling_size,
                               self.pathPlots, bootstrapping)
        # save the index list
        with open(os.path.join(self.pathFolder, 'index_list.pickle'), 'wb') as fp:
            pickle.dump(index_list, fp)

    def save_files(self):
        """ 
        save the processed sentences and the raw text
        """
        if self.documents is not None:
            # save the processed documents
            with open(os.path.join(self.pathFolder, 'raw_text_{}.pickle'.format(self.language)), 'wb') as fp:
                pickle.dump(self.documents, fp)
        
        # save the processed sentences
        with open(os.path.join(self.pathFolder, '{}.pickle'.format(self.language)), 'wb') as fp:
            pickle.dump(self.sent, fp)
        
        
    def load_processed_sentences(self):
        """ 
        load the sentences
        """
        # load processed documents
        with open(os.path.join(self.pathFolder, '{}.pickle'.format(self.language)), 'rb') as f:
            self.sent = pickle.load(f)
            
        self.logger.info('Clean sentences loaded: {}'.format('{}.pickle'.format(self.language)))
        self.logger.info('Number of Sentences: {}'.format(len(self.sent)))
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
    
    
    def load_ngram(self, ngram_filename='ngram.json'):
        """
        load n-gram parameters
        """
        with open(ngram_filename) as json_file:
            self.n_gram = json.load(json_file)
        self.logger.info('n_gram: {}'.format(self.n_gram))
        return self.ngram
    
    def execute_preprocessing(self, sentences_original, lowercase, lemmatize):
        """ 
        conduct text preprocessing 
        """
        nlp = get_spacy_model(self.language)
        self.logger.info('Start sentence cleaning! (this can take a while...)')                   
        # clean the sentences        
        sentences = sentence_preprocessing(sentences_original, 
                                           nlp,
                                           lowercase=lowercase,
                                           lemmatize=lemmatize)
        # remove empty list
        sentences = [sentence for sentence in sentences if sentence and len(sentence) > 1]
        
        self.logger.info('Number of Sentences: {}'.format(len(sentences)))
        self.logger.debug('Start ngram calculation!')                   

        # load ngram parameters from json file
        self.load_ngram()
        # calculate bigram
        bigram, trigram = get_bigram_and_trigram_model(sentences, 
                                                       self.n_gram,
                                                       self.language,
                                                       self.pathFolder)
        t = time()
        # sentence level analysis       
        self.sent = list(trigram[bigram[sentences]])
        self.logger.info('Number of Sentences: {}'.format(len(self.sent)))
        self.logger.info('Time to create bigrams and trigrams: {} mins'.format(round((time() - t) / 60, 2)))
        
        return self.sent



def main(corpus_info):
    """
    conduct copurs text processing
    """
     # load main path from config file
    config = configparser.ConfigParser()    
    config.read('config.ini')              
        
    prepro = TextPreprocessing(corpus_info['filename'], 
                               corpus_info['language'], 
                               corpus_info['name_of_folder'], 
                               config['paths']['main'], 
                               corpus_info['pathLoad'] )
    
    prepro.set_paths()
    # start logger
    logger = prepro.start_logger()
    
    logger.info('corpus_info: {}'.format(corpus_info))
    logger.info('bootstrapping: {}'.format(bootstrapping))
    logger.info('sampling_size: {}'.format(sampling_size))
    logger.info('num_corpus: {}'.format(num_corpus))
    logger.info('sampling_size: {}'.format(sampling_size))        

    # document level
    if prepro.filename.split('.')[0] != 'europarl-v7':

        documents = prepro.read_csv(corpus_info['delimiter'], corpus_info['column_name'])
        # text preprocessing:
        prepro.clean_abbreviation()
        # get original setnences
        sentences_original = prepro.get_sentences_from_documents()
        # get cleaned sentences
        sentences = prepro.execute_preprocessing(sentences_original,
                                                 lowercase,
                                                 lemmatize)
    
    # sentence level
    elif prepro.filename.split('.')[0] == 'europarl-v7':
        documents = []
        # get original setnences

        sentences_original = prepro.read_sentences()
        # get cleaned sentences
        sentences = prepro.execute_preprocessing(sentences_original,
                                                 lowercase,
                                                 lemmatize)
    # load sentences
    elif load:
        sentences = prepro.load_processed_sentences()
 
    # create list of sentences indexs, used for bootstrapping 
    if create_index_list:
        prepro.create_index_list(num_corpus, sampling_size, bootstrapping)        

    if plot:
        # plot word frequency
        _ = plot_word_frequency(sentences, prepro.pathPlots, corpus_info['filename'])
        
    if save:
        prepro.save_files()
        
    return documents, sentences, sentences_original



if __name__ == "__main__":
    
    # set the sentence sampling size
    create_index_list = True
    # downsample the number of sentences extracted from the corpus
    sampling_size = None
    # bootstrapp the sentences
    bootstrapping = False
    # set the numer of corpora (relevant for bootstrapping)
    num_corpus = 1

    lemmatize = False
    lowercase = False    

    # save sentences and index list
    save = True
    plot = True
    load = False
    
    for number in [1]:    
        corpus_info = select_corpus(number)
        documents, sentences, sentences_original = main(corpus_info)
    