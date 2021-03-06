# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 20:37:49 2019

@author: mauro
"""
import os
import multiprocessing
import pickle
import itertools
import random
import json
from time import time

import numpy as np
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec

from utils.preprocessing import create_folder
from utils.log import initalize_logger
from utils.experiment_setup import select_experiment


cores = multiprocessing.cpu_count()  # Count the number of cores in a computer
np.random.seed(0)


def create_corpus(index_list, sentences):
    """
    use the list of indexes and the sentences and create several corpora
    """
    t = time()
    corpus = {}
    for count, index_sublist in enumerate(index_list):
        corpus[count] = []
        for idx in index_sublist:
            corpus[count].append(sentences[idx])
    print(
        "Create Cropus: {} mins, index: {}".format(round((time() - t) / 60, 2), count)
    )
    return corpus


class callback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        """
        get model training loss.
        """
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print("Loss after epoch {}: {}".format(self.epoch, loss))
        else:
            print(
                "Loss after epoch {}: {}".format(
                    self.epoch, loss - self.loss_previous_step
                )
            )
        self.epoch += 1
        self.loss_previous_step = loss


def shuffle_tokens_per_sentence(sent):
    """
    shuffle the tokens of the sentences
    """
    for sublist in sent:
        random.shuffle(sublist)
    return sent


class ModelTraining(object):
    """
    train the word2vec model
    """

    def __init__(self, sent_list, language, count, folder_name):
        self.model_name = "{}_{}_full_model.model".format(language, count)
        self.sent_list = sent_list
        self.folder_name = folder_name
        self.language = language

    def load_hyperparameters(self, json_filename="hyperparameters.json"):
        """
        load hyperparamters
        """
        with open(json_filename) as json_file:
            self.hyperparameters = json.load(json_file)
        return self.hyperparameters

    def set_hyperparamters(self, hyperparameters):
        """
        set hyperparamters
        """
        self.hyperparameters = hyperparameters
        return self.hyperparameters

    def training(self):
        """
        wrapper around the gensim word2vec training, save the model in the end
        """
        t = time()
        # init word2vec
        self.w2v_model = Word2Vec(
            min_count=self.hyperparameters["min_count"],
            window=self.hyperparameters["window"],
            size=self.hyperparameters["size"],
            workers=self.hyperparameters["workers"],
            sg=self.hyperparameters["sg"],
            hs=self.hyperparameters["hs"],
            negative=self.hyperparameters["negative"],
            sample=self.hyperparameters["sample"],
            alpha=self.hyperparameters["alpha"],
            min_alpha=self.hyperparameters["min_alpha"],
        )

        self.w2v_model.build_vocab(self.sent_list, progress_per=6000000)
        print("Time to build vocab: {} mins".format(round((time() - t) / 60, 2)))

        # train the w2v model
        self.w2v_model.train(
            self.sent_list,
            total_examples=self.w2v_model.corpus_count,
            epochs=self.hyperparameters["epochs"],
            report_delay=1,
            compute_loss=True,
            callbacks=[callback()],
        )

        print(
            "Time to train the model: {:.2f} mins".format(round((time() - t) / 60, 2))
        )
        return self.w2v_model

    def save(self):
        """
        save full model
        """
        self.w2v_model.save(os.path.join(self.folder_name, self.model_name))
        print("Model saved: {}".format(self.model_name))

    def evaluation(self, question_words="questions-words.txt"):
        """
        evaluate the word2vec models
        """
        if "en" in self.language:
            evaluation = self.w2v_model.wv.evaluate_word_analogies(
                analogies=question_words
            )[0]
            print("Evaluation:", evaluation)

    def frequency_dist(self):
        """
        plot the word frequency per model name and save the plot
        """
        fig = plt.figure(figsize=(10, 4))
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.title(self.model_name)
        fdist1 = FreqDist(list(itertools.chain.from_iterable(self.sent_list)))
        fdist1.plot(30, cumulative=False)

        fig.savefig(
            os.path.join(self.folder_name, self.model_name + ".png"),
            bbox_inches="tight",
        )


def main(pathFolder, name_of_folder, languages):
    """
    conduct the model training
    """
    # setup the folders
    pathModels = create_folder(os.path.join(pathFolder, "Models"))
    pathLogs = create_folder(os.path.join(pathFolder, "Logs"))

    # load the index list
    with open(os.path.join(pathFolder, "index_list.pickle"), "rb") as fp:
        index_list = pickle.load(fp)

    # todo: remove
    language = languages[0]

    # init the logger
    logger = initalize_logger(language, pathLogs, stage="training")
    logger.info("shuffle: {}".format(shuffle))

    # load the processed sentences
    with open(os.path.join(pathFolder, "{}.pickle".format(name_of_folder)), "rb") as fp:
        sentences = pickle.load(fp)

    # create the bootstrap corpora
    corpus = create_corpus(index_list, sentences)

    # train the models for the created corpora (sentences, index_list)
    for count, sentences_list in enumerate(corpus.values()):
        print("Number of model trained: {}/{}".format(count, len(corpus.values())))

        if shuffle:
            # shuffle the tokens per sentence
            sentences_list = shuffle_tokens_per_sentence(sentences_list)

        word2vec_model = ModelTraining(sentences_list, language, count, pathModels)

        # get set of hyperparameters
        hyperparameters = word2vec_model.load_hyperparameters()
        logger.info("hyperparamters: {}".format(hyperparameters))

        # train the model
        w2v_model = word2vec_model.training()

        # save the model
        if save:
            word2vec_model.save()
        # evaluate the word2vec model
        if evaluate:
            word2vec_model.evaluation()
        # plot word frequency
        if plot:
            word2vec_model.frequency_dist()

        return w2v_model


if __name__ == "__main__":

    plot = True
    evaluate = True
    shuffle = False
    save = True

    experiment_name = "legal_code"

    pathFolders, *_ = select_experiment(experiment_name)

    for pathFolder, languages in pathFolders:
        name_of_folder = os.path.split(pathFolder)[-1]
        w2v_model = main(pathFolder, name_of_folder, languages)
