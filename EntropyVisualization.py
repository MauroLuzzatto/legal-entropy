# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 18:26:14 2020

@author: mauro

"""

import pandas as pd  # For data handling
import os, sys
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from matplotlib import colors as mcolors
import json
import configparser
from scipy.stats import entropy

from EntropyEvaluation import get_list_of_relevant_words, create_folder
from auxiliary.experiment_setup import select_experiment

def pos_tag(text, nlp):
    doc = nlp(text)
    return doc[0].pos_


def get_label_by_language(language):
    """ get label for the plot """
    
    if language in ['en4', 'en5', 'en15']:
        plot_label = 'Supreme Court EN'
        shade = True
        ls = '-'
        c = 'C2'
    elif language in ['en2', 'en8', 'en18']:
        plot_label = 'EuroParl EN'
        shade = True        
        ls = '-'
        c = 'C8'

    elif language in ['de2', 'de8', 'de18']:
        plot_label = 'EuroParl DE'
        shade = True
        ls = '-'
        c = 'C4'

    elif language in ['de5', 'de15']:
        plot_label = 'BGH Strafsenat'
        shade = True
        ls = '-'
        c = 'C0'
        
        
    elif language in ['de6', 'de16']:
        plot_label = 'BGH Zivilsenat'
        shade = True
        ls = '-'     
        c = 'C1'
            
    elif language in ['de7', 'de17']:
        plot_label = 'BGH DE'
        shade = True
        ls = '-'
    
    else:
        plot_label = language
        shade = True
        ls = '-'
        c = 'C1'
        
    if language in ['de15', 'de16', 'en15', 'de17', 'de18', 'en18']:
        plot_label += ' shuffled'

    return plot_label, shade, ls, c


# load main path from config file
config = configparser.ConfigParser()
config.read('config.ini')
pathMain = config['paths']['main']
name = 'legal_code'

pathFolders, pathSave, _, num_corpus, _ = select_experiment(name)

use_relevant_words = False

# load the legend to the data
with open(os.path.join(pathSave, 'legend.json')) as json_file:
    legend_dict = json.load(json_file)
    
# load the ambiguity distributions
# distribution_lan
with open(os.path.join(pathSave, 'entropy_distribution.pickle'), 'rb') as fp:
    entropy_distribution = pickle.load(fp) 



pathPlot = create_folder(os.path.join(pathSave, 'Plots'))
pathCSV = create_folder(os.path.join(pathSave, 'CSV'))

#num_corpus = 25
# languages = list(entropy_distribution.keys())
names = list(entropy_distribution.keys())
colors = ['C{}'.format(ii) for ii in range(10)]


##########################################################################
# Plot word entroyp mean and std in scatter plot


entropy_dict = {}
summary_dict = {}

for count, name in enumerate(names):
    
    mean_entropy = []
    std_entropy = []
    word_count = []
    
    for w in entropy_distribution[name].keys():
        word_count.append(np.mean([entropy_distribution[name][w][num][1] for num in range(num_corpus)]))
        mean_entropy.append(np.mean([entropy_distribution[name][w][num][0] for num in range(num_corpus)]))
        std_entropy.append(np.std([entropy_distribution[name][w][num][0] for num in range(num_corpus)]))

    
    entropy_dict[name] = list(entropy_distribution[name].keys())
    entropy_dict['count_' + name] = word_count
    entropy_dict['mean_' + name] = mean_entropy
    entropy_dict['std_' + name] = std_entropy
    entropy_dict['samples_' + name] =  [num_corpus for _ in entropy_distribution[name].keys()]
    entropy_dict['underscore_' + name] = [w.count('_') for w in entropy_dict[name]]
    
    summary_dict[name] = {}
    
    probabilites = np.array(word_count)/np.sum(word_count)
    summary_dict[name]['global_unigram_entropy'] = entropy(probabilites, base=2)

    summary_dict[name]['mean_entropy'] = np.mean(mean_entropy)
    summary_dict[name]['std_entropy'] = np.std(mean_entropy)
    
    summary_dict[name]['mean_count'] = np.mean(word_count)
    summary_dict[name]['std_count'] = np.std(word_count)
    
    # summary_dict[name]['mean_ngram'] = np.mean(entropy_dict['underscore_' + name])
    # summary_dict[name]['std_ngram'] = np.std(entropy_dict['underscore_' + name])

df_summary = pd.DataFrame(summary_dict)

save = True
if save:
    df_summary.to_csv(os.path.join(pathCSV, 'summary.csv'), sep=';')
            


class BasePlot(object):
    
    def __init__(self, entropy_dict, names, figure_name='plot.png', pathPlot=r'', 
                 xlabel=None, ylabel=None, lim=None, ylim=None, 
                 figsize=(8,5), cumulative=True, grid=True, save=True):
        
        self.path = os.path.join(pathPlot, figure_name)
        self.names = names
        self.entropy_dict = entropy_dict
        self.column = 'mean_'
        
        self.save = save    
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.grid = grid
        
        self.cumulative = cumulative
        self.figsize = figsize

        self.fontsize_legend = 12

        
        
    def figure(self):
        self.fig = plt.figure(figsize=self.figsize)

        
    def plot(self):
                
        for name in self.names:                
            sns.kdeplot(self.entropy_dict[self.column + name],
                        label=name,
                        cumulative = self.cumulative,
                        shade=False)
        
    def label(self):
        
        if self.legend:
            plt.legend(loc ='lower right', 
                       fontsize=self.fontsize_legend)    
        if self.xlim is not None:
            plt.xlim(self.xlim)
        if self.ylim is not None:
            plt.ylim(self.ylim)
        if self.ylabel is not None:
            plt.ylabel(self.ylabel)
        if self.xlabel is not None:
            plt.xlabel(self.xlabel)
        if self.grid:
            plt.grid()
            
        plt.show()
    
    def save(self):
        self.fig.savefig(self.path)

    def create(self):
        
        self.figure()
        self.plot()
        self.label()
        
        if self.save:
            self.save()
        
# class KdePlot(BasePlot):
    
#     def plot()
# class
# figure
# data
# plot
# label
# save

def boxplot():
    """
    boxplot entropy
    """

    save = True
    fig = plt.figure(figsize = (len(names) + 1, 5))    

    data = [entropy_dict['mean_' + name] for name in names]
    positions = [ii for ii in range(1, len(names) + 1)]
    
    plt.boxplot(data, 
                positions=positions, 
                labels=names,
                notch=True)
    
    plt.grid()    
    plt.tight_layout()
    plt.ylabel('Word Entropy')

    
    if save:
        fig.savefig(os.path.join(pathPlot, 'boxplot_entropy_{}.png'.format('_'.join(names))))

def boxplot_ngram():
    """
    boxplot ngram
    """

    save = True
    fig = plt.figure(figsize = (len(names) + 1, 5))    

    data = [entropy_dict['underscore_' + name] for name in names]
    positions = [ii for ii in range(1, len(names) + 1)]
    
    plt.boxplot(data, 
                positions=positions, 
                labels=names,
                notch=True)
    
    plt.grid()    
    plt.tight_layout()
    plt.ylabel('#ngram')

    
    if save:
        fig.savefig(os.path.join(pathPlot, 'boxplot_ngram_{}.png'.format('_'.join(names))))



def boxplot_count():
    """
    boxplot count
    """

    save = True
    fig = plt.figure(figsize = (len(names) + 1, 5))    

    data = [np.log10(entropy_dict['count_' + name]) for name in names]
    positions = [ii for ii in range(1, len(names) + 1)]
    
    plt.boxplot(data, 
                positions=positions, 
                labels=names,
                notch=True)
    
    plt.grid()    
    plt.tight_layout()
    plt.ylabel('Word Count [log10]')

    
    if save:
        fig.savefig(os.path.join(pathPlot, 'boxplot_count_{}.png'.format('_'.join(names))))


def plot_histogram():
    """
    plot entropy histogram
    """

    save = True
    fig = plt.figure(figsize = (8, 5))    

    for count, name in enumerate(names):
                
        plot_label, shade, ls, _ = get_label_by_language(name)
            
        plt.hist(entropy_dict['mean_' + name], 
                 bins = 200, 
                 label=name,
                 color=colors[count],
                 alpha = 0.3)
    
    plt.grid()
    plt.legend(loc = 'upper left')
    plt.xlim(0,18)
    #plt.ylim(0,0.2)
    
    plt.xlabel('Word Entropy')
    plt.tight_layout()
    plt.legend(loc = 'upper left')
    
    if save:
        fig.savefig(os.path.join(pathPlot, 'hist_entropy_{}.png'.format('_'.join(names))))


def plot_histogram_count():
    """
    plot entropy histogram
    """

    save = True
    fig = plt.figure(figsize = (8, 5))    

    for count, name in enumerate(names):
                
        plot_label, shade, ls, _ = get_label_by_language(name)
            
        plt.hist(np.log10(entropy_dict['count_' + name]), 
                 bins = 100, 
                 label=name,
                 color = colors[count],
                 alpha=0.3)
    
    plt.grid()
    plt.legend(loc = 'upper left')
    # plt.xlim(0,18)
    #plt.ylim(0,0.2)
    
    plt.xlabel('Word count [log10]')
    plt.tight_layout()
    plt.legend(loc = 'upper left')
    
    if save:
        fig.savefig(os.path.join(pathPlot, 'hist_count_{}.png'.format('_'.join(names))))



def plot_KDE():
    """
    plot KDE
    """

    save = True

    fig = plt.figure(figsize = (8, 5))    
    
    
    for name in names:
        plot_label, shade, ls, _ = get_label_by_language(name)
        
        sns.kdeplot(entropy_dict['mean_' + name],
                    label=name,
                    ls = ls,
                    shade=False)
    
    plt.grid()
    plt.legend(loc = 'upper left')
    plt.xlabel('Word Entropy')
    
    plt.xlim(0,18)
    plt.ylim(0,0.25)
    
    if save:
        fig.savefig(os.path.join(pathPlot, 'KDE_entropy_{}.png'.format('_'.join(names))))



def plot_ECDF_entropy():
    """
    
    """
    
    save = True
    _fontsize_legend = 13
    
    # CDF all values in one plot
    fig = plt.figure(figsize = (8, 5))    
    
    for name in names:
        _, shade, ls, _ = get_label_by_language(name)
                
        sns.kdeplot(entropy_dict['mean_' + name],
                    label=name,
                    cumulative = True,
                    shade=False)
    
    plt.grid()
    plt.legend(loc = 'lower right', fontsize = _fontsize_legend)
    plt.xlim(0,16)
    plt.ylim(0,1)
    plt.ylabel('ECDF')
    plt.xlabel('Word Entropy')
    plt.show()

    
    if save:
        fig.savefig(os.path.join(pathPlot, 'ECDF_entropy{}.png'.format('_'.join(names))))



def plot_ECDF_count():
    """
    
    """
    
    save = True
    _fontsize_legend = 13
    
    # CDF all values in one plot
    fig = plt.figure(figsize = (8, 5))    
    
    for name in names:            
        sns.kdeplot(np.log10(entropy_dict['count_' + name]),
                    label=name,
                    cumulative = True,
                    shade=False)
    plt.grid()
    plt.legend(loc = 'lower right', fontsize = _fontsize_legend)
    plt.ylim(0,1)
    plt.ylabel('ECDF')
    plt.xlabel('Word Count [log10]')
    plt.show()

    
    if save:
        fig.savefig(os.path.join(pathPlot, 'ECDF_count{}.png'.format('_'.join(names))))
    



def scatter_entropy_vs_count():
    
    save = True
    _s = 3
    _alpha = 0.1
    
    fig = plt.figure(figsize = (8, 5))    

    for count, name in enumerate(names):
        
        plt.scatter(entropy_dict['mean_' + name], 
                    np.log10(entropy_dict['count_' + name]), 
                    c = colors[count],
                    s = _s, 
                    alpha = _alpha,
                    label = name)   
                
    plt.grid()
    plt.xlabel('Word Entropy')
    plt.ylabel('Word Count [log10]')
    plt.tight_layout()
    plt.show()
    
    if save:
        fig.savefig(os.path.join(pathPlot, 'scatter_entropy_vs_count_{}.png'.format('_'.join(names))))

    



def plot_KDE_2D():

    save = True    
    for count, name in enumerate(names):
    
        fig = plt.figure(figsize = (6, 5))    

        # Create a cubehelix colormap to use with kdeplot    
        cmap = sns.cubehelix_palette(start=0, light=1, as_cmap=True)    
    
        # Generate and plot a random bivariate dataset
        x = entropy_dict['mean_' + name]
        y = np.log10(entropy_dict['count_' + name])
        
        
        sns.kdeplot(x, y, cmap=cmap, shade=True)
        plt.title(name)
        plt.grid()
        plt.xlabel('Word Entropy')
        plt.ylabel('Word Count [log10]')
        plt.tight_layout()
        # plt.xlim(0,16)
        # plt.ylim(0,4)
        
        plt.show()
        
        if save:
            fig.savefig(os.path.join(pathPlot, 'KDE_2D_entropy_vs_count_{}.png'.format(name)))

    


    
    # # Set up the matplotlib figure
    # f, axes = plt.subplots(1, 3, figsize=(9, 4), sharex=True, sharey=True)
    
    # # Rotate the starting point around the cubehelix hue circle
    # for ax, name in zip(axes.flat, names):
    
    #     # Create a cubehelix colormap to use with kdeplot
    #     cmap = sns.cubehelix_palette(start=0, light=1, as_cmap=True)
    
    #     x = entropy_dict['mean_' + name]
    #     y = np.log10(entropy_dict['count_' + name])
    #     sns.kdeplot(x, y, cmap=cmap, shade=True, ax=ax)
        
    #     plt.title(name)
    #     plt.grid()
    #     plt.xlabel('Word Entropy')
    #     plt.ylabel('Word Count [log10]')
    #     # plt.xlim(0,16)
    #     # plt.ylim(0,4)
    
    # f.tight_layout()


def entropy_to_csv():

    save = True
    for count, name in enumerate(names):
        
        columns = [name, 'count_' + name, 'mean_' + name, 'underscore_' + name]
        index = [name, 'word frequency', 'entropy', 'underscores']
        df = pd.DataFrame([entropy_dict[col] for col in columns], index = index).T
        df.sort_values(by = name, inplace = True)
        df.reset_index(inplace = True)
        df.drop(columns=['index'], inplace =True)
#         df = pd.DataFrame(entropy_distribution[name], index = ['entropy']).T
#         df.sort_values(by = 'entropy', inplace = True)
#         df = df[df['entropy']<=0.5].iloc[:150]
#         df.reset_index(inplace = True)

    
        if save:
            df.to_csv(os.path.join(pathCSV, '{}.csv'.format(name)), sep = ';', encoding='utf-8-sig')
            

boxplot()
boxplot_count()
entropy_to_csv()
boxplot_ngram()

sys.exit()
plot_histogram()
plot_histogram_count()
plot_KDE()
plot_ECDF_entropy()
plot_ECDF_count()
scatter_entropy_vs_count()
# plot_KDE_2D()
    

sys.exit()
