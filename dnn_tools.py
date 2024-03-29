import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sld_pipeline import *
import tensorflow as tf  ## this code runs with tf2.0-cpu!!!
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import sys

# to use latex with matplotlib
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# # # # # # # # # # # #

def get_dfs(data_types, data_dir, data_tag=""):
    dfs = []
    class_labels = []
    for i, t in enumerate(data_types):
        print('appending ' + t + ' dataframe...')
        if t == 'sld_mu':
            dfs.append(pd.read_csv(data_dir + '/flat_slmu' + data_tag + '.csv'))
            dfs[i]['sig_label'] = 1
            dfs[i]['type_label'] = 1
            class_labels.append('sl\_mu')
        elif t == 'ppim':
            dfs.append(pd.read_csv(data_dir + '/flat_ppim' + data_tag + '.csv'))
            dfs[i]['sig_label'] = 0
            dfs[i]['type_label'] = 2
            class_labels.append('p\_pim')
        elif t == 'fastpi':
            dfs.append(pd.read_csv(data_dir + '/flat_fastpi' + data_tag + '.csv'))
            dfs[i]['sig_label'] = 0
            dfs[i]['type_label'] = 3
            class_labels.append('fastpi')
        elif t == 'pim_gam':
            dfs.append(pd.read_csv(data_dir + '/flat_pimgam' + data_tag + '.csv'))
            dfs[i]['sig_label'] = 0
            dfs[i]['type_label'] = 4
            class_labels.append('pim\_gam')
        elif t == 'sld_e':
            dfs.append(pd.read_csv(data_dir + '/flat_sle' + data_tag + '.csv'))
            dfs[i]['sig_label'] = 0
            dfs[i]['type_label'] = 5
            class_labels.append('sl\_e')

    return dfs, class_labels

# # # # # # # # # # # #

def plot_confusion_matrix(y_true, y_pred, classes, ax,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    #fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', vmin=0.0, vmax=1.0, cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    return ax

# # # # # # # # # # # #

def dec_score_comp(mod, x_train, x_test):
    # generate decision score histograms to calculate overtraining parameter...
    decis_train = mod.predict(x_train).ravel()
    decis_test = mod.predict(x_test).ravel()
    # make histograms with 100 bins, get bin contents
    counts_train, bin_edges_train = np.histogram(decis_train, 100, range=(0,1))
    counts_test, bin_edges_test = np.histogram(decis_test, 100, range=(0,1))
    # rescale the train scores to the test scores by integral...
    counts_train = np.sum(counts_test)/np.sum(counts_train) * counts_train

    diff_sum = 0
    for i in range(len(counts_train)):
        diff_sum += (counts_train[i] - counts_test[i])**2
    diff_sum = diff_sum**0.5 / np.sum(counts_train)
    return diff_sum

# # # # # # # # # # # #

def multiclass_model(n_inputs, n_classes, n_hidden, hidden_nodes, input_dropout=0.0, biases=True):
    model = Sequential()
    if input_dropout > 0.0:
        model.add(Dropout(input_dropout, input_shape=(n_inputs, )))
        model.add(Dense(hidden_nodes[0], activation='relu', use_bias=biases))
    else:
        model.add(Dense(hidden_nodes[0], input_dim=n_inputs,
                        activation='relu', use_bias=biases))

    for i in range(n_hidden - 1):
        model.add(Dense(hidden_nodes[i+1], activation='relu', use_bias=biases))

    model.add(Dense(n_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# # # # # # # # # # # #

def binary_model(n_inputs, n_hidden, hidden_nodes, input_dropout=0.0, biases=True):
    model = Sequential()
    if input_dropout > 0.0:
        model.add(Dropout(input_dropout, input_shape=(n_inputs, )))
        model.add(Dense(hidden_nodes[0], activation='relu', use_bias=biases))
    else:
        model.add(Dense(hidden_nodes[0], input_dim=n_inputs,
                        activation='relu', use_bias=biases))

    for i in range(n_hidden - 1):
        model.add(Dense(hidden_nodes[i+1], activation='relu', use_bias=biases))

    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
