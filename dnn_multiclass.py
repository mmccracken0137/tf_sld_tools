import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from sld_pipeline import *
from dnn_tools import *

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

# suppresses an annoying tf warning...
#tf.logging.set_verbosity(tf.logging.ERROR)

# hyper-parameters and data parameters
write_model_files = False # True
bkgd_type = 'all' #ppim' # 'fastpi' # 'ppim'
data_tag = '_small'
kf_type = 'p4vect'
kf_tag = ''
if kf_type == 'p4only':
    kf_tag = '_p4only'
drop_non_p4x4 = True #False

hidden_layer_nodes = [200, 100]
n_hidden_layers = len(hidden_layer_nodes)
dropout_frac = 0.25
n_epochs = 10

data_dir = sys.argv[1]

# read in feats files
print('\nreading data files...\n')
kf_tag = '' #_p4only'
# df_sld = pd.read_csv("feats_files/feats_sl_mu" + kf_tag + "_1___" + data_tag + ".csv")#, dtype='float64')
# df_ppim = pd.read_csv("feats_files/feats_ppim" + kf_tag + "_1___" + data_tag + ".csv")
# df_fastpi = pd.read_csv("feats_files/feats_ppim_fastpi" + kf_tag + "_1___" + data_tag + ".csv")
df_sld = pd.read_csv(data_dir + "/flat_slmu" + data_tag + ".csv")#, dtype='float64')
df_ppim = pd.read_csv(data_dir + "/flat_ppim" + data_tag + ".csv")
df_fastpi = pd.read_csv(data_dir + "/flat_fastpi" + data_tag + ".csv")
df_pimgam = pd.read_csv(data_dir + "/flat_pimgam" + data_tag + ".csv")

# add types/labels
df_sld['sig_label'] = 1
df_ppim['sig_label'] = 0
df_fastpi['sig_label'] = 0
df_pimgam['sig_label'] = 0

df_sld['type_label'] = 1
df_ppim['type_label'] = 2
df_fastpi['type_label'] = 3
df_pimgam['type_label'] = 4

class_labels = []
if bkgd_type == 'ppim':
    df = [df_sld, df_ppim]
    class_labels = ['sld\_mu', 'p\_pim']
elif bkgd_type == 'fastpi':
    df = [df_sld, df_fastpi]
    class_labels = ['sld\_mu', 'fastpi']
elif bkgd_type == 'all':
    df = [df_sld, df_ppim, df_fastpi, df_pimgam]
    class_labels = ['sld\_mu', 'p\_pim', 'fastpi', 'pim\_gam']
else:
    print('error: incorrect background type')
    sys.exit()
df = pd.concat(df)

### DROP higher beam energies
# df = df[df.beam_E < 5.0]

sld_add_features(df)

to_drop_nonp4x4 = ['run','rftime',
                   'kp_beta_time','kp_chisq_time','kp_ndf_time','kp_ndf_trk','kp_chisq_trk',
                   'kp_ndf_dedx','kp_chisq_dedx', 'kp_dedx_cdc','kp_dedx_fdc','kp_dedx_tof',
                   'kp_dedx_st','kp_ebcal','kp_eprebcal','kp_efcal','kp_bcal_delphi',
                   'kp_bcal_delz','kp_fcal_doca',
                   'p_beta_time','p_chisq_time','p_ndf_time','p_ndf_trk','p_chisq_trk',
                   'p_ndf_dedx','p_chisq_dedx','kp_dedx_cdc','p_dedx_fdc','p_dedx_tof',
                   'p_dedx_st','p_ebcal','p_eprebcal','p_efcal','p_bcal_delphi',
                   'p_bcal_delz','p_fcal_doca',
                   'mum_beta_time','mum_chisq_time','mum_ndf_time','mum_ndf_trk','mum_chisq_trk',
                   'mum_ndf_dedx','mum_chisq_dedx','mum_dedx_cdc','mum_dedx_fdc','mum_dedx_tof',
                   'mum_dedx_st','mum_ebcal','mum_eprebcal','mum_efcal','mum_bcal_delphi',
                   'mum_bcal_delz','mum_fcal_doca']
if drop_non_p4x4:
    df.drop(to_drop_nonp4x4, axis=1, inplace=True)

df.dropna(inplace=True)

y = df['sig_label']
y_type = df['type_label']
#print(y_type)

# generate one-hot label encoding
y_type_encoded, y_type_cats = y_type.factorize()
oh_enc = preprocessing.OneHotEncoder(categories='auto')
y_type_1hot = oh_enc.fit_transform(y_type_encoded.reshape(-1,1)).toarray()

df.drop(['event', 'sig_label', 'type_label'], axis=1, inplace=True)

to_drop = []
for f in df.columns:
    if 'meas' in f:
        to_drop.append(f)
#print(to_drop)
df.drop(to_drop, inplace=True, axis=1)

X_train, X_test, y_1hot_train, y_1hot_test = train_test_split(df, y_type_1hot,
                                                              test_size=0.25, random_state=42)


print('fitting sklearn.preprocessing.StandardScaler to X_train...')
scaler = preprocessing.StandardScaler().fit(X_train)
# print(scaler.mean_)
# print(scaler.scale_)

print('applying scaler to train data...\n')
X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)

### dnn classifier
print('initializing nn classifier...\n')

def binary_model(n_inputs, n_classes, n_hidden, hidden_nodes, input_dropout=0.0, biases=True):
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

def dec_score_comp(mod, x_train, x_test):
    # generate decision score histograms to calculate overtraining parameter...
    decis_train = keras_model.predict(x_train).ravel()
    decis_test = keras_model.predict(x_test).ravel()
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

keras_model = binary_model(len(X_train.columns), len(y_1hot_train[0]),
                           n_hidden=n_hidden_layers, hidden_nodes=hidden_layer_nodes,
                           input_dropout=dropout_frac)
print(keras_model.summary())

print('training nn classifier...\n')

epochs, train_accs, test_accs = [], [], []
eval_accs, eval_loss = [], []
train_loss, test_loss = [], []
sig_ot_comp, bkgd_ot_comp = [], []

for i in range(n_epochs):
    print("\nEPOCH " + str(i) + "/" + str(n_epochs))
    #if i > 0:
    history = keras_model.fit(X_train_scale, y_1hot_train, epochs=1, batch_size=100, verbose=1)

    epochs.append(i)
    eval_accs.append(history.history['accuracy'][-1])
    eval_loss.append(history.history['loss'][-1])
    loss, acc = keras_model.evaluate(X_train_scale, y_1hot_train, verbose=2)
    train_accs.append(acc)
    train_loss.append(loss)
    print("training --> loss = %0.4f, \t acc = %0.4f"%(loss, acc))
    loss, acc = keras_model.evaluate(X_test_scale, y_1hot_test, verbose=2)
    test_accs.append(acc)
    test_loss.append(loss)
    print("testing --> loss = %0.4f, \t acc = %0.4f"%(loss, acc))

    # sig_ot_comp.append(dec_score_comp(keras_model, X_train_scale[y_train>0.5], X_test_scale[y_test>0.5]))
    # bkgd_ot_comp.append(dec_score_comp(keras_model, X_train_scale[y_train<0.5], X_test_scale[y_test<0.5]))

print("")

## confusion matrix...
y_1hot_pred_train = keras_model.predict(X_train_scale)
y_pred_train = y_1hot_pred_train.argmax(axis=1)

y_1hot_pred = keras_model.predict(X_test_scale)
y_pred = y_1hot_pred.argmax(axis=1)

matrix_train = metrics.confusion_matrix(y_1hot_train.argmax(axis=1), y_1hot_pred_train.argmax(axis=1))
matrix_train = matrix_train.astype('float') / matrix_train.sum(axis=1)[:, np.newaxis]
matrix_test = metrics.confusion_matrix(y_1hot_test.argmax(axis=1), y_1hot_pred.argmax(axis=1))
matrix_test = matrix_test.astype('float') / matrix_test.sum(axis=1)[:, np.newaxis]

#dirp

# fpr_keras, tpr_keras, thresholds_keras = metrics.roc_curve(y_test, y_pred_keras)
# auc_keras = metrics.auc(fpr_keras, tpr_keras)
# print("auc score on test: %0.4f" % auc_keras)

fig = plt.figure(figsize=(11,7))
plt.subplot(2,2,1)
plt.plot(epochs, eval_accs, label='train, dropout=' + str(dropout_frac))
plt.plot(epochs, train_accs, label='train')
plt.plot(epochs, test_accs, label='test')
plt.title('model accuracy')
plt.legend(loc="lower right")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper left')

plt.subplot(2,2,2)
plt.plot(epochs, eval_loss, label='train, dropout=' + str(dropout_frac))
plt.plot(epochs, train_loss, label='train')
plt.plot(epochs, test_loss, label='test')
plt.title('loss (bin. cross-entropy)')
plt.legend(loc="upper right")
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')

ax = plt.subplot(2,2,3)
print('training results...')
plot_confusion_matrix(y_1hot_train.argmax(axis=1), y_1hot_pred_train.argmax(axis=1), class_labels, ax,
                          normalize=True,
                          title='confusion matrix, train')

print('\ntesting results...')
ax = plt.subplot(2,2,4)
plot_confusion_matrix(y_1hot_test.argmax(axis=1), y_1hot_pred.argmax(axis=1), class_labels, ax,
                          normalize=True,
                          title='confusion matrix, test')

# sig_decis_train = keras_model.predict(X_train_scale[y_train>0.5]).ravel()
# bkgd_decis_train = keras_model.predict(X_train_scale[y_train<=0.5]).ravel()
# sig_decis = keras_model.predict(X_test_scale[y_test>0.5]).ravel()
# bkgd_decis = keras_model.predict(X_test_scale[y_test<=0.5]).ravel()
#
# # decsion scores
# plt.subplot(3,2,3)
# #sig_decis_train *= np.amax(sig_decis) / np.amax(sig_decis_train)
# sig_counts_train, sig_bin_edges_train = np.histogram(sig_decis_train, 100, range=(0,1))
# sig_counts, sig_bin_edges = np.histogram(sig_decis, 100, range=(0,1))
# bkgd_counts_train, bkgd_bin_edges_train = np.histogram(bkgd_decis_train, 100, range=(0,1))
# bkgd_counts, bkgd_bin_edges = np.histogram(bkgd_decis, 100, range=(0,1))
#
# sig_bin_cents = (sig_bin_edges[:-1] + sig_bin_edges[1:])/2.
# bkgd_bin_cents = (bkgd_bin_edges[:-1] + bkgd_bin_edges[1:])/2.
#
# #sig_counts_train = np.amax(sig_counts)/np.amax(sig_counts_train) * sig_counts_train
# #bkgd_counts_train = np.amax(bkgd_counts)/np.amax(bkgd_counts_train) * bkgd_counts_train
# sig_counts_train = np.sum(sig_counts)/np.sum(sig_counts_train) * sig_counts_train
# bkgd_counts_train = np.sum(bkgd_counts)/np.sum(bkgd_counts_train) * bkgd_counts_train
#
# plt.hist(sig_decis, color='r', alpha=0.5, range=(0,1), bins=100,
#          label=r'$\Lambda \rightarrow p \mu^{-} \overline{\nu}$, test')
#          #log=True)
# plt.hist(bkgd_decis, color='b', alpha=0.5, range=(0,1), bins=100,
#          label=r'background (' + bkgd_type + '), test')
# plt.plot(sig_bin_cents, sig_counts_train, 'r.',
#          label=r'$\Lambda \rightarrow p \mu^{-} \overline{\nu}$, train (scaled)', markersize=3)
# plt.plot(bkgd_bin_cents, bkgd_counts_train, 'b.',
#          label=r'background (' + bkgd_type + '), train (scaled)', markersize=3)
# plt.title('decision function')
# plt.legend(loc="upper center")
# plt.xlabel(r'NN output')
# plt.tight_layout
# #plt.show()
#
# # ROC plot
# roc_auc = metrics.auc(fpr_keras, tpr_keras)
# plt.subplot(3,2,4)
# plt.plot(fpr_keras, tpr_keras, lw=1, label='ROC (area = %0.3f)'%(roc_auc))
# plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6)) #, label='luck')
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('false positive rate')
# plt.ylabel('true positive rate')
# plt.title('receiver operating characteristic')
# plt.legend(loc="lower right")
# plt.grid()
#
# # decision score comparison, overtraining
# plt.subplot(3,2,5)
# plt.plot(epochs, sig_ot_comp, label='signal')
# plt.plot(epochs, bkgd_ot_comp, label='bkgd')
# plt.xlabel('epochs')
# plt.ylabel('decision profile diff.')

plt.tight_layout()
plt.show()

# # save the model and weights
# if write_model_files:
#     file_str = './models/' + 'bkgd:' + bkgd_type + '_kf:' + kf_type + '_'
#     file_str += 'neps:' + str(n_epochs) + '_layers:'
#     for i in range(len(hidden_layer_nodes)):
#         file_str += str(hidden_layer_nodes[i])
#         if i != len(hidden_layer_nodes) - 1:
#             file_str += '.'
#     file_str += '_do:' + ("%0.2f" % dropout_frac)
#     file_str += '_auc:' + '%0.3f' % roc_auc
#
#     print(file_str)
#
#     model_json = keras_model.to_json()
#     with open(file_str + ".json", "w") as json_file:
#         json_file.write(model_json)
#
#     keras_model.save_weights(file_str + ".h5")
#     print("Saved model to disk")
#
#     fig.savefig(file_str + '.png')
