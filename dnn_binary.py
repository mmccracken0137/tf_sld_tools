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
import json
import sys

# to use latex with matplotlib
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# suppresses an annoying tf warning...
#tf.logging.set_verbosity(tf.logging.ERROR)

# hyper-parameters and data parameters
# read in from json config file supplied on command line
with open(sys.argv[1], 'r') as f:
    config = json.load(f)

write_model_files = config['RUN']['WRITE_MODEL_FILES']
bkgd_type = config['DATA']['BKGD_TYPE']
data_tag = config['DATA']['DATA_TAG']
kf_type = config['DATA']['KF_TYPE']
kf_tag = ''
if kf_type == 'p4only':
    kf_tag = '_p4only'
drop_non_p4x4 = config['DATA']['DROP_NON_P4X4']

hidden_layer_nodes = config['MODEL']['HIDDEN_LAYER_NODES']
n_hidden_layers = len(hidden_layer_nodes)
dropout_frac = config['MODEL']['DROPOUT_FRAC']
n_epochs = config['RUN']['N_EPOCHS']

data_dir = config['DATA']['DATA_DIR']
data_types = config['DATA']['DATA_TYPES']

# read in feats files
print('\nreading data files...\n')
df, class_labels = get_dfs(data_types, data_dir, data_tag)
df = pd.concat(df)

### DROP higher beam energies
# df = df[df.beam_E < 5.0]

print("")
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
df.drop(['event', 'sig_label', 'type_label'], axis=1, inplace=True)

to_drop = []
for f in df.columns:
    if 'meas' in f:
        to_drop.append(f)
#print(to_drop)
df.drop(to_drop, inplace=True, axis=1)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=42)

#print(X_train.shape, X_test.shape)

#print(X_train.head())

print('fitting sklearn.preprocessing.StandardScaler to X_train...')
scaler = preprocessing.StandardScaler().fit(X_train)
# print(scaler.mean_)
# print(scaler.scale_)

print('applying scaler to train data...\n')
X_train_scale = scaler.transform(X_train)
X_test_scale = scaler.transform(X_test)

### dnn classifier
print('initializing nn classifier...\n')

keras_model = binary_model(len(X_train.columns), n_hidden=n_hidden_layers,
                           hidden_nodes=hidden_layer_nodes,
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
    history = keras_model.fit(X_train_scale, y_train, epochs=1, batch_size=100, verbose=1)

    epochs.append(i)
    eval_accs.append(history.history['accuracy'][-1])
    eval_loss.append(history.history['loss'][-1])
    loss, acc = keras_model.evaluate(X_train_scale, y_train, verbose=2)
    train_accs.append(acc)
    train_loss.append(loss)
    print("training --> loss = %0.4f, \t acc = %0.4f"%(loss, acc))
    loss, acc = keras_model.evaluate(X_test_scale, y_test, verbose=2)
    test_accs.append(acc)
    test_loss.append(loss)
    print("testing --> loss = %0.4f, \t acc = %0.4f"%(loss, acc))

    sig_ot_comp.append(dec_score_comp(keras_model, X_train_scale[y_train>0.5], X_test_scale[y_test>0.5]))
    bkgd_ot_comp.append(dec_score_comp(keras_model, X_train_scale[y_train<0.5], X_test_scale[y_test<0.5]))

print("")
y_pred_keras = keras_model.predict(X_test_scale).ravel()
fpr_keras, tpr_keras, thresholds_keras = metrics.roc_curve(y_test, y_pred_keras)
auc_keras = metrics.auc(fpr_keras, tpr_keras)
print("auc score on test: %0.4f" % auc_keras)

fig = plt.figure(figsize=(11,7))

plt.subplot(3,2,1)
plt.plot(epochs, eval_accs, label='train, dropout=' + str(dropout_frac))
plt.plot(epochs, train_accs, label='train')
plt.plot(epochs, test_accs, label='test')
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.legend(loc="lower right")
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.subplot(3,2,2)
plt.plot(epochs, eval_loss, label='train, dropout=' + str(dropout_frac))
plt.plot(epochs, train_loss, label='train')
plt.plot(epochs, test_loss, label='test')
#plt.plot(history.history['val_acc'])
plt.title('loss (bin. cross-entropy)')
plt.legend(loc="upper right")
plt.ylabel('loss')
plt.xlabel('epoch')

# decsion scores
plt.subplot(3,2,3)
sig_decis_train = keras_model.predict(X_train_scale[y_train>0.5]).ravel()
bkgd_decis_train = keras_model.predict(X_train_scale[y_train<=0.5]).ravel()
sig_decis = keras_model.predict(X_test_scale[y_test>0.5]).ravel()
bkgd_decis = keras_model.predict(X_test_scale[y_test<=0.5]).ravel()
#sig_decis_train *= np.amax(sig_decis) / np.amax(sig_decis_train)
sig_counts_train, sig_bin_edges_train = np.histogram(sig_decis_train, 100, range=(0,1))
sig_counts, sig_bin_edges = np.histogram(sig_decis, 100, range=(0,1))
bkgd_counts_train, bkgd_bin_edges_train = np.histogram(bkgd_decis_train, 100, range=(0,1))
bkgd_counts, bkgd_bin_edges = np.histogram(bkgd_decis, 100, range=(0,1))

sig_bin_cents = (sig_bin_edges[:-1] + sig_bin_edges[1:])/2.
bkgd_bin_cents = (bkgd_bin_edges[:-1] + bkgd_bin_edges[1:])/2.

#sig_counts_train = np.amax(sig_counts)/np.amax(sig_counts_train) * sig_counts_train
#bkgd_counts_train = np.amax(bkgd_counts)/np.amax(bkgd_counts_train) * bkgd_counts_train
sig_counts_train = np.sum(sig_counts)/np.sum(sig_counts_train) * sig_counts_train
bkgd_counts_train = np.sum(bkgd_counts)/np.sum(bkgd_counts_train) * bkgd_counts_train

plt.hist(sig_decis, color='r', alpha=0.5, range=(0,1), bins=100,
         label=r'$\Lambda \rightarrow p \mu^{-} \overline{\nu}$, test')
         #log=True)
plt.hist(bkgd_decis, color='b', alpha=0.5, range=(0,1), bins=100,
         label=r'background (' + bkgd_type + '), test')
plt.plot(sig_bin_cents, sig_counts_train, 'r.',
         label=r'$\Lambda \rightarrow p \mu^{-} \overline{\nu}$, train (scaled)', markersize=3)
plt.plot(bkgd_bin_cents, bkgd_counts_train, 'b.',
         label=r'background (' + bkgd_type + '), train (scaled)', markersize=3)
plt.title('decision function')
plt.legend(loc="upper center")
plt.xlabel(r'NN output')
plt.tight_layout
#plt.show()

# ROC plot
roc_auc = metrics.auc(fpr_keras, tpr_keras)
plt.subplot(3,2,4)
plt.plot(fpr_keras, tpr_keras, lw=1, label='ROC (area = %0.3f)'%(roc_auc))
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6)) #, label='luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('receiver operating characteristic')
plt.legend(loc="lower right")
plt.grid()

# decision score comparison, overtraining
plt.subplot(3,2,5)
plt.plot(epochs, sig_ot_comp, label='signal')
plt.plot(epochs, bkgd_ot_comp, label='bkgd')
plt.xlabel('epochs')
plt.ylabel('decision profile diff.')

plt.tight_layout()
plt.show()

# save the model and weights
if write_model_files:
    data_str = ':'.join(data_types)
    file_str = './binary_models/' + 'types:' + data_str + '_kf:' + kf_type + '_'
    file_str += 'neps:' + str(n_epochs) + '_layers:'
    for i in range(len(hidden_layer_nodes)):
        file_str += str(hidden_layer_nodes[i])
        if i != len(hidden_layer_nodes) - 1:
            file_str += '.'
    file_str += '_do:' + ("%0.2f" % dropout_frac)
    file_str += '_auc:' + '%0.3f' % roc_auc

    print(file_str)

    model_json = keras_model.to_json()
    with open(file_str + ".json", "w") as json_file:
        json_file.write(model_json)

    keras_model.save_weights(file_str + ".h5")
    print("Saved model to disk")

    fig.savefig(file_str + '.png')
