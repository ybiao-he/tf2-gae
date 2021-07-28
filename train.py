from __future__ import division
from __future__ import print_function

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from input_data import load_data
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

from model import GCNModelAE, GCNModelVAE

import os
import time

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

def get_roc_score(model, edges_pos, edges_neg, features, adj_norm, emb=None):
    if emb is None:
        emb = model.embeddings([features, adj_norm])
        emb = emb.numpy()

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Settings
learning_rate = 0.01
epochs = 200
hidden1 = 32
hidden2 = 16
weight_decay = 0
dropout = 0
model_str = 'gcn_ae'
dataset_str = 'cora'
use_features = 1

# Load data
adj, features = load_data(dataset_str)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

if use_features == 0:
    features = sp.identity(features.shape[0])  # featureless

# Some preprocessing
adj_norm = preprocess_graph(adj)
num_nodes = adj.shape[0]

num_features = features.shape[-1]
features_nonzero = features.nnz

# Create model
model = GCNModelVAE(num_features, num_nodes, features_nonzero)

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

features = convert_sparse_matrix_to_sparse_tensor(features)
adj_norm = convert_sparse_matrix_to_sparse_tensor(adj_norm)

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = convert_sparse_matrix_to_sparse_tensor(adj_label)
adj_label = tf.reshape(tf.sparse.to_dense(adj_label,
                                          validate_indices=False), [-1])
adj_label = tf.cast(adj_label, tf.float32)


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
epochs = 200

val_roc_score = []

# Iterate over epochs.
for epoch in range(epochs):
    print("Start of epoch %d" % (epoch,))
    t = time.time()
    with tf.GradientTape() as tape:
        reconstructed = model([features, adj_norm])
        loss = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=adj_label, 
                                                                              logits=reconstructed, 
                                                                              pos_weight=pos_weight))
    
    correct_prediction = tf.math.equal(
                         tf.cast(tf.math.greater_equal(tf.math.sigmoid(reconstructed), 0.5), tf.int32),
                                 tf.cast(adj_label, tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("The loss is: {}; The accuracy is: {}.".format(loss.numpy(), accuracy.numpy()))
    
    roc_curr, ap_curr = get_roc_score(model, val_edges, val_edges_false, features, adj_norm)
    val_roc_score.append(roc_curr)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss),
      "train_acc=", "{:.5f}".format(accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
      "val_ap=", "{:.5f}".format(ap_curr),
      "time=", "{:.5f}".format(time.time() - t))
    
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

roc_score, ap_score = get_roc_score(model, val_edges, val_edges_false, features, adj_norm)
print('Test ROC score: ' + str(roc_score))
print('Test AP score: ' + str(ap_score))
