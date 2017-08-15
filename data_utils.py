# coding: utf-8

import numpy as np
import os
from six.moves import cPickle as pickle

def one_hot(batch_of_labels, num_classes, scope=None):
    with tf.name_scope(scope, 'one_hot', [batch_of_labels]):
        batch_size = labels.get_shape()[0]
        indices = tf.expand_dims(tf.range(0, batch_size), 1)
        labels = tf.cast(tf.expand_dims(labels, 1), indices.dtype)
        concatenated_labels = tf.concat(axis=1, values=[indices, labels])
        onehot_labels = tf.sparse_to_dense(
            concatenated_labels, tf.stack([batch_size, num_classes]), 1.0, 0.0)
        onehot_labels.set_shape([batch_size, num_classes])
        return onehot_labels

def n_accuracy(logits, labels, n):
    """
    Reports the top-n accuracy of a set of logits vs. the true-labels.
    For example, if n is 5, then this function will return how often a given
    set of logits has a distribution such that the true label is within the
    highest five values for the logit
    """

    if n==1:
        return np.sum(np.argmax(labels,1) == np.argmax(logits,1)) / float(labels.shape[0])
    else:
        raise NotImplementedError, "TODO - finish this"
        

def load_cifar100_dataset(dirname, labels='fine', transpose_permutation=(0,2,3,1)):
    """Loads the dataset as a tuple of numpy arrays.
    
    Args:
        dirname - file location of extracted cifar100 tarball
        labels - string, either 'fine' or 'coarse'
        transpose_permuation - list of 4 integers. 0,2,3,1 gives a size,32,32,3. 0,1,2,3 gives size,3,32,32 
    
    Returns:
        tuple - ( (x train, y train, (x test, y test) )
    
    Raises:
        ioerror - no file location for dirname
        attributeerror - labels not 'coarse' or 'fine'. transpose_permutation not len(4) and integers
    
    """
    
    #Verify paths exists for training and testing set
    if not os.path.exists(dirname):
        raise IOError, "Cannot find path %s" % path
    
    if labels not in ['fine', 'coarse']:
        raise AttributeError, "Labels argument must be set to 'coarse' or 'fine'"
        
    if len(set(transpose_permutation)) != 4:
        raise AttributeError, "Expect transpose permutation to be "

    full_path = os.path.abspath(dirname)
    
    train_path = os.path.join(full_path, 'train')
    test_path = os.path.join(full_path, 'test')
    
    #Load the training set
    with open(train_path, 'rb') as tr_f:
        tr_data_raw = pickle.load(tr_f)
        tr_data = {}
        
        for key, val in tr_data_raw.items():
            tr_data[key.decode('utf8')] = val #32 x 32 x 3 images.
    
    tr_X = tr_data['data']
    
    if labels=='fine':
        tr_y = tr_data['fine_labels']
    elif labels=='coarse':
        tr_y = tr_data['coarse_labels']
    
    tr_X = tr_X.reshape(tr_X.shape[0], 3, 32, 32)
    tr_y = np.reshape(tr_y, (len(tr_y), 1))
    
    #Load the testing set
    with open(test_path, 'rb') as te_f:
        te_data_raw = pickle.load(te_f)
        te_data = {}
        
        for key, val in te_data_raw.items():
            te_data[key.decode('utf8')] = val #32 x 32 x 3 images.
    
    te_X = te_data['data']
    
    if labels=='fine':
        te_y = te_data['fine_labels']
    elif labels=='coarse':
        te_y = te_data['coarse_labels']
    
    te_X = te_X.reshape(te_X.shape[0], 3, 32, 32)
    te_y = np.reshape(te_y, (len(te_y), 1))
    
    #scale to 255, transpose as needed
    tr_X = np.transpose(tr_X.astype('float32') / 255., transpose_permutation)
    te_X = np.transpose(te_X.astype('float32') / 255., transpose_permutation)
    
    return (tr_X, tr_y), (te_X, te_y), 100

def load_dataset(dataset_name):
    if dataset_name == 'cifar100':
        return load_cifar100_dataset("data/cifar-100-python", labels='fine')
    elif dataset_name == None:
        raise NotImplementedError, "Add support for custom image dataset %s in data_utils..." % dataset_name
    else:
        raise NotImplementedError, "No other datasets naively supported for dataset_name %s" % dataset_name
