# coding: utf-8

import tensorflow as tf

def regularizer(tensor_list, reg_type='l2', weight_lambda=1.0, scope=None):
    """
    Creates a function that runs a tensorflow l1 or l2 regularizer for a weight tensor and returns the value
    
    Args:
        t_list - list of tensors - what we are regularizing
        reg-type - string, in 'l1' or 'l2'
        weight_lambda - positive float for how much to multiply the regularization term by in total network loss
        scope - string, tensorflow parent scope
        
    Returns:
        regularizer - the function defined below that itself returns the result of the final tf.multiply
         operation when running l1 or l2 loss on a list of weight tensors.
    
    Raises:
        AttributeError - if the regularizer type isn't 'l1' or 'l2'
    """
    
    if reg_type not in ['l1', 'l2']:
        raise AttributeError, "reg type needs to be either 'l1' or 'l2'"
    
    with tf.name_scope(scope, 'regularizer', [tensor for tensor in tensor_list]):
        w_lambda = tf.convert_to_tensor(weight_lambda,
                                   dtype=tensor_list[0].dtype.base_dtype,
                                   name="reg_scale")
        if reg_type=='l2':
            losses = [tf.nn.l2_loss(t) for t in tensor_list]
        elif reg_type=='l1':
            losses = [tf.reduce_sum(tf.abs(t)) for t in tensor_list]
        
        loss = tf.multiply(w_lambda, tf.add_n(losses), name="reg_val")
       
        return loss

def softmax_cross_entropy_with_laplace_smoothing(logits, one_hot_labels, laplace_pseudocount=0., scale=1.0, scope=None):
    """Creates and returns a softmax cross entropy from logits to one hot labels with additive smoothing. Scales
    by an optional scaler. 
    
    Args:
     logits - Tensor. Shape is [batch_size, num_classes]. Alternatively, a list of tensors, each with shape [batch_size, num_classes]. this is necessary because often we have models with multiple classifiers.
     one_hot_labels - Tensor. Shape is [batch_size, num_classes]
     laplace_psuedocount - float between 0 and 1
     scale - float, scale total loss by this factor. Alternatively, a list of floats that is the same length as the logits argument list.
     scope - optional tensorflow parent scope
    
    Returns: Tensor containing the softmax cross entropy loss
    """
    
    if logits.__class__.__name__=='Tensor':
        logits = [logits]
        if isinstance(scale, list):
            assert len(scale)==1, "If using a single Tensor as logits argument, scale must be a single float or a list with a single float."
        else:
            scale = [scale]
    else:
        assert isinstance(logits, list), "logits argument must be Tensor class from tensorflow, a list of Tensors"
        assert isinstance(scale, list), "if using a list of Tensors as logits argument, you must pass a list of scales."
        assert len(scale)==len(logits), "You must have a scale for each Tensor in the logits argument."
    
    scale = [tf.convert_to_tensor(w,
                                  dtype=logits[0].dtype.base_dtype,
                                  name='ce_loss_w_%d'%idx) for idx,w in enumerate(scale)]
        
    assert laplace_pseudocount < 1. and laplace_pseudocount > 0., "pseudocount needs to be between 0 and 1"
    
    logits[0].get_shape().assert_is_compatible_with(one_hot_labels.get_shape())
    
    with tf.name_scope(scope, 'softmax_cross_entropy_loss', [logits, one_hot_labels, scale]):
        one_hot_labels = tf.cast(one_hot_labels, logits[0].dtype)
        
        positives = 1.0 - laplace_pseudocount
        negatives = laplace_pseudocount / float(one_hot_labels.get_shape()[-1].value) # num classes
        
        one_hot_labels = one_hot_labels * positives + negatives
        
        #inception models have multiple classifiers, so we weight all their logits
        losses = []
        print scale
        for idx, (classifier_logits, classifier_weight) in enumerate(zip(logits, scale)):
            ce = tf.nn.softmax_cross_entropy_with_logits(logits=classifier_logits, labels=one_hot_labels, name="cross_entropy_loss_%d" % idx)
            losses.append(tf.multiply(classifier_weight, tf.reduce_mean(ce), name="ce_value_%d" % idx))
        return tf.add_n(losses, "weighted_classifier_loss")