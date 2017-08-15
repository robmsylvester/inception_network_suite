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
    by an optional scaler
    
    Args:
     logits - Tensor. Shape is [batch_size, num_classes]
     one_hot_labels - Tensor. Shape is [batch_size, num_classes]
     laplace_psuedocount - float between 0 and 1
     scale - float, scale total loss by this factor
     scope - optional tensorflow parent scope
    
    Returns: Tensor containing the softmax cross entropy loss
    """
    
    scale = tf.convert_to_tensor(scale,
                                dtype=logits.dtype.base_dtype,
                                name='ce_loss_scale')
    
    assert laplace_pseudocount < 1. and laplace_pseudocount > 0., "pseudocount needs to be between 0 and 1"
    
    logits.get_shape().assert_is_compatible_with(one_hot_labels.get_shape())
    
    with tf.name_scope(scope, 'softmax_cross_entropy_loss', [logits, one_hot_labels]):
        one_hot_labels = tf.cast(one_hot_labels, logits.dtype)
        
        positives = 1.0 - laplace_pseudocount
        negatives = laplace_pseudocount / float(one_hot_labels.get_shape()[-1].value) # num classes
        
        one_hot_labels = one_hot_labels * positives + negatives
        
        ce = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels, name="cross_entropy_loss")
        
        loss = tf.multiply(scale, tf.reduce_mean(ce), name="ce_value")
        return loss
        
        


# In[ ]:



