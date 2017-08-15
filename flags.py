import tensorflow as tf
import os
import sys

#==========================Dataset Parameters====================================
tf.app.flags.DEFINE_string("dataset", 'cifar100',
                          "Supported datasets for inception experiments. If using custom images, enter None")

#=========================Validation Parameters==================================
tf.app.flags.DEFINE_float("validation_ratio", 0.2,
                          "The ratio, between 0 and 1, of the TRAINING set to use as validation data.")
tf.app.flags.DEFINE_integer("validation_frequency", 2000,
                            "The number of training iterations per session call to the validation set")

#==========================Model Parameters======================================
tf.app.flags.DEFINE_string("inception_model", 'googlenet',
                           "The inception model to use on your dataset. must be v3 or v4. resnet support later")

tf.app.flags.DEFINE_integer("batch_size", 64,
                            "The number of images to feed per training step.")

tf.app.flags.DEFINE_float("l2_lambda_weight", 0.001,
                          "The initial weight to use as a multiplier to the L2 sum. This can be decayed over training.")
                          
tf.app.flags.DEFINE_float("l2_lambda_weight_decay", 0.999,
                          "Every decay_steps number of training steps, the current lambda weight will be multiplied by this number.")
tf.app.flags.DEFINE_integer("l2_lambda_weight_decay_steps", 1000,
                          "The number of weight decay steps before running a lambda weight decay operation.")

def flag_test():
    supported_datasets = ['cifar100', None]
    assert tf.app.flags.FLAGS.dataset in supported_datasets, "Your dataset must be cifar100, imagenet, or None"
    
    supported_inception_models = ['googlenet', 'v3', 'v4']
    assert tf.app.flags.FLAGS.inception_model in supported_inception_model, "Your inception model flag must be 'v3' or 'v4'"
                          
    assert tf.app.flags.FLAGS.batch_size > 0, "Batch size must be a positive integer"
    
    assert tf.app.flags.FLAGS.validation_ratio is None or (tf.app.flags.FLAGS.validation_ratio >= 0. and tf.app.flags.FLAGS.validation_ratio <= 1.), "Validation Set Ratio must be between 0 and 1, or None to be turned off"
    
    assert tf.app.flags.FLAGS.l2_lambda_weight > 0, "l2 lambda weight must be positive"
    assert tf.app.flags.FLAGS.l2_lambda_weight_decay > 0. and tf.app.flags.FLAGS.l2_lambda_weight_decay <= 1., "Lambda weight decay for L2 must be within the range (0, 1.]"
    assert tf.app.flags.FLAGS.l2_lambda_weight_decay_steps > 0, "l2 lambda weight decay steps must be positive"
                         