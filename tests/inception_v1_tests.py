import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, one_hot_encoding, flatten
import time
from sys import path
path.insert(0, '../models/')
path.insert(0, '../')
import flags
import data_utils
import loss
from InceptionLayers import InceptionStemV1, InceptionModuleV1, InceptionClassifierV1, InceptionV1

def run_module_unit_test(use_fake_data=False, test_mode="full_model"):
    #Test mode can either be "module", "stem", "classifier_auxiliary", "classifier_basic" or "full_model"
    fl = tf.app.flags.FLAGS
    BATCH_SIZE = fl.batch_size
    L2_WEIGHT = fl.l2_lambda_weight
    
    if use_fake_data:
        #load fake data. imagenet uses 224*224*3, but put whatever you want here.
        train_X = np.random.rand(BATCH_SIZE*5, 224, 224, 3)
        train_y = np.random.randint(low=0, high=1000, size=(BATCH_SIZE*5, 1))
        NUM_LABELS = 1000
        test_X = np.random.rand(BATCH_SIZE, 224, 224, 3)
        test_y = np.random.randint(low=0, high=1000, size=(BATCH_SIZE*4, 1))
    else:
        #TODO - toss away this NUM_LABELS when done testing
        (train_X, train_y), (test_X, test_y), NUM_LABELS = data_utils.load_dataset(fl.dataset)

    #extract a random validation set from the training set
    validation_size = np.floor(train_X.shape[0]*fl.validation_ratio).astype(int)
    shuf = np.random.permutation(train_X.shape[0])
    train_X = train_X[shuf]
    train_y = train_y[shuf]
    validation_X, validation_y = train_X[:validation_size], train_y[:validation_size]
    train_X, train_y = train_X[validation_size:], train_y[validation_size:]
    
    IMAGE_LEN = train_X.shape[1]
    IMAGE_WID = train_X.shape[2]
    IMAGE_SIZE = IMAGE_LEN
    NUM_CHANNELS = train_X.shape[3]

    g = tf.Graph()
    with g.as_default():
        tf_train_X = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        tf_train_y = tf.placeholder(tf.int32, shape=(BATCH_SIZE,1))
        tf_validation_X = tf.placeholder(tf.float32, shape=validation_X.shape)
        tf_validation_y = tf.placeholder(tf.int32, shape=(validation_y.shape[0],1))
        tf_test_X = tf.placeholder(tf.float32, shape=test_X.shape)
        tf_test_y = tf.placeholder(tf.int32, shape=(test_y.shape[0],1))
        
        if test_mode == "stem":
            expected_output_shape = [BATCH_SIZE, IMAGE_SIZE/8, IMAGE_SIZE/8, 192] #three times the length and width of an image are halved, hence the 8.
            inception_model = InceptionStemV1(filter_sizes=[64, 64, 192],
                                            input_shape=tf_train_X.get_shape(),
                                            output_shape=expected_output_shape,
                                            scope="stem1")
        
        elif test_mode == "module":
            expected_output_shape = [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 64]
            inception_model = InceptionModuleV1(dtype = tf.float32,
                                        input_shape = tf_train_X.get_shape().as_list(),
                                        output_shape = expected_output_shape,
                                        filter_sizes = [16, 24, 32, 4, 8, 8], #indexes 0,2,4,5 must add up to output[-1]
                                        scope="module1")
        
        elif test_mode == "classifier_auxiliary":
            expected_output_shape = [BATCH_SIZE, NUM_LABELS]
            inception_model = InceptionClassifierV1(dtype=tf.float32,
                                        auxiliary_weight_constant=0.3,
                                        filter_sizes=[10,1024],
                                        auxiliary_classifier=True,
                                        input_shape=tf_train_X.get_shape().as_list(), 
                                        output_shape=expected_output_shape,
                                        scope="classifier_auxiliary1")
            
        elif test_mode == "classifier_basic":
            expected_output_shape = [BATCH_SIZE, NUM_LABELS]
            inception_model = InceptionClassifierV1(dtype=tf.float32,
                                        input_shape=tf_train_X.get_shape().as_list(),
                                        output_shape=expected_output_shape,
                                        scope="classifier_basic1")
           
        elif test_mode == "full_model":
            expected_output_shape = [BATCH_SIZE, NUM_LABELS]
            inception_model = InceptionV1(dtype=tf.float32,
                                          filter_size_dict="imagenet_auto",
                                          filter_size_reduction_factor=4,
                                          auxiliary_classifier_weights=[0.3,0.3],
                                          use_mini_model=False,
                                          input_shape=tf_train_X.get_shape().as_list(), #224x224x3 imagenet images
                                          output_shape=expected_output_shape, 
                                          scope="inception1")
            
        
        inception_model.create_model()
        global_step = tf.Variable(0)
        
        #set up a learning rate and learning rate decay mechanism
        lr_calc = tf.train.exponential_decay(0.01, global_step, 100, 0.999, staircase=True)
        lr_min = 0.0001
        lr = tf.maximum(lr_calc, lr_min)
        
        #set up an l2 regulariztaion and its decay mechanism operation
        l2_lambda_weight = tf.Variable(fl.l2_lambda_weight, dtype=tf.float32)
        l2_lambda_decay = tf.constant(fl.l2_lambda_weight_decay, dtype=tf.float32)
        l2_lambda_decay_op = l2_lambda_weight.assign(
            l2_lambda_weight * l2_lambda_decay)
        
        #reshape the images and their labels
        #flat_inputs = flatten(tf_train_X, scope="flatten_pixel_channels")
        one_hot_train_outputs = one_hot_encoding(tf.squeeze(tf_train_y), NUM_LABELS, on_value=1.0, off_value=0.0)
        one_hot_validation_outputs = one_hot_encoding(tf.squeeze(tf_validation_y), NUM_LABELS, on_value=1.0, off_value=0.0)
        one_hot_test_outputs = one_hot_encoding(tf.squeeze(tf_test_y), NUM_LABELS, on_value=1.0, off_value=0.0)

        #A cheap model that tosses a fully-connected layer on to the flattened result of the 4d Tensor
        
        if test_mode in ["module", "stem"]: #not testing any classification, so we build a dummy FC layer to connect to logits
            
            flattened_incept_out_size = expected_output_shape[1]*expected_output_shape[2]*expected_output_shape[3]
            
            w_l2 = tf.get_variable("w_l2",
                           shape=(flattened_incept_out_size, one_hot_train_outputs.get_shape()[1]),
                           dtype=tf.float32,
                           initializer=xavier_initializer())
            b_l2 = tf.get_variable("b_l2",
                           shape=(one_hot_train_outputs.get_shape()[1]),
                           dtype=tf.float32,
                           initializer=tf.zeros_initializer())
            
            def model_with_linear_classifier(inp, training=True):
                inception_out = inception_model.run_model(inp)
                flat_inputs = flatten(inception_out)
                return tf.matmul(flat_inputs, w_l2) + b_l2
            
            train_out = model_with_linear_classifier(tf_train_X)
            train_predictions = tf.nn.softmax(train_out)
            validation_out = model_with_linear_classifier(tf_validation_X, training=False)
            validation_predictions = tf.nn.softmax(validation_out)
            test_out = model_with_linear_classifier(tf_test_X, training=False)
            test_predictions = tf.nn.softmax(test_out)
        
        #using a model with a classifier in it
        else:
            train_out = inception_model.run_model(tf_train_X, training=True)
            train_predictions = tf.nn.softmax(train_out)
            validation_out = inception_model.run_model(tf_validation_X, training=False)
            validation_predictions = tf.nn.softmax(validation_out)
            test_out = inception_model.run_model(tf_test_X, training=False)
            test_predictions = tf.nn.softmax(test_out)
        
        #separate the losses so we can compare them in the session
        ce_loss = loss.softmax_cross_entropy_with_laplace_smoothing(train_out, one_hot_train_outputs, laplace_pseudocount=0.00001, scale=[0.3,0.3,1.0] if test_mode=='full_model' else 1.0)
        
        #collect all the parameters in the model to do l2 regulariztion
        regularization_parameters = inception_model.model_parameters
        if test_mode in ["module", "stem"]:
            regularization_parameters.extend((w_l2, b_l2))
        
        reg_loss = loss.regularizer(regularization_parameters, reg_type='l2', weight_lambda=0.001)
        
        total_loss = tf.reduce_mean(ce_loss + reg_loss)
        opt = tf.train.GradientDescentOptimizer(lr).minimize(total_loss, global_step=global_step)
        
        #we also declare this in the graph and run it in the session
        init_op = tf.global_variables_initializer()
        
    with tf.Session(graph=g, config=tf.ConfigProto(log_device_placement=True)) as sess:
    
        sess.run(init_op)
        total_steps = 0
        num_epochs = 100

        for epoch in range(num_epochs):
            shuf = np.random.permutation(train_X.shape[0])
            train_X = train_X[shuf]
            train_y = train_y[shuf]
            processed=0

            while processed+BATCH_SIZE <= train_X.shape[0]:
                batch_X = train_X[processed:processed+BATCH_SIZE]
                batch_y = train_y[processed:processed+BATCH_SIZE]
                processed += BATCH_SIZE

                feed_dict = {tf_train_X:batch_X,
                            tf_train_y:batch_y}

                _, l, rl, pred, l2lw = sess.run([opt, total_loss, reg_loss, train_predictions, l2_lambda_weight], feed_dict=feed_dict)
                total_steps += 1
                
                if total_steps % fl.l2_lambda_weight_decay_steps == 0:
                    sess.run(l2_lambda_decay_op)
                
                #Validation Set
                if total_steps % fl.validation_frequency == 0:
                    feed_dict = {tf_validation_X:validation_X,
                                 tf_validation_y:validation_y}
                    pred_labels, true_labels = sess.run([validation_predictions, one_hot_validation_outputs], feed_dict=feed_dict)
                    print("Validation Top-1 accuracy is " + str(100.0*data_utils.n_accuracy(pred_labels, true_labels, 1)) + "%")
        
        #Test Set
        feed_dict = {tf_test_X:test_X,tf_test_y:test_y}
        pred_labels, true_labels = sess.run([test_predictions, one_hot_test_outputs], feed_dict=feed_dict)
        print("Test Top-1 accuracy is " + str(100.0*data_utils.n_accuracy(pred_labels, true_labels, 1)) + "%")       