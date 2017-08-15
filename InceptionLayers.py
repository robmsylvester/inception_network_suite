import tensorflow as tf
import numpy as np

#TODO - input shape is still default None keyword argument but that throws error, so make it required explicit tensor shape

#TODO - move standard_model_width_reduction=None arg to inception model class
class InceptionModuleV1:
    """This class implements the basic building block layer of inception v1 modules for object
    recognition. It runs a 1x1, 3x3, and 5x5 convolutions as well as max pooling operations to a single
    input before concatenating the outputs along the depth axis. At least, that is one version of the inception
    architecure. There are several versions of inception that have come out over the years in various papers
    
    v1: https://arxiv.org/pdf/1409.4842.pdf (using dimension reduction model, figure 2b)
        Essentially, this builds GoogLeNet inception modules after the stem is built.
    
    Returns:
    Depth_Concatenated_Output - The tensor outputs from the different convolutions in the module are
    concatenated along this final depth axis
    """
    
    def __init__(self,
                 layer_number, #TODO - this can go. pass scope in instead for naming module chains
                 filter_sizes=None,
                 dtype=tf.float32,
                 n_factorization=None, #TODO - useless in V1
                 input_shape=None,
                 output_shape=None,
                 verify_shape=True):
        """
        Creates the Tensorflow parameters for an inception v1 module network and checks the network
        filter sizes as well as the input shape for any arithmetic inconsistencies that
        would arise.
        
        Args:
        filter_sizes: this is a python tuple of SIX integer filter sizes in the follwing order:

                      filter_sizes[0] = filter size of 1x1 convolution dimensionality reduction. tower 1
                      filter_sizes[1] = filter size of 1x1 convolution before 3x3 convolution. tower 2.
                      filter_sizes[2] = filter size of 3x3 convolution. tower 2
                      filter_sizes[3] = filter size of 1x1 convolution before 5x5 convolution. tower 3
                      filter_sizes[4] = filter size of 5x5 convolution. tower 3
                      filter_sizes[5] = filter size of pool projection after 1x1 convolution. tower 4
        
        layer_number - positive integer, just to keep straight the modules. also used in scope naming variables for easier global access    
        
        input_shape - a tuple of integers specifying the network input shape. If passed, expects a 4-d list
                    such that the batch size is the first integer, then length, then width, then depth.
                    if None, then no checks will be done on input size and you will catch the errors
                    at runtime when the TF Graph is created. Batch size can be whatever.
                    
                    tensor.get_shape().as_list() will turn your tensor shape into a list.
        
        n_factorization - positive integer, if using a v2 inception with factorization (https://arxiv.org/pdf/1512.00567v3.pdf),
                          this is the value for n.
        
        output_shape - Tuple of 4 integers. Expected shape of the output that can be used to verify
        verify_shape - boolean. If true, verify the filter sizes now instead of in tensorflow graph
        """
        
        #Auto-size ratio assumes that your image inputs come in as 28x28
        
        
        #If shape is a tensorflow shape object instead of a list, implicitly convert it
        if input_shape.__class__.__name__ == 'TensorShape':
            input_shape = input_shape.as_list()
        if output_shape.__class__.__name__ == 'TensorShape':
            output_shape = output_shape.as_list()
        
        if input_shape is not None:
            assert len(input_shape) == 4, "Input must be 4-dimensional. (batch_size, len, width, depth)" #verify tensor rank
        
        self.model = None
        self.dtype=dtype
        self.layer_number = layer_number
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.filter_sizes = filter_sizes
        self.n_factorization = n_factorization
        self.l = input_shape[1]
        self.w = input_shape[2]
        assert self.l == self.w, "Image width and height of input tensors are not equal. Expect square images"
        self.d = input_shape[3]
        
        #now we need to verify the image is the right size with the right filters and that all the convolutions are sound
        if verify_shape:
            assert self.output_shape is not None, "Need to enter an output shape if you want to verify it is correct"
            self.verify_shape_parameters()
    
    def create_model(self):
        
        num_input_channels = self.input_shape[3]
        
        def create_tower_parameters(name, inp_shape):
            w = tf.get_variable( str(self.layer_number) + '_w_' + name,
                                            shape=inp_shape,
                                            dtype=self.dtype,
                                            initializer=tf.contrib.layers.xavier_initializer())

            b = tf.get_variable( str(self.layer_number) + '_b_' + name,
                                            shape=[ inp_shape[3] ],
                                            dtype=self.dtype,
                                            initializer=tf.zeros_initializer() )
            return w,b
        
        #the first 3 towers start with 1x1 convolutions
        self.W_1x1_tower_1, self.b_1x1_tower_1 = create_tower_parameters('1x1_tower1',
                                                                         (1, 1, num_input_channels, self.filter_sizes[0]))
        self.W_1x1_tower_2, self.b_1x1_tower_2 = create_tower_parameters('1x1_tower2',
                                                                         (1, 1, num_input_channels, self.filter_sizes[1]))
        self.W_1x1_tower_3, self.b_1x1_tower_3 = create_tower_parameters('1x1_tower3',
                                                                         (1, 1, num_input_channels, self.filter_sizes[3]))
        
        #the fourth tower starts with a max pool that doesn't change shape, followed by a 1x1 convolution
        self.W_1x1_tower_4, self.b_1x1_tower_4 = create_tower_parameters('1x1_tower4',
                                                                        (1, 1, num_input_channels, self.filter_sizes[5]))
                                                                        
        #The second tower has a 3x3 convolution
        self.W_3x3_tower_2, self.b_3x3_tower_2 = create_tower_parameters('3x3_tower2',
                                                                        (3, 3, self.filter_sizes[1], self.filter_sizes[2]))
        
        #The third tower has a 5x5
        self.W_5x5_tower_3, self.b_5x5_tower_3 = create_tower_parameters('5x5_tower3',
                                                                        (5, 5, self.filter_sizes[3], self.filter_sizes[4]))
        
        self.model = "inception_v1_module_with_1x1_convolutions"
        
        self.model_parameters = [self.W_1x1_tower_1, self.b_1x1_tower_1,
                                 self.W_1x1_tower_2, self.b_1x1_tower_2,
                                 self.W_1x1_tower_3, self.b_1x1_tower_3,
                                 self.W_1x1_tower_4, self.b_1x1_tower_4,
                                 self.W_3x3_tower_2, self.b_3x3_tower_2,
                                 self.W_5x5_tower_3, self.b_5x5_tower_3]
                                 
    def run_model(self, tensor_input):
        """Run an inception module given a tensor input.
        
        Args:
            tensor_input - a 4f tensorflow Tensor with shape (batch_size, length, width, depth)
            
        Returns:
            Tensor - The result of the depth concatenation followed by relu at the end of the module
        
        Raises:
            ValueError, if self.model is not defined (create_v1_model has not been called)
        """
        
        if self.model is None:
            raise ValueError, "Cannot call run_model if model is not defined. Call create_model()"
        
        
        #Tower One is just a 1x1. Notice we don't relu it yet.
        tower_one_filter_out = tf.nn.conv2d(tensor_input,
                                            self.W_1x1_tower_1, #weight parameters are 
                                            strides=[1,1,1,1], #these towers use 1x1 strides.
                                            padding='SAME') + self.b_1x1_tower_1
        
        #Tower Two has the 1x1 reduction (with a relu), followed by 3x3
        tower_two_filter_reduction = tf.nn.relu(tf.nn.conv2d(tensor_input,
                                                 self.W_1x1_tower_2,
                                                 strides=[1,1,1,1],
                                                 padding='SAME') + self.b_1x1_tower_2)
        
        tower_two_filter_out = tf.nn.conv2d(tower_two_filter_reduction,
                                           self.W_3x3_tower_2,
                                           strides=[1,1,1,1],
                                           padding='SAME') + self.b_3x3_tower_2
        
        #Tower Three has the 1x1 followed by 5x5
        tower_three_filter_reduction = tf.nn.relu(tf.nn.conv2d(tensor_input,
                                                 self.W_1x1_tower_3,
                                                 strides=[1,1,1,1],
                                                 padding='SAME') + self.b_1x1_tower_3)
        
        tower_three_filter_out = tf.nn.conv2d(tower_three_filter_reduction,
                                           self.W_5x5_tower_3,
                                           strides=[1,1,1,1],
                                           padding='SAME') + self.b_5x5_tower_3
        
        #Tower Four has max pool followed by 1x1
        tower_four_pool_layer = tf.nn.max_pool(tensor_input,
                                                  ksize=[1,3,3,1],
                                                  strides=[1,1,1,1],
                                                  padding='SAME')
        
        tower_four_filter_out = tf.nn.conv2d(tower_four_pool_layer,
                                            self.W_1x1_tower_4,
                                            strides=[1,1,1,1],
                                            padding='SAME') + self.b_1x1_tower_4
        
        #And use a ReLu activation on the concatenation of these towers all at once.
        return tf.nn.relu(tf.concat([tower_one_filter_out,
                        tower_two_filter_out,
                        tower_three_filter_out,
                        tower_four_filter_out],
                        axis=3))
            
        
    def verify_shape_parameters(self):
        """
        Verifies the shapes of the inception architecture and image filters before running them in TF.
        
        Image width and length aren't changed in Inception Modules, because of same 0-padding, so we
        really need to just check that these are the same values across the input and outputs
        
        We then verify that the image depths sum to the expected values when they are depth concatenated
        at the final operation of the inception module
        
        These are all stored in the inception module class, so we return nothing and just run assertions
        """
        
        assert self.input_shape[0] == self.output_shape[0], "Dim 0 (Batch size) for input and output shapes are not equal"
        assert self.input_shape[1] == self.output_shape[1], "Dim 1 (Length pixels) for input and output shapes are not equal"
        assert self.input_shape[2] == self.output_shape[2], "Dim 2 (Width Pixels) for input and output shapes are not equal"
        
        #make sure the depth concatenated output sizes match up across all the towers
        conv_1x1_out_size = self.filter_sizes[0]
        reduced_conv_3x3_out_size = self.filter_sizes[2]
        reduced_conv_5x5_out_size = self.filter_sizes[4]
        pool_conv_1x1_out_size = self.filter_sizes[5]
        expected_total_out_size = conv_1x1_out_size + reduced_conv_3x3_out_size + reduced_conv_5x5_out_size + pool_conv_1x1_out_size
        
        assert self.output_shape[3] == expected_total_out_size, "Depth concatenated output depth is %d but expected %d." % (expected_total_out_size, self.output_shape[3])
    
    
        
        
        
        
class InceptionStemV1:
    """This class implements the stem for the inception v1 architecture. It comes before the modules. It consists
    of a several convolutional layers, two pooling layers, and two local response normalization layers.
    
    Read more about it here:
    v1: https://arxiv.org/pdf/1409.4842.pdf (using dimension reduction model, figure 2b)
        Here, we build the stem
    
    Returns:
    Pool_Output - The output form the final operation (a 3x3 max pool) over the input space.
    """
    
    def __init__(self,
                 dtype=tf.float32,
                 filter_sizes=None,
                 lrn_bias=2.,
                 lrn_alpha=0.0001,
                 lrn_beta=0.75,
                 input_shape=None,
                 output_shape=None,
                 verify_shape=True):
        """
        Creates the Tensorflow parameters for an inception stem v1 network and checks the network
        filter sizes as well as the input shape for any arithmetic inconsistencies that
        would arise.
        
        Args:
        filter_sizes: this is a python tuple of THREE integer filter sizes. Here's where they come in:

                      input
                      7x7conv_s2 <- output depth is filter_size[0]
                      3x3pool_s2
                      localResponseNorm
                      1x1conv <- output depth is filter_size[1]
                      3x3conv <- output depth is filter_size[2]
                      localResponseNorm
                      3x3pool_s2
        
        lrn_bias - float, local response normalization parameter.
        lrn_alpha - float, local response normalization parameter.
        lrn_beta - float, local response normalization parameter.
        *For these three parameters, see:
           - https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization
           - Sec 3.3 of http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
        
        input_shape - a tuple of integers specifying the network input shape. If passed, expects a 4-d list
                    such that the batch size is the first integer, then length, then width, then depth.
                    if None, then no checks will be done on input size and you will catch the errors
                    at runtime when the TF Graph is created. Batch size can be whatever. Alternatively, can be
                    a tensorflow TensorShape object and it will be silently converted to a list.

        output_shape - Tuple of 4 integers. Expected shape of the output that can be used to verify
        verify_shape - boolean. If true, verify the filter sizes against output_shape now instead of in tensorflow graph
        """
        
        #If shape is a tensorflow shape object instead of a list, implicitly convert it
        if input_shape.__class__.__name__ == 'TensorShape':
            input_shape = input_shape.as_list()
        if output_shape.__class__.__name__ == 'TensorShape':
            output_shape = output_shape.as_list()
        
        if input_shape is not None:
            assert len(input_shape) == 4, "Input must be 4-dimensional. (batch_size, len, width, depth)" #verify tensor rank
        
        self.model = None
        self.dtype=dtype
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.filter_sizes = filter_sizes
        self.lrn_bias=lrn_bias
        self.lrn_alpha=lrn_alpha
        self.lrn_beta=lrn_beta
        assert input_shape[1] == input_shape[2], "Image width and height of input tensors are not equal. Expect square images"
        
        #now we need to verify the image is the right size with the right filters and that all the convolutions are sound
        if verify_shape:
            assert self.output_shape is not None, "Need to enter an output shape if you want to verify it is correct"
            self.verify_shape_parameters()
    
    def create_model(self):
        
        num_input_channels = self.input_shape[3]
        
        #layer 1 - We start with a 7x7 convolution
        self.W_7x7_stem = tf.get_variable('v1_stem_7x7_w',
                                          shape=(7,7,num_input_channels, self.filter_sizes[0]),
                                          dtype=self.dtype,
                                          initializer=tf.contrib.layers.xavier_initializer())
        self.b_7x7_stem = tf.get_variable('v1_stem_7x7_b',
                                          shape=(self.filter_sizes[0]),
                                          dtype=self.dtype,
                                          initializer=tf.zeros_initializer())
        
        #layer 2 - max pool requires no graph parameters here
        #layer 3 - local response normalization requires no graph parameters here either
        
        #layer 4 - 1x1 convolutional reduction. recall depth will not have changed through pool or local response norm layers
        self.W_1x1_stem = tf.get_variable('v1_stem_1x1_w',
                                          shape=(1,1,self.filter_sizes[0], self.filter_sizes[1]),
                                          dtype=self.dtype,
                                          initializer=tf.contrib.layers.xavier_initializer())
        self.b_1x1_stem = tf.get_variable('v1_stem_1x1_b',
                                          shape=(self.filter_sizes[1]),
                                          dtype=self.dtype,
                                          initializer=tf.zeros_initializer())
        
        #layer 5 - 3x3 convolutional reduction
        self.W_3x3_stem = tf.get_variable('v1_stem_3x3_w',
                                          shape=(3,3,self.filter_sizes[1], self.filter_sizes[2]),
                                          dtype=self.dtype,
                                          initializer=tf.contrib.layers.xavier_initializer())
        self.b_3x3_stem = tf.get_variable('v1_stem_3x3_b',
                                          shape=(self.filter_sizes[2]),
                                          dtype=self.dtype,
                                          initializer=tf.zeros_initializer())
        
        #layer 6 - another local response norm. no graph params
        #layer 7 - another pool. no graph params
        
        self.model = "inception_v1_stem"
        
        self.model_parameters = [self.W_7x7_stem, self.b_7x7_stem,
                                 self.W_1x1_stem, self.b_1x1_stem,
                                 self.W_3x3_stem, self.b_3x3_stem]
        
    def run_model(self, tensor_input):
        """Run an inception v1 stem given a tensor input.
        
        Args:
            tensor_input - a 4f tensorflow Tensor with shape (batch_size, length, width, depth)
            
        Returns:
            Tensor - The result of the max pool 3x3 at the end of the stem
        
        Raises:
            ValueError, if self.model is not defined (create_v1_model has not been called)
        """
        
        if self.model is None:
            raise ValueError, "Cannot call run_model if model is not defined. Call create_model()"
        
        #layer 1 is 7x7 conv with relu
        out_7x7 = tf.nn.relu(tf.nn.conv2d(tensor_input,
                               self.W_7x7_stem,
                               strides=[1,2,2,1],
                               padding='SAME') + self.b_7x7_stem)
        #print("out7x7 shape is" + str(out_7x7.get_shape()))
        
        #layer 2 is a 3x3 max pool with 2x2 strides
        pool_1_out = tf.nn.max_pool(out_7x7,
                                    ksize=[1,3,3,1],
                                    strides=[1,2,2,1],
                                    padding='SAME',
                                    data_format='NHWC')
        #print("pool_1_out shape is" + str(pool_1_out.get_shape()))
        
        #layer 3 is a local response norm
        norm_1_out = tf.nn.lrn(pool_1_out,
                               bias=self.lrn_bias,
                               alpha=self.lrn_alpha,
                               beta=self.lrn_beta)
        #print("norm_1_out shape is" + str(norm_1_out.get_shape()))
        #layer 4 is a 1x1 convolutional reduction with a relu. it uses valid padding
        out_1x1 = tf.nn.relu(tf.nn.conv2d(norm_1_out,
                                          self.W_1x1_stem,
                                          strides=[1,1,1,1],
                                          padding='VALID') + self.b_1x1_stem) #GoogLeNet uses valid padding on this one convolution only
        #print("out_1x1 shape is" + str(out_1x1.get_shape()))
        
        #layer 5 is a 3x3 convolution reduction with a relu
        out_3x3 = tf.nn.relu(tf.nn.conv2d(out_1x1,
                                          self.W_3x3_stem,
                                          strides=[1,1,1,1],
                                          padding='SAME') + self.b_3x3_stem)
        #print("out_3x3 shape is" + str(out_3x3.get_shape()))
        
        #layer 6 is another local response normalization
        norm_2_out = tf.nn.lrn(out_3x3,
                               bias=self.lrn_bias,
                               alpha=self.lrn_alpha,
                               beta=self.lrn_beta)
        #print("norm_2_out shape is" + str(norm_2_out.get_shape()))
        
        #finally, a 3x3 pool
        pool_2_out = tf.nn.max_pool(norm_2_out,
                                    ksize=[1,3,3,1],
                                    strides=[1,2,2,1],
                                    padding='SAME',
                                    data_format='NHWC')
        #print("pool_2_out shape is" + str(pool_2_out.get_shape()))
        return pool_2_out

        
    def verify_shape_parameters(self):
        """
        Verifies the shapes of the inception architecture and image filters before running them in TF.
        """
        assert self.input_shape[0] == self.output_shape[0], "Dim 0 (Batch size) for input and output shapes are not equal. They are %d and %d, respectively " % (self.input_shape[0], self.output_shape[0])

        
        #TODO - flesh this out. not totally necessary since tensorflow will catch it and probably give a better message anyway
        # could abstract it away into a class that checks general convolutional arithmetic
        
        
class InceptionClassifierV1:
    """This class implements the classifiers, including the auxiliary classifiers, for the inception v1 architecture. 
    It comes before the modules. It consists of fully connected layers, dropout layers, pooling layers, softmax actiation functions, and convolutional layers if the classifiers are auxiliary.
    
    Read more about it here:
    v1: https://arxiv.org/pdf/1409.4842.pdf
    
    Returns:
    Softmax Output - The output form the final operation (softmax) over the input space.
    """
    
    def __init__(self,
                 num_labels,
                 filter_sizes=None,
                 dtype=tf.float32,
                 auxiliary_weight_constant=1.0,
                 auxiliary_classifier=False,
                 input_shape=None,
                 output_shape=None,
                 verify_shape=True):
        """
        Creates the Tensorflow parameters for an inception stem v1 network and checks the network
        filter sizes as well as the input shape for any arithmetic inconsistencies that
        would arise.
        
        Args:
        num_labels: integer, the depth of the fully-connected layer that will output to the softmax to get the final logits
        
        filter_sizes: this is a python tuple of TWO integer filter sizes, or the value 'None'. Here's where they come in:

                      If classifier is auxiliary:
                      input
                      averagepool 5x5
                      conv 1x1 <- output depth is filter_size[0].
                      fc       <- output depth is filter_size[1], but note this is the size BEFORE flattening the conv 1x1,
                                  so if this number is 100, and you have a 64x12x12x2, you now have 64 x 28800
                      dropout
                      fc
                      softmax_activation
                      
                      If classifier is not auxiliary (it's the final classifier in the network), there are no filter sizes to use
        
        auxiliary_weight_constant - float - scalar with which to multiply final softmax values for auxiliary classifier. default value is 1.0, ie, no effect.
        
        auxiliary_classifier - If True, follows auxiliary classifier in googlenet implementation. this means a 5x5 pool,
        convolutional reduction, and another fully connected layer.
        
        input_shape - a tuple of integers specifying the network input shape. If passed, expects a 4-d list
                    such that the batch size is the first integer, then length, then width, then depth.
                    if None, then no checks will be done on input size and you will catch the errors
                    at runtime when the TF Graph is created. Batch size can be whatever. Alternatively, can be
                    a tensorflow TensorShape object and it will be silently converted to a list.

        output_shape - Tuple of 4 integers. Expected shape of the output that can be used to verify. Probably [batch, 1, 1, depth]
        verify_shape - boolean. If true, verify the filter sizes against output_shape now instead of in tensorflow graph
        """
        
        #If shape is a tensorflow shape object instead of a list, implicitly convert it
        if input_shape.__class__.__name__ == 'TensorShape':
            input_shape = input_shape.as_list()
        if output_shape.__class__.__name__ == 'TensorShape':
            output_shape = output_shape.as_list()
        
        if input_shape is not None:
            assert len(input_shape) == 4, "Input must be 4-dimensional. (batch_size, len, width, depth)" #verify tensor rank
        
        self.model = None
        self.num_labels = num_labels
        self.dtype=dtype
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.auxiliary_classifier = auxiliary_classifier
        self.auxiliary_weight_constant = tf.constant(auxiliary_weight_constant)
        
        if self.auxiliary_classifier:
            assert len(filter_sizes) == 2, "Expected 2 filter sizes for an auxiliary v1 inception classifier. (these two sizes are conv reduction depth, fc1 depth)"
            self.filter_sizes = filter_sizes
        else:
            assert filter_sizes is None, "If not using an auxiliary classifier, self.filter_sizes must be None. It is %s" % str(filter_sizes)
            self.filter_sizes = None

        assert input_shape[1] == input_shape[2], "Image width and height of input tensors are not equal. Expect square images"
        
        #now we need to verify the image is the right size with the right filters and that all the convolutions are sound
        if verify_shape:
            assert self.output_shape is not None, "Need to enter an output shape if you want to verify it is correct"
            self.verify_shape_parameters()
    
    def create_model(self):
        if self.auxiliary_classifier:
            print("creating auxiliary model")
            self._create_auxiliary_model()
            self.model = "inception_v1_auxiliary_classifier_model"
        else:
            print("creating basic model")
            self._create_basic_model()
            self.model = "inception_v1_basic_classifier_model"
    
    def _create_basic_model(self):
        #layer 1 - We start with a 7x7, 3 valid-padding pooling layer, which has no parameters
        
        #layer 2 - then a fully connected layer to what will be the flattened input from the convolutional layer
        #We have to do a little arithmetic to figure out what the size of this will be.
        #The pool is a 7x7 average pool with stride 1, with valid padding.
        #In tensorflow, valid padding is given by:
        #    out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
        #    out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))
        #We will plug these values in accordingly because we will need them after the convolutional 1x1 reduction layer
        out_height = np.ceil(float(self.input_shape[1] - 7 + 1) / 1.0) #5x5, 1x1 strides
        out_width = np.ceil(float(self.input_shape[2] - 7 + 1) / 1.0)
        
        self.W_fc1_classifier = tf.get_variable('v1_classifier_fc1_w',
                                          shape=(out_height*out_width*self.input_shape[3], self.num_labels),
                                          dtype=self.dtype,
                                          initializer=tf.contrib.layers.xavier_initializer())
        self.b_fc1_classifier = tf.get_variable('v1_classifier_fc1_b',
                                          shape=(self.num_labels),
                                          dtype=self.dtype,
                                          initializer=tf.zeros_initializer())
        
        self.model = "inception_v1_basic_classifier"
        
        self.model_parameters = [self.W_fc1_classifier, self.b_fc1_classifier]
        
    
    def _create_auxiliary_model(self):
        num_input_channels = self.input_shape[3]
        
        #layer 1 - We start with a 5x5, 3 valid-padding pooling layer, which has no parameters
        
        #layer 2 - then a fully connected layer to what will be the flattened input from the convolutional layer
        #We have to do a little arithmetic to figure out what the size of this will be.
        #The pool is a 5x5 average pool with stride 3, with valid padding.
        #In tensorflow, valid padding is given by:
        #    out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
        #    out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))
        #We will plug these values in accordingly because we will need them after the convolutional 1x1 reduction layer
        out_height = np.ceil(float(self.input_shape[1] - 5 + 1) / 3.0) #5x5, 3x3 strides
        out_width = np.ceil(float(self.input_shape[2] - 5 + 1) / 3.0)

        #then the 1x1 convolutional reduction
        self.W_1x1_classifier = tf.get_variable('v1_classifier_1x1_w',
                                          shape=(1,1,num_input_channels, self.filter_sizes[0]),
                                          dtype=self.dtype,
                                          initializer=tf.contrib.layers.xavier_initializer())
        self.b_1x1_classifier = tf.get_variable('v1_classifier_1x1_b',
                                          shape=(self.filter_sizes[0]),
                                          dtype=self.dtype,
                                          initializer=tf.zeros_initializer())
        
        #layer 3 - a fully connected layer
        self.W_fc1_classifier = tf.get_variable('v1_classifier_fc1_w',
                                          shape=(out_height*out_width*self.filter_sizes[0], self.filter_sizes[1]),
                                          dtype=self.dtype,
                                          initializer=tf.contrib.layers.xavier_initializer())
        self.b_fc1_classifier = tf.get_variable('v1_classifier_fc1_b',
                                          shape=(self.filter_sizes[1]),
                                          dtype=self.dtype,
                                          initializer=tf.zeros_initializer())
        
        #layer 4 - a fully connected layer to output the logits for each class
        self.W_fc2_classifier = tf.get_variable('v1_classifier_fc2_w',
                                          shape=(self.filter_sizes[1], self.num_labels),
                                          dtype=self.dtype,
                                          initializer=tf.contrib.layers.xavier_initializer())
        self.b_fc2_classifier = tf.get_variable('v1_classifier_fc2_b',
                                          shape=(self.num_labels),
                                          dtype=self.dtype,
                                          initializer=tf.zeros_initializer())
        
        #layer 5 - a softmax, which requires no parameters
        self.model = "inception_v1_auxiliary_classifier"
        self.model_parameters = [self.W_1x1_classifier, self.b_1x1_classifier,
                                self.W_fc1_classifier, self.b_fc1_classifier,
                                self.W_fc2_classifier, self.b_fc2_classifier]
        
    def run_model(self, tensor_input, training=True):
        """Run an inception v1 stem given a tensor input.
        
        Args:
            tensor_input - a 4f tensorflow Tensor with shape (batch_size, length, width, depth)
            training - boolean, controls whether or not to use dropout
            
        Returns:
            Tensor - The result of the softmax activation at the end of the chain
        
        Raises:
            ValueError, if self.model is not defined (create_v1_model has not been called)
        """
        
        if self.model is None:
            raise ValueError, "Cannot call run_model if model is not defined. Call create_model()"
        
        if self.auxiliary_classifier:
            return self._run_auxiliary_model(tensor_input, training=training)
        else:
            return self._run_basic_model(tensor_input, training=training)
    
    def _run_auxiliary_model(self, tensor_input, training=True):
        """See run_model
        
        Runs five layers. They are:
        
        1. 5x5 average pool with valid padding. flattened to 2d
        2. 1x1 convolutional reduction with stride 1
        3. fully connected layer, relu activation function
        4. dropout
        5. fully connected layer, no activation function, to num_labels classifications
        (7) - Multiplied by auxiliary weight for final output logits

        #TODO - implement dropout
        """
        pool_1_out = tf.nn.avg_pool(tensor_input,
                                ksize=[1,5,5,1],
                                strides=[1,3,3,1],
                                padding='VALID',
                                data_format='NHWC')
        
        out_1x1 = tf.nn.relu(tf.nn.conv2d(pool_1_out,
                                          self.W_1x1_classifier,
                                          strides=[1,1,1,1],
                                          padding='SAME') + self.b_1x1_classifier)
        
        flat_out_1x1 = tf.contrib.layers.flatten(out_1x1)
           
        #we have to flatten the 1x1 output to get it to 2d for the fully-connected layer
        #flat_out_1x1 = tf.contrib.layers.flatten(tf.nn.relu(tf.nn.conv2d(pool_1_out,
        #                                  self.W_1x1_classifier,
        #                                  strides=[1,1,1,1],
        #                                  padding='SAME') + self.b_1x1_classifier))
        
        fc1_out = tf.nn.relu(tf.add( tf.matmul(flat_out_1x1, self.W_fc1_classifier), self.b_fc1_classifier ) )
        
        #TODO - add dropout here if training
        
        unnormalized_logits = tf.add( tf.matmul(fc1_out, self.W_fc2_classifier), self.b_fc2_classifier )
        
        #need to scale by the auxiliary classifier weight
        #return tf.multiply( unnormalized_logits, self.auxiliary_weight_constant )
        return unnormalized_logits
    
    def _run_basic_model(self, tensor_input, training=True):
        """See run_model
        
        Runs three layers. They are:
        
        1. 7x7 average pool with valid padding
        2. dropout
        3. fully connected layer, no activation function, to num_labels classifications
        
        #TODO - implement dropout
        """
        
        #we have to flatten the pool output to get it to 2d for the fully-connected layer
        pool_1_out = tf.contrib.layers.flatten(tf.nn.avg_pool(tensor_input,
                                                                ksize=[1,7,7,1],
                                                                strides=[1,1,1,1],
                                                                padding='VALID',
                                                                data_format='NHWC'))
        
        #TODO - dropout goes here
        
        unnormalized_logits = tf.add( tf.matmul(pool_1_out, self.W_fc1_classifier), self.b_fc1_classifier )
                
        return unnormalized_logits
        
        
    def verify_shape_parameters(self):
        """
        Verifies the shapes of the inception architecture and image filters before running them in TF.
        """
        assert self.input_shape[0] == self.output_shape[0], "Dim 0 (Batch size) for input and output shapes are not equal. They are %d and %d, respectively " % (self.input_shape[0], self.output_shape[0])
        
        #TODO - flesh this out

