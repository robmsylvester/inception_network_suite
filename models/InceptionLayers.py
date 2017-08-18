import tensorflow as tf
from tensorflow.python.ops import variable_scope
import numpy as np
import data_utils

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
                 filter_sizes=None,
                 dtype=tf.float32,
                 input_shape=[None,224,224,3],#imagenet
                 output_shape=[None,224,224,64],
                 scope=None):
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
        
        name - the name that will be appended to the tensorflow scope to create a scope under which to use this module
        
        input_shape - a tuple of integers specifying the network input shape. Expects a 4-d list
                    such that the batch size is the first and will silently be None, so use that value.
                    then length, then width, then depth, all of which are integers. length and width
                    must be the same size
                        
                    tensor.get_shape().as_list() will turn your tensor shape into a list.
        
        n_factorization - positive integer, if using a v2 inception with factorization (https://arxiv.org/pdf/1512.00567v3.pdf),
                          this is the value for n.
        
        output_shape - Tuple of 4 integers. Expected shape of the output that can be used to verify
        scope - tensorflow scope, defaults to 'inceptionV1module'
        """
        
        self.input_shape, self.output_shape, self.filter_sizes = self.validate_dimensions(input_shape, output_shape, filter_sizes)
        self.model = None
        self.scope = "inceptionV1_module1" if not scope else scope #no empty strings, None. 
        #print("Creating InceptionModule with input shape %s" % input_shape) 
        self.dtype=dtype
    
    def create_model(self):
        
        with variable_scope.variable_scope(self.scope, dtype=self.dtype) as scope:
            
            num_input_channels = self.input_shape[3]

            def create_tower_parameters(name, inp_shape):
                w = tf.get_variable('_w_' + name,
                                    shape=inp_shape,
                                    dtype=self.dtype,
                                    initializer=tf.contrib.layers.xavier_initializer())

                b = tf.get_variable( '_b_' + name,
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
                                 
    def run_model(self, tensor_input, training=True):
        """Run an inception module given a tensor input.
        
        Args:
            tensor_input - a 4f tensorflow Tensor with shape (batch_size, length, width, depth)
            training - boolean, controls whether or not to use dropout layer
            
        Returns:
            Tensor - The result of the depth concatenation followed by relu at the end of the module
        
        Raises:
            ValueError, if self.model is not defined (create_v1_model has not been called)
            ValueError, if tensor_input.get_shape().as_list() does not match self.input_shape
        """
        
        if self.model is None:
            raise ValueError, "Cannot call run_model if model is not defined. Call create_model()"
        if tensor_input.get_shape().as_list()[1:] != self.input_shape[1:]:
            raise ValueError, "Tensor input to run_model doesn't match expect input_shape for module class (ignoring batch size). input is %s, but expected %s" % (str(tensor_input.get_shape().as_list()), str(self.input_shape))
        
        with variable_scope.variable_scope(self.scope, dtype=self.dtype) as scope:
        
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

    def validate_dimensions(self, input_shape, output_shape, filter_sizes):
        #If shape is a tensorflow shape object instead of a list, implicitly convert it
        if input_shape.__class__.__name__ == 'TensorShape':
            input_shape = input_shape.as_list()
        if output_shape.__class__.__name__ == 'TensorShape':
            output_shape = output_shape.as_list()

        """
        Verify the shapes of the inception architecture and image filters before running them in TF.

        Image width and length aren't changed in Inception Modules, because of same 0-padding, so we
        really need to just check that these are the same values across the input and outputs

        We then verify that the image depths sum to the expected values when they are depth concatenated
        at the final operation of the inception module.
        """

        assert len(input_shape) == 4, "Input must be 4-dimensional. (batch_size, len, width, depth)" #verify tensor rank
        assert input_shape[1] == input_shape[2], "Expected Square Images. Images are %d by %d" % (input_shape[1], input_shape[2])
        assert input_shape[1] == output_shape[1], "Dim 1 (Length pixels) for input and output shapes are not equal. %d and %d" % (input_shape[1],output_shape[1])
        assert input_shape[2] == output_shape[2], "Dim 2 (Width Pixels) for input and output shapes are not equal"

        #make sure the depth concatenated output sizes match up across all the towers
        conv_1x1_out = filter_sizes[0]
        reduced_conv_3x3_out = filter_sizes[2]
        reduced_conv_5x5_out = filter_sizes[4]
        pool_conv_1x1_out = filter_sizes[5]
        expected_total_out = conv_1x1_out + reduced_conv_3x3_out + reduced_conv_5x5_out + pool_conv_1x1_out
        assert output_shape[3] == expected_total_out, "Depth concatenated output depth is %d but expected %d." % (expected_total_out, output_shape[3])

        return input_shape, output_shape, filter_sizes
        
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
                 scope=None):
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
        scope - the tensorflow scope.
        """
        
        self.input_shape, self.output_shape, self.filter_sizes = self.validate_dimensions(input_shape, output_shape, filter_sizes)
        self.scope = "inceptionV1_stem" if not scope else scope
        #print("Creating InceptionStem with input shape %s" % input_shape) 
        self.model = None
        self.dtype=dtype
        self.lrn_bias=lrn_bias
        self.lrn_alpha=lrn_alpha
        self.lrn_beta=lrn_beta
      
    def create_model(self):
        
        with variable_scope.variable_scope(self.scope, dtype=self.dtype) as scope:
        
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
        
    def run_model(self, tensor_input, training=True):
        """Run an inception v1 stem given a tensor input.
        
        Args:
            tensor_input - a 4f tensorflow Tensor with shape (batch_size, length, width, depth)
            training - boolean, controls whether or not to use dropout layer
            
        Returns:
            Tensor - The result of the max pool 3x3 at the end of the stem
        
        Raises:
            ValueError, if self.model is not defined (create_v1_model has not been called)
            ValueError, if self.input_shape doesn't match tensor_input shape
        """
        
        if self.model is None:
            raise ValueError, "Cannot call run_model if model is not defined. Call create_model()"
        if tensor_input.get_shape().as_list()[1:] != self.input_shape[1:]:
            raise ValueError, "Tensor input to run_model doesn't match expect input_shape for module class (ignoring batch size). input is %s, but expected %s" % (str(tensor_input.get_shape().as_list()), str(self.input_shape))
        
        with variable_scope.variable_scope(self.scope, dtype=self.dtype) as scope:
        
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

        
    def validate_dimensions(self, input_shape, output_shape, filter_sizes):
        #If shape is a tensorflow shape object instead of a list, implicitly convert it
        if input_shape.__class__.__name__ == 'TensorShape':
            input_shape = input_shape.as_list()
        if output_shape.__class__.__name__ == 'TensorShape':
            output_shape = output_shape.as_list()

        """
        Verify the shapes of the inception stem and image filters before running them in TF.

        We then verify that the image depths sum to the expected values.
        """

        assert len(input_shape) == 4, "Input must be 4-dimensional. (batch_size, len, width, depth)" #verify tensor rank
        assert input_shape[1] == input_shape[2], "Expected Square Images. Images are %d by %d" % (input_shape[1], input_shape[2])
        assert input_shape[1]/8 == output_shape[1], "Dim 1 (Length pixels) for input and output shapes are not as expected. The inception stem should reduce the image length and width by a factor of 8. Instead, input and output sizes are %d and %d, respectively." % (input_shape[1],output_shape[1])
        assert input_shape[2]/8 == output_shape[2], "Dim 1 (Length pixels) for input and output shapes are not as expected. The inception stem should reduce the image length and width by a factor of 8. Instead, input and output sizes are %d and %d, respectively." % (input_shape[1],output_shape[1])

        #make sure the depth concatenated output sizes are valid
        #TODO - they are just convolutional depths, so really they dont need to be validated here.

        return input_shape, output_shape, filter_sizes
        
        
class InceptionClassifierV1:
    """This class implements the classifiers, including the auxiliary classifiers, for the inception v1 architecture. 
    It comes before the modules. It consists of fully connected layers, dropout layers, pooling layers, softmax actiation functions, and convolutional layers if the classifiers are auxiliary.
    
    Read more about it here:
    v1: https://arxiv.org/pdf/1409.4842.pdf
    
    Returns:
    Softmax Output - The output form the final operation (softmax) over the input space.
    """
    
    def __init__(self,
                 filter_sizes=None,
                 dtype=tf.float32,
                 auxiliary_weight_constant=1.0,
                 auxiliary_classifier=False,
                 input_shape=None,
                 output_shape=None,
                 scope=None):
        """
        Creates the Tensorflow parameters for an inception stem v1 network and checks the network
        filter sizes as well as the input shape for any arithmetic inconsistencies that
        would arise.
        
        Args:        
        filter_sizes: this is a python tuple of TWO integer filter sizes, or the value 'None'. Here's where they come in:

                      If classifier is auxiliary:
                      input
                      averagepool 5x5
                      conv 1x1 <- output depth is filter_size[0].
                      fc       <- output depth AFTER FLATTENING. the 1x1 conv. Make sure this is divisble by reduction_factor**2
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

        output_shape - Tuple of 2 integers. Expected shape of the output. [batch size, num labels]        
        scope - tensorflow scope
        
        Returns - Tensor, the output of the network with size output_shape
        """
        
        self.scope = "inceptionV1_classifier" if not scope else scope #no empty strings, None.         
        self.model = None
        self.dtype=dtype
        self.auxiliary_classifier = auxiliary_classifier
        self.auxiliary_weight_constant = tf.constant(auxiliary_weight_constant)
        
        self.input_shape, self.output_shape, self.filter_sizes = self.validate_dimensions(input_shape, output_shape, filter_sizes)
    
    def create_model(self):
        with variable_scope.variable_scope(self.scope, dtype=self.dtype) as scope:
            if self.auxiliary_classifier:
                self._create_auxiliary_model()
                self.model = "inception_v1_auxiliary_classifier_model"
            else:
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
                                          shape=(out_height*out_width*self.input_shape[3], self.output_shape[1]),
                                          dtype=self.dtype,
                                          initializer=tf.contrib.layers.xavier_initializer())
        self.b_fc1_classifier = tf.get_variable('v1_classifier_fc1_b',
                                          shape=(self.output_shape[1]),
                                          dtype=self.dtype,
                                          initializer=tf.zeros_initializer())
        
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
                                          shape=(self.filter_sizes[1], self.output_shape[1]),
                                          dtype=self.dtype,
                                          initializer=tf.contrib.layers.xavier_initializer())
        self.b_fc2_classifier = tf.get_variable('v1_classifier_fc2_b',
                                          shape=(self.output_shape[1]),
                                          dtype=self.dtype,
                                          initializer=tf.zeros_initializer())
        
        #layer 5 - a softmax, which requires no parameters
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
            ValueError, if self.input_shape doesn't match tensor_input shape
        """
        
        if self.model is None:
            raise ValueError, "Cannot call run_model if model is not defined. Call create_model()"
        if tensor_input.get_shape().as_list()[1:] != self.input_shape[1:]:
            raise ValueError, "Tensor input to run_model doesn't match expect input_shape for module class (ignoring batch size). input is %s, but expected %s" % (str(tensor_input.get_shape().as_list()), str(self.input_shape))
        
        with variable_scope.variable_scope(self.scope, dtype=self.dtype) as scope:
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
        
        fc1_out = tf.nn.relu(tf.add( tf.matmul(flat_out_1x1, self.W_fc1_classifier), self.b_fc1_classifier ) )
        
        #TODO - add dropout here if training
        
        unnormalized_logits = tf.add( tf.matmul(fc1_out, self.W_fc2_classifier), self.b_fc2_classifier )

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
        
    def validate_dimensions(self, input_shape, output_shape, filter_sizes):
        #If shape is a tensorflow shape object instead of a list, implicitly convert it
        if input_shape.__class__.__name__ == 'TensorShape':
            input_shape = input_shape.as_list()
        if output_shape.__class__.__name__ == 'TensorShape':
            output_shape = output_shape.as_list()
        
        assert len(output_shape) == 2, "Output must be 2-dimensional. (batch_size, num_labels)"
        assert len(input_shape) == 4, "Input must be 4-dimensional. (batch_size, len, width, depth)"
        
        assert input_shape[1] == input_shape[2], "Expected Square Images. Images are %d by %d" % (input_shape[1], input_shape[2])

        if self.auxiliary_classifier:
            assert len(filter_sizes) == 2, "Expected 2 filter sizes for an auxiliary v1 inception classifier. (these two sizes are conv reduction depth, fc1 depth)"
        else:
            assert filter_sizes is None, "If not using an auxiliary classifier, filter_sizes must be None. It is %s" % str(filter_sizes)

        #make sure the depth concatenated output sizes are valid
        #TODO - they are just convolutional depths, so really they dont need to be validated here.

        return input_shape, output_shape, filter_sizes
        

class InceptionV1():
    """This class puts together the inception pieces in InceptionLayers to replicate the googLenet architecture
    present over hnnyah -> https://arxiv.org/pdf/1409.4842.pdf
    recognition. It runs a 1x1, 3x3, and 5x5 convolutions as well as max pooling operations to a single
    input before concatenating the outputs along the depth axis. At least, that is one version of the inception
    architecure. There are several versions of inception that have come out over the years in various papers
    
    v1: https://arxiv.org/pdf/1409.4842.pdf (Figure 3 and Table 1, specifically)
    
    For the walkthrough of the model, see the filter_sizes argument or the run_model method.
    
    Returns:
        A list of three tensors, one for each of the three network classifiers, of the shape [BATCH_SIZE, NUM_LABELS]
    
    Raises:
        ValueError, if the filter_sizes aren't correct for the specific input and output shape
        These can be difficult to fine-tune, so I advise using the pre_loaded ones or providing
        a scale factor argument to this class to make it easy
    """
    
    def __init__(self,
                 filter_size_dict="imagenet_auto",
                 filter_size_reduction_factor=4,
                 auxiliary_classifier_weights=[0.3,0.3],
                 use_mini_model=False, #skips three of the inception layers that dont change output size
                 dtype=tf.float32,
                 input_shape=[64,224,224,3], #224x224x3 imagenet images
                 output_shape=[64,1,1,1000], 
                 scope=None):
        """
        Creates the Tensorflow parameters for an inception v1 module network and checks the network
        filter sizes as well as the input shape for any arithmetic inconsistencies that
        would arise.
        
        Args:
        filter_size_dict: this is a python tuple of filter_sizes for the inception network. This is a tough one
                      to get right. You provide the sizes of everything in the inception network. Or, you load
                      the standard network sizes scaled by a reducible factor, like four or eight, so that the
                      arithmetic works correctly.
                      
                      "imagenet_auto" - loads filter sizes for a network to run the googlenet 
                      size network at https://arxiv.org/pdf/1409.4842.pdf (Table 1), but scaled by 1/4
                      
                      Such a table with a 1/4 scale loads the following filters in the inception network:
                      
                      filter_size_dict['stem'] = [16,16,48]
                      filter_size_dict['module1'] = [16,24,32,4,8,8]
                      filter_size_dict['module2'] = [32,32,48,8,24,16]
                      filter_size_dict['module3'] = [48,24,52,4,12,16]
                      filter_size_dict['aux_classifier1'] = [32,1024]
                      filter_size_dict['module4'] = [40, 28, 56, 6, 16, 16]
                      filter_size_dict['module5'] = [32, 32, 64, 6, 16, 16]
                      filter_size_dict['module6'] = [28, 36, 72, 8, 16, 16]
                      filter_size_dict['aux_classifier2'] = [32,1024]
                      filter_size_dict['module7'] = [64, 40, 80, 8, 32, 32]
                      filter_size_dict['module8'] = [64, 40, 80, 8, 32, 32]
                      filter_size_dict['module9'] = [96, 48, 96, 12, 32, 32]
                      filter_size_dict['classifier3'] = None
                      
        filter_size_reduction_factor -  this parameter is ignored unless 'imagenet_auto' is used for the filter. this number, n,
        will be used to scale the depths of the filters in Table 1 of https://arxiv.org/pdf/1409.4842.pdf by a factor of 1/n.
        The default value of 4 is recommended if you have a powerful GPU architecture. MUST be [1,2,4,8]
        
        auxiliary_classifier_weights - list of two floats. scale the loss function from these softmax logit outputs
               to the true labels for the first auxiliary classifier, and 
        
        input_shape - a tuple of integers specifying the network input shape. If passed, expects a 4-d list
                    such that the batch size is the first integer, then length, then width, then depth.
                    if None, then no checks will be done on input size and you will catch the errors
                    at runtime when the TF Graph is created. Batch size can be whatever.
                    
                    tensor.get_shape().as_list() will turn your tensor shape into a list.
        
        output_shape - Tuple of 4 integers. Expected shape of the output that can be used to verify
        
        scope - Tensorflow scope. If None, Will create 'InceptionModule' scope
        """
        
        self.scope = "inceptionV1_classifier" if not scope else scope #no empty strings, None.
        self.dtype=dtype
        self.model = None
        self.input_shape, self.output_shape, self.filter_size_dict = self.validate_dimensions(input_shape, output_shape, filter_size_dict)
        
        #TODO - move this below logic elsewhere
        if filter_size_reduction_factor not in [1,2,4,8]:
            raise ValueError("The filter_size_reduction_factor argument must be 1,2,4 or 8. This is because all of the depth values in the network need to be divisible by this number, otherwise the architecture must be changed.")
        else:
            self.filter_size_reduction_factor = filter_size_reduction_factor
        
        if filter_size_dict == "imagenet_auto":
            self.filter_size_dict, self.layer_output_shapes = data_utils.load_imagenet_architecture_filters(model="v1", reduction_factor=4, batch_size=self.input_shape[0])

    
    def create_model(self):

        self.layers = []
        
        #we need to create one stem
        self.layers.append(InceptionStemV1(filter_sizes=self.filter_size_dict['stem'],
                                            input_shape=self.input_shape,
                                            output_shape=self.layer_output_shapes['stem'],
                                            scope="stem"))
        self.layers.append(InceptionModuleV1(input_shape = self.layer_output_shapes['stem'],
                                            output_shape = self.layer_output_shapes['module1'],
                                            filter_sizes = self.filter_size_dict['module1'],
                                            scope="module1"))
        self.layers.append(InceptionModuleV1(input_shape = self.layer_output_shapes['module1'],
                                            output_shape = self.layer_output_shapes['module2'],
                                            filter_sizes = self.filter_size_dict['module2'],
                                            scope="module2"))
        
        #then pool with 2x2 strides before the third inception module, so we have to divide the input_shape by 2
        #in the second and third dimensions
        pooled_input_shape = [self.layer_output_shapes['module2'][0],\
                            self.layer_output_shapes['module2'][1]/2,\
                            self.layer_output_shapes['module2'][2]/2,\
                            self.layer_output_shapes['module2'][3]]
        
        self.layers.append(InceptionModuleV1(input_shape = pooled_input_shape,
                                            output_shape = self.layer_output_shapes['module3'],
                                            filter_sizes = self.filter_size_dict['module3'],
                                            scope="module3"))
        
        #one of these goes to an auxiliary classifier, so we put that first
        self.layers.append(InceptionClassifierV1(dtype=tf.float32,
                                                auxiliary_weight_constant=0.3,
                                                filter_sizes=self.filter_size_dict['aux_classifier1'],
                                                auxiliary_classifier=True,
                                                input_shape=self.layer_output_shapes['module3'],
                                                output_shape=self.layer_output_shapes['aux_classifier1'],
                                                scope="aux_classifier1"))
        #then three inception modules
        self.layers.append(InceptionModuleV1(input_shape = self.layer_output_shapes['module3'],
                                            output_shape = self.layer_output_shapes['module4'],
                                            filter_sizes = self.filter_size_dict['module4'],
                                            scope="module4"))
        self.layers.append(InceptionModuleV1(input_shape = self.layer_output_shapes['module4'],
                                            output_shape = self.layer_output_shapes['module5'],
                                            filter_sizes = self.filter_size_dict['module5'],
                                            scope="module5"))
        self.layers.append(InceptionModuleV1(input_shape = self.layer_output_shapes['module5'],
                                            output_shape = self.layer_output_shapes['module6'],
                                            filter_sizes = self.filter_size_dict['module6'],
                                            scope="module6"))
        
        #then one more auxiliary classifier
        self.layers.append(InceptionClassifierV1(dtype=tf.float32,
                                                auxiliary_weight_constant=0.3,
                                                filter_sizes=self.filter_size_dict['aux_classifier2'],
                                                auxiliary_classifier=True,
                                                input_shape=self.layer_output_shapes['module6'],
                                                output_shape=self.layer_output_shapes['aux_classifier2'],
                                                scope="aux_classifier2"))
        
        #then a module
        self.layers.append(InceptionModuleV1(input_shape = self.layer_output_shapes['module6'],
                                            output_shape = self.layer_output_shapes['module7'],
                                            filter_sizes = self.filter_size_dict['module7'],
                                            scope="module7"))
        
        #another pool cuts down the input size on the next module
        pooled_input_shape = [self.layer_output_shapes['module7'][0],\
                            self.layer_output_shapes['module7'][1]/2,\
                            self.layer_output_shapes['module7'][2]/2,\
                            self.layer_output_shapes['module7'][3]]

        #then a module
        self.layers.append(InceptionModuleV1(input_shape = pooled_input_shape,
                                            output_shape = self.layer_output_shapes['module8'],
                                            filter_sizes = self.filter_size_dict['module8'],
                                            scope="module8"))
        #final module before classifier
        self.layers.append(InceptionModuleV1(input_shape = self.layer_output_shapes['module8'],
                                            output_shape = self.layer_output_shapes['module9'],
                                            filter_sizes = self.filter_size_dict['module9'],
                                            scope="module9"))
        
        self.layers.append(InceptionClassifierV1(dtype=tf.float32,
                                                filter_sizes=self.filter_size_dict['classifier3'],
                                                auxiliary_classifier=False,
                                                input_shape=self.layer_output_shapes['module9'],
                                                output_shape=self.layer_output_shapes['classifier3'],
                                                scope="classifier3"))
        
        #Hot diggity! We're done. Run create_model to make the parameters
        with variable_scope.variable_scope(self.scope, dtype=self.dtype) as scope:
            self.model_parameters_2d = []
            for layer in self.layers:
                layer.create_model()
                self.model_parameters_2d.append(layer.model_parameters)

            self.model = "inception_v1"
            
            #now we can access the layers to regularize them by just calling your_inception_model_name.model_parameters
            self.model_parameters = [item for sub_list in self.model_parameters_2d for item in sub_list]
    
    def run_model(self, tensor_input, training=True):
        """Run an inception v1 stem given a tensor input.
        This follows Figure 3 and Table 1 from https://arxiv.org/pdf/1409.4842.pdf
        
        Args:
            tensor_input - a 4f tensorflow Tensor with shape (batch_size, length, width, depth)
            training - boolean, controls whether or not to use dropout
            
        Returns:
            List of Tensorflow tensors, one for each of the three classifiers, of shape [batch_size, num_labels]
        
        Raises:
            ValueError, if self.model is not defined (create_v1_model has not been called)
            ValueError, if self.input_shape doesn't match tensor_input shape
        """
        
        if self.model is None:
            raise ValueError, "Cannot call run_model if model is not defined. Call create_model()"
        if tensor_input.get_shape().as_list()[1:] != self.input_shape[1:]:
            raise ValueError, "Tensor input to run_model doesn't match expect input_shape for module class (ignoring batch size). input is %s, but expected %s" % (str(tensor_input.get_shape().as_list()), str(self.input_shape))
        
        classifications = [] # list of tensors
        
        #0 : stem
        #1-3 : modules, with maxpool after 2
        #4 : classifier
        #5-7 : modules
        #8 : classifier
        #9-11: modules, with maxpool after 9
        #12 : classifier
        max_pool_before_layers = [3,10]

        with variable_scope.variable_scope(self.scope, dtype=self.dtype) as scope:
            layer_out = tensor_input #this will start our loop
            for idx,layer in enumerate(self.layers):
                
                if idx in max_pool_before_layers:
                    layer_out = tf.nn.max_pool(layer_out,
                                               ksize=[1,3,3,1],
                                               strides=[1,2,2,1],
                                               padding='SAME')
                
                if layer.__class__.__name__ == 'InceptionClassifierV1':
                    classifications.append ( layer.run_model(layer_out, training=training) ) 
                else:
                    layer_out = layer.run_model(layer_out, training=training)
                    #print("Running layer %d" % idx)
                    #print("Output is %s" % layer_out.get_shape())
            return classifications
    
    
    def validate_dimensions(self, input_shape, output_shape, filter_sizes):
        
        #If shape is a tensorflow shape object instead of a list, implicitly convert it
        if input_shape.__class__.__name__ == 'TensorShape':
            input_shape = input_shape.as_list()
        if output_shape.__class__.__name__ == 'TensorShape':
            output_shape = output_shape.as_list()
        
        assert len(output_shape) == 2, "Output must be 2-dimensional. (batch_size, num_labels)"
        assert len(input_shape) == 4, "Input must be 4-dimensional. (batch_size, len, width, depth)"
        
        return input_shape, output_shape, filter_sizes
            