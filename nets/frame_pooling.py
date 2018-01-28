# ------------------------------------------------------------------------------
# ActionVLAD: Learning spatio-temporal aggregation for action classification
# Copyright (c) 2017 Carnegie Mellon University and Adobe Systems Incorporated
# Please see LICENSE on https://github.com/rohitgirdhar/ActionVLAD/ for details
# ------------------------------------------------------------------------------
import numpy as np
import cPickle as pickle

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.platform import tf_logging as logging

FLAGS = tf.app.flags.FLAGS
# NetVLAD Parameters
tf.app.flags.DEFINE_float('netvlad_alpha', 1000.0,
                          """Alpha to use for netVLAD.""")


def softmax(target, axis, name=None):
    with tf.name_scope(name, 'softmax', [target]):
        max_axis = tf.reduce_max(target, axis, keep_dims=True)
        target_exp = tf.exp(target-max_axis)
        normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
        softmax = target_exp / normalize
        return softmax


def netvlad(net, videos_per_batch, weight_decay, netvlad_initCenters):
    end_points = {}
    # VLAD pooling
    try:
      netvlad_initCenters = int(netvlad_initCenters)
      # initialize the cluster centers randomly
      cluster_centers = np.random.normal(size=(
        netvlad_initCenters, net.get_shape().as_list()[-1]))
      logging.info('Randomly initializing the {} netvlad cluster '
                   'centers'.format(cluster_centers.shape))
    except ValueError:
      with open(netvlad_initCenters, 'rb') as fin:
        kmeans = pickle.load(fin)
        cluster_centers = kmeans.cluster_centers_
    with tf.variable_scope('NetVLAD'):
        # normalize features
        net_normed = tf.nn.l2_normalize(net, 3, name='FeatureNorm')
        end_points[tf.get_variable_scope().name + '/net_normed'] = net_normed
        vlad_centers = slim.model_variable(
            'centers',
            shape=cluster_centers.shape,
            initializer=tf.constant_initializer(cluster_centers),
            regularizer=slim.l2_regularizer(weight_decay))
        end_points[tf.get_variable_scope().name + '/vlad_centers'] = vlad_centers
        vlad_W = slim.model_variable(
            'vlad_W',
            shape=(1, 1, ) + cluster_centers.transpose().shape,
            initializer=tf.constant_initializer(
                cluster_centers.transpose()[np.newaxis, np.newaxis, ...] *
                2 * FLAGS.netvlad_alpha),
            regularizer=slim.l2_regularizer(weight_decay))
        end_points[tf.get_variable_scope().name + '/vlad_W'] = vlad_W
        vlad_B = slim.model_variable(
            'vlad_B',
            shape=cluster_centers.shape[0],
            initializer=tf.constant_initializer(
                -FLAGS.netvlad_alpha *
                np.sum(np.square(cluster_centers), axis=1)),
            regularizer=slim.l2_regularizer(weight_decay))
        end_points[tf.get_variable_scope().name + '/vlad_B'] = vlad_B
        conv_output = tf.nn.conv2d(net_normed, vlad_W, [1, 1, 1, 1], 'VALID')
        dists = tf.nn.bias_add(conv_output, vlad_B)
        assgn = softmax(dists, axis=3)
        end_points[tf.get_variable_scope().name + '/assgn'] = assgn

        vid_splits = tf.split(0, videos_per_batch, net_normed)
        assgn_splits = tf.split(0, videos_per_batch, assgn)
        num_vlad_centers = vlad_centers.get_shape()[0]
        vlad_centers_split = tf.split(0, num_vlad_centers, vlad_centers)
        final_vlad = []
        for feats, assgn in zip(vid_splits, assgn_splits):
            vlad_vectors = []
            assgn_split_byCluster = tf.split(3, num_vlad_centers, assgn)
            for k in range(num_vlad_centers):
                res = tf.reduce_sum(
                    tf.mul(tf.sub(
                    feats,
                    vlad_centers_split[k]), assgn_split_byCluster[k]),
                    [0, 1, 2])
                vlad_vectors.append(res)
            vlad_vectors_frame = tf.pack(vlad_vectors, axis=0)
            final_vlad.append(vlad_vectors_frame)
        vlad_rep = tf.pack(final_vlad, axis=0, name='unnormed-vlad')
        end_points[tf.get_variable_scope().name + '/unnormed_vlad'] = vlad_rep
        with tf.name_scope('intranorm'):
            intranormed = tf.nn.l2_normalize(vlad_rep, dim=2)
        end_points[tf.get_variable_scope().name + '/intranormed_vlad'] = intranormed
        with tf.name_scope('finalnorm'):
            vlad_rep = tf.nn.l2_normalize(tf.reshape(
                intranormed,
                [intranormed.get_shape().as_list()[0], -1]),
                dim=1)
    return vlad_rep, end_points



def seqvlad(net, videos_per_batch, weight_decay, netvlad_initCenters):
    # Sequential VLAD pooling
    end_points = {}
    try:
      print('net shape():', net.get_shape().as_list())
      netvlad_initCenters = int(netvlad_initCenters)
      # initialize the cluster centers randomly
      cluster_centers = np.random.normal(size=(
        netvlad_initCenters, net.get_shape().as_list()[-1]))
      logging.info('Randomly initializing the {} netvlad cluster '
                   'centers'.format(cluster_centers.shape))
    except ValueError:

      print('<netvlad_initCenters> must be a [interger] for <seqvlad> pooling types ...')
      exit()
      # with open(netvlad_initCenters, 'rb') as fin:
      #   kmeans = pickle.load(fin)
      #   cluster_centers = kmeans.cluster_centers_
    with tf.variable_scope('SeqVLAD'):
        # normalize features
        net_normed = tf.nn.l2_normalize(net, 3, name='FeatureNorm')
        end_points[tf.get_variable_scope().name + '/net_normed'] = net_normed

        # model parameters
        centers_num = 64
        # share_w
        share_w = slim.model_variable('share_w',
                              shape=[3, 3, 512, centers_num], #[filter_height, filter_width, in_channels, out_channels]
                              initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                              regularizer=slim.l2_regularizer(weight_decay),
                              )
        share_b = slim.model_variable('share_b',
                              shape=[64],
                              initializer=tf.truncated_normal_initializer(stddev=0.1),
                              regularizer=slim.l2_regularizer(weight_decay))

        centers = slim.model_variable('centers',
                              shape=[1,512,centers_num],
                              initializer=tf.constant_initializer(cluster_centers),
                              regularizer=slim.l2_regularizer(weight_decay),
                              )

        U_z = slim.model_variable('U_z',
                              shape=[3, 3, centers_num, centers_num], #[filter_height, filter_width, in_channels, out_channels]
                              initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                              regularizer=slim.l2_regularizer(weight_decay),
                              )
        U_r = slim.model_variable('U_r',
                              shape=[3, 3, centers_num, centers_num], #[filter_height, filter_width, in_channels, out_channels]
                              initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                              regularizer=slim.l2_regularizer(weight_decay),
                              )
        U_h = slim.model_variable('U_h',
                              shape=[3, 3, centers_num, centers_num], #[filter_height, filter_width, in_channels, out_channels]
                              initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                              regularizer=slim.l2_regularizer(weight_decay),
                              )

        # add parameters to end_poins
        end_points[tf.get_variable_scope().name + '/share_w'] = share_w
        end_points[tf.get_variable_scope().name + '/share_b'] = share_b
        end_points[tf.get_variable_scope().name + '/centers'] = centers
        end_points[tf.get_variable_scope().name + '/U_z'] = U_z
        end_points[tf.get_variable_scope().name + '/U_r'] = U_r
        end_points[tf.get_variable_scope().name + '/U_h'] = U_h

        # seqvlad 
        input_shape = net.get_shape().as_list()
        timesteps = input_shape[0]//videos_per_batch
        assert input_shape[0]%videos_per_batch==0
        # assignment = tf.reshape(net,[videos_per_batch, -1, input_shape[]])
        w_conv_x = tf.add(tf.nn.conv2d(net, share_w, [1,1,1,1], 'SAME', name='w_conv_x'),tf.reshape(share_b,[1, 1, 1, centers_num]))
        
        assignments = tf.reshape(w_conv_x,[videos_per_batch, -1, input_shape[1], input_shape[2], centers_num])
        print('assignments', assignments.get_shape().as_list())


        axis = [1,0]+list(range(2,5))  # axis = [1,0,2]
        assignments = tf.transpose(assignments, perm=axis)

        input_assignments = tf.TensorArray(
                dtype=assignments.dtype,
                size=timesteps,
                tensor_array_name='input_assignments')
        if hasattr(input_assignments, 'unstack'):
          input_assignments = input_assignments.unstack(assignments)
        else:
          input_assignments = input_assignments.unpack(assignments)  

        hidden_states = tf.TensorArray(
                dtype=tf.float32,
                size=timesteps,
                tensor_array_name='hidden_states')

        def get_init_state(x, output_dims):

          initial_state = tf.zeros_like(x)
          initial_state = tf.reduce_sum(initial_state,axis=[0,4])
          initial_state = tf.expand_dims(initial_state,dim=-1)
          initial_state = tf.tile(initial_state,[1,1,1,output_dims])
          return initial_state
        def step(time, hidden_states, h_tm1):
          assign_t = input_assignments.read(time) # batch_size * dim
          print('h_tm1', h_tm1.get_shape().as_list())
          r = tf.nn.sigmoid(assign_t+ tf.nn.conv2d(h_tm1, U_r, [1,1,1,1], 'SAME', name='r'))
          z = tf.nn.sigmoid(assign_t+ tf.nn.conv2d(h_tm1, U_z, [1,1,1,1], 'SAME', name='z'))

          hh = tf.tanh(assign_t+ tf.nn.conv2d(r*h_tm1, U_h,  [1,1,1,1], 'SAME', name='hh'))

          h = (1-z)*hh + z*h_tm1
        
          hidden_states = hidden_states.write(time, h)

          return (time+1,hidden_states, h)

        time = tf.constant(0, dtype='int32', name='time')
        print('assignments', assignments.get_shape().as_list())
        initial_state = get_init_state(assignments,centers_num)
        print('initial_state', initial_state.get_shape().as_list())

        feature_out = tf.while_loop(
                cond=lambda time, *_: time < timesteps,
                body=step,
                loop_vars=(time, hidden_states, initial_state ),
                parallel_iterations=32,
                swap_memory=True)


        hidden_states = feature_out[-2]
        if hasattr(hidden_states, 'stack'):
          assignment = hidden_states.stack()
        else:
          assignment = hidden_states.pack()

        
        
        axis = [1,0]+list(range(2,5))  # axis = [1,0,2]
        assignment = tf.transpose(assignment, perm=axis)


        assignment = tf.reshape(assignment,[-1,input_shape[1]*input_shape[2],centers_num])

        # assignment = tf.nn.softmax(assignment,dim=-1)

        # for alpha * c
        a_sum = tf.reduce_sum(assignment,-2,keep_dims=True)
        a = tf.multiply(a_sum,centers)
        # for alpha * x
        assignment = tf.transpose(assignment,perm=[0,2,1])
        net = tf.reshape(net,[-1,input_shape[1]*input_shape[2],input_shape[3]])
        vlad = tf.matmul(assignment,net)
        vlad = tf.transpose(vlad, perm=[0,2,1])

        # for differnce
        vlad = tf.subtract(vlad,a)

        vlad = tf.reshape(vlad,[videos_per_batch, -1, input_shape[3], centers_num])
        vlad_rep = tf.reduce_sum(vlad, axis=1)

        end_points[tf.get_variable_scope().name + '/unnormed_vlad'] = vlad_rep
        with tf.name_scope('intranorm'):
          vlad_rep = tf.nn.l2_normalize(vlad_rep, 1)
        end_points[tf.get_variable_scope().name + '/intranormed_vlad'] = vlad_rep
        with tf.name_scope('finalnorm'):
          vlad_rep = tf.reshape(vlad_rep,[-1, input_shape[3]*centers_num])
          vlad_rep = tf.nn.l2_normalize(vlad_rep,-1)


    return vlad_rep, end_points
def pool_conv(net, videos_per_batch, type='avg'):
    """
    Pool all the features across the frame and across all the frames
    for the video to get a single representation.
    Useful as a way to debug NetVLAD, as this should be worse than 
    NetVLAD with k = 1.
    """
    if type == 'avg':
      method = tf.reduce_mean
    elif type == 'max':
      method = tf.reduce_max
    else:
      raise ValueError('Not Found')
    with tf.name_scope('%s-conv' % type):
        vid_splits = tf.split(0, videos_per_batch, net);
        vids_pooled = [method(vid, [0, 1, 2]) for vid in vid_splits]
        return tf.pack(vids_pooled, axis=0)
