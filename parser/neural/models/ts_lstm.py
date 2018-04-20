#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from parser.neural import linalg
from parser.configurable import Configurable
from parser.neural.models.nn import NN

#***************************************************************
class TS_LSTM(NN):
  """"""
  PAD = 0
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
    super(TS_LSTM, self).__init__(*args, **kwargs)
    print ('window:',self.window_size, ' stride:',self.time_stride)
    return

  #=============================================================
  # inputs: (batch_size, max_len, embed_size)
  def __call__(self, inputs, output_size, placeholder):
    """"""
    slided, s_seq_lens, remain, r_seq_lens = self.split_to_slide(inputs, placeholder)
    
    #"""
    for i in xrange(self.n_layers):
      with tf.variable_scope('TS_LSTM'):
        with tf.variable_scope('RNN%d' % i):
          slided, _ = self.RNN(slided, output_size, s_seq_lens)
        with tf.variable_scope('RNN%d' % i, reuse = True):
          remain, _ = self.RNN(remain, output_size, r_seq_lens)
    #"""
    return slided, s_seq_lens, remain, r_seq_lens

  #=============================================================
  def split_to_slide(self, inputs, placeholder):
    seq_len = tf.shape(inputs)[1]
    offset = tf.constant(self.time_stride)
    slided = tf.slice(inputs, [0, 0, 0], [-1, self.window_size, -1])
    #slided = tf.slice(inputs, [0, 0, 0], [-1, offset, -1])
    s_seq_lens = tf.reduce_sum(tf.to_int32(tf.greater(tf.slice(placeholder,[0,0],[-1,self.window_size]), self.PAD)), axis=1)

    def slide(offset, slided, s_seq_lens):
      #slided = tf.concat([slided, tf.slice(inputs, [0, offset, 0], [-1, self.window_size, -1])], 0)
      slided = tf.concat([slided, inputs[:, offset : offset+self.window_size, :]], 0)
      lens = tf.reduce_sum(tf.to_int32(tf.greater(tf.slice(placeholder,[0,offset],[-1,self.window_size]), self.PAD)), axis=1)
      s_seq_lens = tf.concat([s_seq_lens, lens], 0)
      return (offset + self.time_stride, slided, s_seq_lens)
    
    offset, slided, s_seq_lens = tf.while_loop(
      cond = lambda offset, _1, _2: offset + self.window_size - 1 < seq_len,
      body = slide,
      loop_vars = (offset, slided, s_seq_lens))
    
    #start = offset-self.time_stride+self.window_size
    #remain = tf.slice(inputs,[0, offset, 0],[-1, seq_len-offset, -1])
    remain = inputs[:, offset:, :]
    r_seq_lens = tf.reduce_sum(tf.to_int32(tf.greater(tf.slice(placeholder,[0,offset],[-1,seq_len-offset]), self.PAD)), axis=1)

    return (slided, s_seq_lens, remain, r_seq_lens)

  #=============================================================
  def RNN(self, inputs, output_size, sequence_lengths):
    """"""
    
    input_size = inputs.get_shape().as_list()[-1]
    cell = self.recur_cell.from_configurable(self, output_size, input_size=input_size, moving_params=self.moving_params)
    
    if self.moving_params is None:
      ff_keep_prob = self.ff_keep_prob
      recur_keep_prob = self.recur_keep_prob
    else:
      ff_keep_prob = 1
      recur_keep_prob = 1
    #print ("nn.py(RNN):cell:",cell,"self.rnn_func:",self.rnn_func)
    #exit() 
    top_recur, end_state = self.rnn_func(cell, inputs, sequence_lengths,
                                 ff_keep_prob=ff_keep_prob,
                                 recur_keep_prob=recur_keep_prob)
    return top_recur, end_state

  #=============================================================
  #@property
  #def window_size(self):
  #  return self._window_size
  #@property
  #def time_stride(self):
  #  return self._time_stride

if __name__ == "__main__":

  configurable = Configurable()
  ts = TS_LSTM.from_configurable(configurable)
  a = tf.placeholder(tf.float32, shape=[None, None, 4])
  holder = tf.placeholder(tf.int32, shape = [None, None])
  s, sl, r, rl = ts(a, 3, holder)

  a_1 = np.reshape(np.arange(40),[2,5,4])
  h_1 = np.array([[1,2,5,4,0],[2,3,4,0,0]])
  a_2 = np.reshape(np.arange(56),[2,7,4])
  h_2 = np.array([[1,2,5,4,1,0,0],[2,3,4,0,0,0,0]])
  print ('a_1:\n',a_1,'\na_2:\n',a_2)
  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    _s, _sl, _r, _rl = sess.run([s, sl, r, rl], feed_dict={a:a_1,holder:h_1})
    print('\ns:\n',_s,'\nsl:\n',_sl,'\nr:\n', _r,'\nrl:\n',_rl)
    _s, _sl, _r, _rl = sess.run([s, sl, r, rl], feed_dict={a:a_2,holder:h_2})
    print('\ns:\n',_s,'\nsl:\n',_sl,'\nr:\n', _r,'\nrl:\n',_rl)