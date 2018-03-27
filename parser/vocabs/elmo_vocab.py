#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Timothy Dozat
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import codecs
import gzip
import warnings
try:
  from backports import lzma
except:
  warnings.warn('Install backports.lzma for xz support')
from collections import Counter

import numpy as np
import tensorflow as tf

import parser.neural.linalg as linalg
from parser.vocabs.base_vocab import BaseVocab

#***************************************************************
class ElmoVocab(BaseVocab):
  """"""
  
  #=============================================================
  def __init__(self, token_vocab, *args, **kwargs):
    """"""
    
    super(ElmoVocab, self).__init__(*args, **kwargs)
    
    self._token_vocab = token_vocab
    
    self.load()
    self.count()
    return
  
  #=============================================================
  def __call__(self, placeholder=None, moving_params=None):
    """"""
    
    embeddings = super(ElmoVocab, self).__call__(placeholder, moving_params=moving_params)
    # (n x b x d') -> (n x b x d)
    with tf.variable_scope(self.name.title()):
      matrix = linalg.linear(embeddings, self.token_embed_size, moving_params=moving_params)
      if moving_params is None:
        with tf.variable_scope('Linear', reuse=True):
          weights = tf.get_variable('Weights')
          tf.losses.add_loss(tf.nn.l2_loss(tf.matmul(tf.transpose(weights), weights) - tf.eye(self.token_embed_size)))
    return matrix
    #return embeddings # changed in saves2/test8
  
  #=============================================================
  def load(self):
    """"""
    
    embeddings = []
    cur_idx = len(self.special_tokens)
    max_rank = self.max_rank
    sid = 0
    wid = 0
    if self.filename.endswith('.xz'):
      open_func = lzma.open
    else:
      open_func = codecs.open
    with open_func(self.filename, 'rb') as f:
      reader = codecs.getreader('utf-8')(f, errors='ignore')
      if self.skip_header == True:
        reader.readline()
      for line_num, line in enumerate(reader):
        if (not max_rank) or line_num < max_rank:
          line = line.rstrip().split(' ')
          if len(line) > 1:
            embeddings.append(np.array(line[1:], dtype=np.float32))
            self[str(sid)+"-"+str(wid)] = cur_idx
            wid += 1
            cur_idx += 1
          else:
            sid += 1
            wid = 0
        else:
          break
    try:
      embeddings = np.stack(embeddings)
      embeddings = np.pad(embeddings, ( (len(self.special_tokens),0), (0,0) ), 'constant')
      self.embeddings = np.stack(embeddings)
    except:
      shapes = set([embedding.shape for embedding in embeddings])
      raise ValueError("Couldn't stack embeddings with shapes in %s" % shapes)
    return
  
  #=============================================================
  def count(self):
    """"""
    
    if self.token_vocab is not None:
      zipf = self.token_vocab.fit_to_zipf(plot=False)
      zipf_freqs = zipf.predict(np.arange(len(self))+1)
    else:
      zipf_freqs = -np.log(np.arange(len(self))+1)
    zipf_counts = zipf_freqs / np.min(zipf_freqs)
    for count, token in zip(zipf_counts, self.strings()):
      self.counts[token] = int(count)
    return
  #=============================================================
  def index(self, sid, wid):
    return self._tok2idx.get(str(sid)+"-"+str(wid), self.UNK)

  #=============================================================
  @property
  def token_vocab(self):
    return self._token_vocab
  @property
  def token_embed_size(self):
    return (self.token_vocab or self).embed_size
  @property
  def embeddings(self):
    return super(ElmoVocab, self).embeddings
  @embeddings.setter
  def embeddings(self, matrix):
    self._embed_size = matrix.shape[1]
    with tf.device('/cpu:0'):
      with tf.variable_scope(self.name.title()):
        self._embeddings = tf.Variable(matrix, name='Embeddings', trainable=False)
    return

#***************************************************************
if __name__ == '__main__':
  """"""
  
  elmo_vocab = ElmoVocab(None)
  idx = elmo_vocab.index(0,2)
  emb = tf.nn.embedding_lookup(elmo_vocab.embeddings, idx)
  idx2 = elmo_vocab.index(1,1)
  emb2 = tf.nn.embedding_lookup(elmo_vocab.embeddings, idx2)
  with tf.Session() as sess:
    sess.run(elmo_vocab.embeddings.initializer)
    print ("get sent 0 word 2 index:{}, it's embedding:\n{}".format(idx, sess.run(emb))) 
    print ("get sent 1 word 1 index:{}, it's embedding:\n{}".format(idx2, sess.run(emb2))) 
 
  print('ElmoVocab passes')
