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
import h5py
import re
from collections import Counter

import numpy as np
import tensorflow as tf

import parser.neural.linalg as linalg
from parser.vocabs.base_vocab import BaseVocab

#***************************************************************
class ElmoVocab(BaseVocab):
  """"""
  
  #=============================================================
  def __init__(self, token_vocab, elmo_file, files, *args, **kwargs):
    """"""
    
    super(ElmoVocab, self).__init__(*args, **kwargs)
    
    self._token_vocab = token_vocab

    self._elmo_file = elmo_file
    self._files = files

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
  @classmethod
  def from_vocab(cls, vocab, elmo_file, files, *args, **kwargs):
    """"""
    
    args += (vocab, elmo_file, files, )
    return cls.from_configurable(vocab, *args, **kwargs)

  #=============================================================
  def iter_sents(self, data_files):
    """"""
    for data_file in data_files:
      with codecs.open(data_file, encoding='utf-8', errors='ignore') as f:
        buff = []
        for line in f:
          line = line.strip()
          if line and not line.startswith('#'):
            if not re.match('[0-9]+[-.][0-9]+', line):
              buff.append(line.split('\t')[1])
          elif buff:
            yield buff
            buff = []
        if buff:
          yield buff

  #=============================================================
  def load(self):
    """"""
    embeddings = []
    cur_idx = len(self.special_tokens)

    if self.elmo_file:
      print ("### Loading ELMo for testset from {}! ###".format(self.elmo_file))
      with h5py.File(self.elmo_file, 'r') as f:
        for sid, sent in enumerate(self.iter_sents(self.files)):
          elmo = f['\t'.join(sent)].value
          assert(len(elmo) == len(sent))
          embeddings.extend(elmo)
          for wid in xrange(len(sent)):
            self["parseset-"+str(sid)+"-"+str(wid)] = cur_idx
            cur_idx += 1

    else:
      if self.filename:
        print ("### Loading ELMo for trainset from {}! ###".format(self.filename))
        with h5py.File(self.filename, 'r') as f:
          for sid, sent in enumerate(self.iter_sents(self.train_files)):
            elmo = f['\t'.join(sent)].value
            assert(len(elmo) == len(sent))
            embeddings.extend(elmo)
            for wid in xrange(len(sent)):
              self["trainset-"+str(sid)+"-"+str(wid)] = cur_idx
              cur_idx += 1

      if self.parse_filename:
        print ("### Loading ELMo for parseset from {}! ###".format(self.parse_filename))
        with h5py.File(self.parse_filename, 'r') as f:
          for sid, sent in enumerate(self.iter_sents(self.parse_files)):
            elmo = f['\t'.join(sent)].value
            assert(len(elmo) == len(sent))
            embeddings.extend(elmo)
            for wid in xrange(len(sent)):
              self["parseset-"+str(sid)+"-"+str(wid)] = cur_idx
              cur_idx += 1

    try:
      embeddings = np.stack(embeddings)
      embeddings = np.pad(embeddings, ( (len(self.special_tokens),0), (0,0) ), 'constant')
      self.embeddings = np.stack(embeddings)
    except:
      shapes = set([embedding.shape for embedding in embeddings])
      raise ValueError("Couldn't stack embeddings with shapes in %s" % shapes)
    return

  """
  #=============================================================
  def load(self):
    """"""
    
    embeddings = []
    cur_idx = len(self.special_tokens)
    max_rank = self.max_rank
    sid = 0
    wid = 0
    if self.filename:
      print ("### Loading ELMo for trainset from {}! ###".format(self.filename))
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
              self["trainset-"+str(sid)+"-"+str(wid)] = cur_idx
              wid += 1
              cur_idx += 1
            else:
              sid += 1
              wid = 0
          else:
            break
    sid = 0
    wid = 0
    if self.parse_filename:
      print ("### Loading ELMo for parseset from {}! ###".format(self.parse_filename))
      if self.parse_filename.endswith('.xz'):
        open_func = lzma.open
      else:
        open_func = codecs.open
      with open_func(self.parse_filename, 'rb') as f:
        reader = codecs.getreader('utf-8')(f, errors='ignore')
        if self.skip_header == True:
          reader.readline()
        for line_num, line in enumerate(reader):
          if (not max_rank) or line_num < max_rank:
            line = line.rstrip().split(' ')
            if len(line) > 1:
              embeddings.append(np.array(line[1:], dtype=np.float32))
              self["parseset-"+str(sid)+"-"+str(wid)] = cur_idx
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
  """

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
  # token = "trainset/parseset-sid-wid"
  def index(self, token):
    return self._tok2idx.get(token, self.UNK)

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
  @property
  def elmo_file(self):
    return self._elmo_file
  @property
  def files(self):
    return self._files

#***************************************************************
if __name__ == '__main__':
  """"""
  
  elmo_vocab = ElmoVocab(None)
  idx = elmo_vocab.index("trainset-1-2")
  emb = tf.nn.embedding_lookup(elmo_vocab.embeddings, idx)
  idx2 = elmo_vocab.index("parseset-1-1")
  emb2 = tf.nn.embedding_lookup(elmo_vocab.embeddings, idx2)
  with tf.Session() as sess:
    sess.run(elmo_vocab.embeddings.initializer)
    print ("get traiset sent 1 word 2 index:{}, it's embedding:\n{}".format(idx, sess.run(emb))) 
    print ("get parseset sent 1 word 1 index:{}, it's embedding:\n{}".format(idx2, sess.run(emb2))) 
 
  print('ElmoVocab passes')
