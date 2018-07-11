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
import re
import codecs
from collections import Counter

import numpy as np
import tensorflow as tf

from parser.vocabs.base_vocab import BaseVocab
from parser.misc.zipf import Zipf

__all__ = ['WordVocab', 'LemmaVocab', 'TagVocab', 'XTagVocab', 'RelVocab', 'PositionVocab']

#***************************************************************
class TokenVocab(BaseVocab):
  """"""
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
     
    recount = kwargs.pop('recount', False)
    initialize_zero = kwargs.pop('initialize_zero', True)
    super(TokenVocab, self).__init__(*args, **kwargs)
    #print ("TokenVocab, __init__, recount:%s, self.filename:%s"%(recount, self.filename))
    if recount:
      self.count()
    else:
      if os.path.isfile(self.filename):
        self.load()
      else:
        self.count()
        self.dump()
    self.index_vocab()
    
    embed_dims = [len(self), self.embed_size]
    if initialize_zero:
      self.embeddings = np.zeros(embed_dims)
    else:
      self.embeddings = np.random.randn(*embed_dims)
    return
  
  #=============================================================
  def count(self, conll_files=None):
    """""" 
    #print ("TokenVocab, count")
    if conll_files is None:
      conll_files = self.train_files
    #print ("token_vocab, count, self.train_files: ", self.train_files)
    for conll_file in conll_files:
      #print ("token_vocab count : %s"%(conll_file))
      with codecs.open(conll_file, encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f):
          try:
            line = line.strip()
            if line and not line.startswith('#'):
              line = line.split('\t')
              if not self.data_type == 'sem15':
                assert len(line) == 10
              token = line[self.conll_idx]
              if not self.cased:
                token = token.lower()
              self.counts[token] += 1
          except:
            raise ValueError('File %s is misformatted at line %d' % (conll_file, line_num+1))
    return
  
  #=============================================================
  def load(self):
    """"""
    
    with codecs.open(self.filename, encoding='utf-8') as f:
      for line_num, line in enumerate(f):
        try:
          line = line.strip()
          if line:
            line = line.split('\t')
            token, count = line
            self.counts[token] = int(count)
        except:
          raise ValueError('File %s is misformatted at line %d' % (train_file, line_num+1))
    return
  
  #=============================================================
  def dump(self):
    """"""
    
    with codecs.open(self.filename, 'w', encoding='utf-8') as f:
      for word, count in self.sorted_counts(self.counts):
        f.write('%s\t%d\n' % (word, count))
    return
  
  #=============================================================
  def index_vocab(self):
    """"""
    
    for token, count in self.sorted_counts(self.counts):
      if ((count >= self.min_occur_count) and
          token not in self and 
          (not self.max_rank or len(self) < self.max_rank)):
        self[token] = len(self)
    return
  
  #=============================================================
  def fit_to_zipf(self, plot=True):
    """"""
    print ("fit_to_zipf,len: %d" % (len(self.counts))) 
    zipf = Zipf.from_configurable(self, self.counts, name='zipf-%s'%self.name)
    if plot:
      zipf.plot()
    return zipf
  
  #=============================================================
  @staticmethod
  def sorted_counts(counts):
    return sorted(counts.most_common(), key=lambda x: (-x[1], x[0]))
  
  #=============================================================
  @property
  def conll_idx(self):
    return self._conll_idx

#***************************************************************
class PositionVocab(TokenVocab):
  _conll_idx = 0
class WordVocab(TokenVocab):
  _conll_idx = 1
class LemmaVocab(WordVocab):
  _conll_idx = 2
class TagVocab(TokenVocab):
  _conll_idx = 3
class XTagVocab(TagVocab):
  _conll_idx = 4

  #=============================================================
  @property
  def conll_idx(self):
    if self.data_type == 'sem15':
      return 3
    else:
      return 4

class RelVocab(TokenVocab):
  _conll_idx = 7

  #=============================================================
  def index(self, token):
    if isinstance(token, list):
      return [(int(tok[0]), super(RelVocab, self).index(tok[1])) for tok in token]
    else:
      return super(RelVocab, self).index(token)

  #=============================================================
  def generate_placeholder(self):
    """"""

    if self.placeholder is None:
      if self.data_form == 'graph':
        self.placeholder = tf.placeholder(tf.int32, shape=[None, None, None], name=self.name)
      else:
        self.placeholder = tf.placeholder(tf.int32, shape=[None, None], name=self.name)
    return self.placeholder
  
  #=============================================================
  def count(self, conll_files=None):
    """""" 
    if not self.data_type == 'sem15':
      super(RelVocab, self).count()
      return
    if conll_files is None:
      conll_files = self.train_files
    #print ("token_vocab, count, self.train_files: ", self.train_files)
    for conll_file in conll_files:
      #print ("token_vocab count : %s"%(conll_file))
      with codecs.open(conll_file, encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f):
          try:
            line = line.strip()
            if line and not line.startswith('#'):
              line = line.split('\t')
              for token in line[7:]:
                if token == '_': continue
                if not self.cased:
                  token = token.lower()
                self.counts[token] += 1
          except:
            raise ValueError('File %s is misformatted at line %d' % (conll_file, line_num+1))
    return

#***************************************************************
if __name__ == '__main__':
  """"""
  
  from parser import Configurable
  from parser.vocabs import PretrainedVocab, TokenVocab, WordVocab
  
  configurable = Configurable()
  if os.path.isfile('saves/defaults/words.txt'):
    os.remove('saves/defaults/words.txt')
  token_vocab = WordVocab.from_configurable(configurable, 1)
  token_vocab = WordVocab.from_configurable(configurable, 1)
  token_vocab.fit_to_zipf()
  #pretrained_vocab = PretrainedVocab.from_vocab(token_vocab)
  #assert min(pretrained_vocab.counts.values()) == 1
  print('TokenVocab passed')
