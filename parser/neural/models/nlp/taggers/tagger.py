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

import numpy as np
import tensorflow as tf

from parser.neural.models.nlp.taggers.base_tagger import BaseTagger

#***************************************************************
class Tagger(BaseTagger):
  """"""
  
  #=============================================================
  def __call__(self, vocabs, moving_params=None):
    """"""
    
    top_recur = super(Tagger, self).__call__(vocabs, moving_params=moving_params)
    int_tokens_to_keep = tf.to_int32(self.tokens_to_keep)
    
    with tf.variable_scope('MLP'):
      with tf.variable_scope('Tag'):
        tags_mlp = self.MLP(top_recur, self.mlp_size)
      if 'xtags' in self.output_vocabs:
        with tf.variable_scope('XTag'):
          xtags_mlp = self.MLP(top_recur, self.mlp_size)
    
    with tf.variable_scope('Tag'):
      tags_logits = self.linear(tags_mlp, len(self.vocabs['tags']))
      tags_probs = tf.nn.softmax(tags_logits)
      tags_preds = tf.to_int32(tf.argmax(tags_logits, axis=-1))
      tags_targets = self.vocabs['tags'].placeholder
      tags_correct = tf.to_int32(tf.equal(tags_preds, tags_targets))*int_tokens_to_keep
      tags_loss = tf.losses.sparse_softmax_cross_entropy(tags_targets, tags_logits, self.tokens_to_keep)
      loss = tags_loss
    if 'xtags' in self.output_vocabs:
      with tf.variable_scope('XTag'):
        xtags_logits = self.linear(xtags_mlp, len(self.vocabs['xtags']))
        xtags_probs = tf.nn.softmax(xtags_logits)
        xtags_preds = tf.to_int32(tf.argmax(xtags_logits, axis=-1))
        xtags_targets = self.vocabs['xtags'].placeholder
        xtags_correct = tf.to_int32(tf.equal(xtags_preds, xtags_targets))*int_tokens_to_keep
        xtags_loss = tf.losses.sparse_softmax_cross_entropy(xtags_targets, xtags_logits, self.tokens_to_keep)
        loss += xtags_loss
    
    n_tags_correct = tf.reduce_sum(tags_correct)
    n_tags_seqs_correct = tf.reduce_sum(tf.to_int32(tf.equal(tf.reduce_sum(tags_correct, axis=1), self.sequence_lengths-1)))
    if 'xtags' in self.output_vocabs:
      with tf.variable_scope('XTag'):
        n_xtags_correct = tf.reduce_sum(xtags_correct)
        n_xtags_seqs_correct = tf.reduce_sum(tf.to_int32(tf.equal(tf.reduce_sum(xtags_correct, axis=1), self.sequence_lengths-1)))
    
    outputs = {
      'tags_logits': tags_logits,
      'tags_probs': tags_probs,
      'tags_preds': tags_preds,
      'tags_targets': tags_targets,
      'tags_correct': tags_correct,
      'n_tags_correct': n_tags_correct,
      'n_tags_seqs_correct': n_tags_seqs_correct,
      
      'loss': loss,
      'n_tokens': self.n_tokens,
      'n_seqs': self.batch_size,
      'tokens_to_keep': self.tokens_to_keep
    }

    if 'xtags' in self.output_vocabs:
      outputs.update({'xtags_logits': xtags_logits,
      'xtags_probs': xtags_probs,
      'xtags_preds': xtags_preds,
      'xtags_targets': xtags_targets,
      'xtags_correct': xtags_correct,
      'n_xtags_correct': n_xtags_correct,
      'n_xtags_seqs_correct': n_xtags_seqs_correct})
    
    return outputs
