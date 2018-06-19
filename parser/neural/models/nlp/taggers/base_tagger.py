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

import re
import codecs
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from parser.misc.colors import ctext, color_pattern
from parser.neural.models.nn import NN

#***************************************************************
class BaseTagger(NN):
  """"""
  
  PAD = 0
  ROOT = 1
  
  #=============================================================
  def __call__(self, vocabs, moving_params=None):
    """"""
    
    self.moving_params = moving_params
    if isinstance(vocabs, dict):
      self.vocabs = vocabs
    else:
      self.vocabs = {vocab.name: vocab for vocab in vocabs}
    
    input_vocabs = [self.vocabs[name] for name in self.input_vocabs]
    embed = self.embed_concat(input_vocabs)
    for vocab in self.vocabs.values():
      if vocab not in input_vocabs:
        vocab.generate_placeholder()
    placeholder = self.vocabs['words'].placeholder
    if len(placeholder.get_shape().as_list()) == 3:
      placeholder = placeholder[:,:,0]
    self._tokens_to_keep = tf.to_float(tf.greater(placeholder, self.ROOT))
    self._batch_size = tf.shape(placeholder)[0]
    self._bucket_size = tf.shape(placeholder)[1]
    self._sequence_lengths = tf.reduce_sum(tf.to_int32(tf.greater(placeholder, self.PAD)), axis=1)
    self._n_tokens = tf.to_int32(tf.reduce_sum(self.tokens_to_keep))
    
    top_recur = embed
    for i in xrange(self.n_layers):
      with tf.variable_scope('RNN%d' % i):
        top_recur, _ = self.RNN(top_recur, self.recur_size)
    return top_recur
  
  #=============================================================
  def process_accumulators(self, accumulators, time=None):
    """"""
    #'n_tokens', 'n_seqs', 'loss', 'n_tags_correct', 'n_tags_seqs_correct', 'n_xtags_correct', 'n_xtags_seqs_correct'
    if 'xtags' in self.output_vocabs:
      n_tokens, n_seqs, loss, tags_corr, tags_seq_corr, xtags_corr, xtags_seq_corr = accumulators
    else:
      n_tokens, n_seqs, loss, tags_corr, tags_seq_corr = accumulators
    acc_dict = {
      'Loss': loss,
      'Tags-TS': tags_corr/n_tokens*100,
      'Tags-SS': tags_seq_corr/n_seqs*100,
    }
    if 'xtags' in self.output_vocabs:
      acc_dict.update({
        'XTags-TS': xtags_corr/n_tokens*100,
        'XTags-SS': xtags_seq_corr/n_seqs*100,
      })
    if time is not None:
      acc_dict.update({
        'Token_rate': n_tokens / time,
        'Seq_rate': n_seqs / time,
      })
    return acc_dict
  
  #=============================================================
  def update_history(self, history, accumulators):
    """"""
    
    acc_dict = self.process_accumulators(accumulators)
    for key, value in acc_dict.iteritems():
      history[key].append(value)
    return history['Tags-TS'][-1]
  
  #=============================================================
  def print_accuracy(self, accumulators, time, prefix='Train'):
    """"""
    
    acc_dict = self.process_accumulators(accumulators, time=time)
    strings = []
    strings.append('Loss: {Loss:7.3f}')
    strings.append('TS: Tags: {Tags-TS:5.2f}%')
    if 'xtags' in self.output_vocabs:
      strings.append('XTags: {XTags-TS:5.2f}%')
    strings.append('SS: Tags: {Tags-SS:5.2f}%')
    if 'xtags' in self.output_vocabs:
      strings.append('XTags: {XTags-SS:5.2f}%')
    strings.append('Speed: {Seq_rate:6.1f} seqs/sec')
    string = '{0}  ' + ' | '.join(strings)
    print(string.format(prefix, **acc_dict))
    return
  
  #=============================================================
  def plot(self, history, prefix='Train'):
    """"""
    
    pass
  
  #=============================================================
  def check(self, preds, sents, fileobj):
    """"""
    if 'xtags' in self.output_vocabs:
      for tokens, tags_preds, xtags_preds in zip(sents, preds[0], preds[1]):
        for token, tags_pred, xtags_pred in zip(zip(*tokens), tags_preds, xtags_preds):
          tag = self.vocabs['tags'][tags_pred]
          xtag = self.vocabs['xtags'][xtags_pred]
          fileobj.write('\t'.join(token+(tag, xtag))+'\n')
        fileobj.write('\n')
    else:
      for tokens, preds in zip(sents, preds[0]):
        for token, pred in zip(zip(*tokens), preds):
          tag = self.vocabs['tags'][pred]
          fileobj.write('\t'.join(token+(tag, ))+'\n')
        fileobj.write('\n')
    return

  #=============================================================
  def write_probs(self, sents, output_file, probs, inv_idxs, merge_lines):
    """"""
    
    # Turns list of tuples of tensors into list of matrices
    tag_probs = [tag_prob for batch in probs for tag_prob in batch[0]]
    tokens_to_keep = [weight for batch in probs for weight in batch[1]]
    if 'xtags' in self.output_vocabs:
      xtag_probs = [xtag_prob for batch in probs for xtag_prob in batch[2]]
    tokens = [sent for batch in sents for sent in batch]
    j = 0
    with codecs.open(output_file, 'w', encoding='utf-8', errors='ignore') as f:
      for i in inv_idxs:
        sent, tag_prob, weights = tokens[i], tag_probs[i], tokens_to_keep[i]
        sent = zip(*sent)
        tag_preds = np.argmax(tag_prob, axis=1)
        xtag_preds = np.zeros(len(tag_preds))
        if 'xtags' in self.output_vocabs:
          xtag_prob = xtag_probs[i]
          xtag_preds = np.argmax(xtag_prob, axis=1)
        merge_line = merge_lines[j]
        for token, tag_pred, xtag_pred, weight in zip(sent, tag_preds[1:], xtag_preds[1:], weights[1:]):
          token = list(token)
          if (int(token[0]) in merge_line.keys()):
            f.write(merge_line[int(token[0])]+'\n')
          token.insert(5, '_')
          token.append('_')
          token.append('_')
          token[3] = self.vocabs['tags'][tag_pred]
          if 'xtags' in self.output_vocabs:
            token[4] = self.vocabs['xtags'][xtag_pred]
          f.write('\t'.join(token)+'\n')
        j += 1
        if j < len(inv_idxs):
          f.write('\n')
    return
  
  #=============================================================
  def write_probs_ensemble(self, sents, output_file, multi_probs, inv_idxs, sum_type, merge_lines, sum_weights=None):
    """"""

    # Turns list of tuples of tensors into list of matrices
    multi_tag_probs = []
    multi_xtag_probs = []
    multi_tokens_to_keep = []
    for probs in multi_probs:
      # Turns list of tuples of tensors into list of matrices
      multi_tag_probs.append([tag_prob for batch in probs for tag_prob in batch[0]])
      multi_tokens_to_keep.append([weight for batch in probs for weight in batch[1]])
      if 'xtags' in self.output_vocabs:
        multi_xtag_probs.append([xtag_prob for batch in probs for xtag_prob in batch[2]])
    tokens = [sent for batch in sents for sent in batch]

    j = 0
    with codecs.open(output_file, 'w', encoding='utf-8', errors='ignore') as f:
      for i in inv_idxs:
        #sent, tag_prob, weights = tokens[i], tag_probs[i], tokens_to_keep[i]
        weights = multi_tokens_to_keep[0][i]
        sent = tokens[i]
        sent = zip(*sent)
        tag_prob = multi_tag_probs[0][i]
        for n in range(1,len(multi_tag_probs)):
          tag_prob += multi_tag_probs[n][i]
        tag_preds = np.argmax(tag_prob, axis=1)
        xtag_preds = np.zeros(len(tag_preds))
        if 'xtags' in self.output_vocabs:
          xtag_prob = multi_xtag_probs[0][i]
          for n in range(1,len(multi_xtag_probs)):
            xtag_prob += multi_xtag_probs[n][i]
          #xtag_prob = xtag_probs[i]
          xtag_preds = np.argmax(xtag_prob, axis=1)
        merge_line = merge_lines[j]
        for token, tag_pred, xtag_pred, weight in zip(sent, tag_preds[1:], xtag_preds[1:], weights[1:]):
          token = list(token)
          if (int(token[0]) in merge_line.keys()):
            f.write(merge_line[int(token[0])]+'\n')
          token.insert(5, '_')
          token.append('_')
          token.append('_')
          token[3] = self.vocabs['tags'][tag_pred]
          if 'xtags' in self.output_vocabs:
            token[4] = self.vocabs['xtags'][xtag_pred]
          f.write('\t'.join(token)+'\n')
        j += 1
        if j < len(inv_idxs):
          f.write('\n')
    return

  #=============================================================
  @property
  def train_keys(self):
    if 'xtags' in self.output_vocabs:
      return ('n_tokens', 'n_seqs', 'loss', 'n_tags_correct', 'n_tags_seqs_correct', 'n_xtags_correct', 'n_xtags_seqs_correct')
    else:
      return ('n_tokens', 'n_seqs', 'loss', 'n_tags_correct', 'n_tags_seqs_correct')
  
  #=============================================================
  @property
  def valid_keys(self):
    if 'xtags' in self.output_vocabs:
      return ('tags_preds', 'xtags_preds')
    else:
      return ('tags_preds', )

  #=============================================================
  @property
  def parse_keys(self):
    if 'xtags' in self.output_vocabs:
      return ('tags_probs', 'tokens_to_keep', 'xtags_probs')
    else:
      return ('tags_probs', 'tokens_to_keep')

  #=============================================================
  @property
  def ensemble_keys(self):
    return self.parse_keys
