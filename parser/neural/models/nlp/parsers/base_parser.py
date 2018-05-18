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
from parser.misc.mst import nonprojective, argmax
from parser.neural.models.nn import NN

#***************************************************************
class BaseParser(NN):
  """"""
  
  PAD = 0
  ROOT = 1
  
  #=============================================================
  def __call__(self, vocabs, ts_lstm=None, stacked_cnn=None, moving_params=None):
    """"""
    
    self.moving_params = moving_params
    if isinstance(vocabs, dict):
      self.vocabs = vocabs
    else:
      self.vocabs = {vocab.name: vocab for vocab in vocabs}
    #print ("base_parser.py(__call__):vocabs:",self.vocabs) 
    input_vocabs = [self.vocabs[name] for name in self.input_vocabs]
    #print ("base_parser.py(__call__):input_vocabs:",input_vocabs)
    #exit()
    #embed = tf.concat([vocab(moving_params=self.moving_params) for vocab in input_vocabs], 2)
    embed = self.embed_concat(input_vocabs)
    for vocab in self.vocabs.values():
      if vocab not in input_vocabs:
        vocab.generate_placeholder()
    placeholder = self.vocabs['words'].placeholder
    #print ("base_parser.py(__call__):placeholder:",placeholder.get_shape())
    #exit()
    if len(placeholder.get_shape().as_list()) == 3:
      placeholder = placeholder[:,:,0]
    self._tokens_to_keep = tf.to_float(tf.greater(placeholder, self.ROOT))
    self._batch_size = tf.shape(placeholder)[0]
    self._bucket_size = tf.shape(placeholder)[1]
    self._sequence_lengths = tf.reduce_sum(tf.to_int32(tf.greater(placeholder, self.PAD)), axis=1)
    self._n_tokens = tf.to_int32(tf.reduce_sum(self.tokens_to_keep))
    #print ("_tok_to_keep:",self._tokens_to_keep,"batch:",self._batch_size,"bucket:",self._bucket_size,"seq_len:",self._sequence_lengths,"tokens:",self._n_tokens)
    #exit()
    top_rep = embed
    if ts_lstm:
      top_rep = ts_lstm(top_rep, self.recur_size, placeholder, self.moving_params)
    elif stacked_cnn:
      top_rep = stacked_cnn(top_rep, self.conv_size, placeholder, self.moving_params)
    else:
      for i in xrange(self.n_layers):
        with tf.variable_scope('RNN%d' % i):
          top_rep, _ = self.RNN(top_rep, self.recur_size)
    #print ("top_rep:", top_rep.get_shape())
    return top_rep
  
  #=============================================================
  def process_accumulators(self, accumulators, time=None):
    """"""
    # 'n_tokens', 'n_seqs', 'loss', 'n_rel_correct', 'n_arc_correct', 'n_correct', 'n_seqs_correct'
    n_tokens, n_seqs, loss, rel_corr, arc_corr, corr, seq_corr = accumulators
    acc_dict = {
      'Loss': loss,
      'LS': rel_corr/n_tokens*100,
      'UAS': arc_corr/n_tokens*100,
      'LAS': corr/n_tokens*100,
      'SS': seq_corr/n_seqs*100,
    }
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
    return history['LAS'][-1]
  
  #=============================================================
  def print_accuracy(self, accumulators, time, prefix='Train'):
    """"""
    
    acc_dict = self.process_accumulators(accumulators, time=time)
    strings = []
    strings.append('Loss: {Loss:7.3f}')
    strings.append('LS: {LS:5.2f}%')
    strings.append('UAS: {UAS:5.2f}%')
    strings.append('LAS: {LAS:5.2f}%')
    strings.append('SS: {SS:5.2f}%')
    strings.append('Speed: {Seq_rate:6.1f} seqs/sec')
    string = '{0}  ' + ' | '.join(strings)
    #print(string.format(prefix, **acc_dict))
    return (string.format(prefix, **acc_dict))
  
  #=============================================================
  def plot(self, history, prefix='Train'):
    """"""
    
    pass
  
  #=============================================================
  def check(self, preds, sents, fileobj):
    """"""

    for tokens, arc_preds, rel_preds in zip(sents, preds[0], preds[1]):
      for token, arc_pred, rel_pred in zip(zip(*tokens), arc_preds, rel_preds):
        arc = self.vocabs['heads'][arc_pred]
        rel = self.vocabs['rels'][rel_pred]
        fileobj.write('\t'.join(token+(arc, rel))+'\n')
      fileobj.write('\n')
    return

  #=============================================================
  def write_probs(self, sents, output_file, probs, inv_idxs, merge_lines):
    """"""
    
    #parse_algorithm = self.parse_algorithm 
    
    # Turns list of tuples of tensors into list of matrices
    arc_probs = [arc_prob for batch in probs for arc_prob in batch[0]]
    rel_probs = [rel_prob for batch in probs for rel_prob in batch[1]]
    tokens_to_keep = [weight for batch in probs for weight in batch[2]]
    tokens = [sent for batch in sents for sent in batch]
    
    with codecs.open(output_file, 'w', encoding='utf-8', errors='ignore') as f:
      j = 0
      for i in inv_idxs:
        sent, arc_prob, rel_prob, weights = tokens[i], arc_probs[i], rel_probs[i], tokens_to_keep[i]
        sent = zip(*sent)
        sequence_length = int(np.sum(weights))+1
        arc_prob = arc_prob[:sequence_length][:,:sequence_length]
        #arc_preds = np.argmax(arc_prob, axis=1)
        arc_preds = nonprojective(arc_prob)
        arc_preds_one_hot = np.zeros([rel_prob.shape[0], rel_prob.shape[2]])
        arc_preds_one_hot[np.arange(len(arc_preds)), arc_preds] = 1.
        rel_preds = np.argmax(np.einsum('nrb,nb->nr', rel_prob, arc_preds_one_hot), axis=1)
        merge_line = merge_lines[j]
        for token, arc_pred, rel_pred, weight in zip(sent, arc_preds[1:], rel_preds[1:], weights[1:]):
          token = list(token)
          # add the merge line
          if (int(token[0]) in merge_line.keys()):
          	f.write(merge_line[int(token[0])]+'\n')
          token.insert(5, '_')
          token.append('_')
          token.append('_')
          token[6] = self.vocabs['heads'][arc_pred]
          token[7] = self.vocabs['rels'][rel_pred]
          f.write('\t'.join(token)+'\n')
        j += 1
        if j < len(inv_idxs):
          f.write('\n')
    return
  
  #=============================================================
  def write_probs_ensemble(self, sents, output_file, multi_probs, inv_idxs, sum_type, merge_lines, sum_weights=None):
    """"""
    assert(sum_type in ["prob", "score"])
    #parse_algorithm = self.parse_algorithm
    multi_arc_probs = []
    multi_rel_probs = []
    multi_tokens_to_keep = []
    if (sum_type == "prob"):
    	for probs in multi_probs:
    		# Turns list of tuples of tensors into list of matrices
    		multi_arc_probs.append([arc_prob for batch in probs for arc_prob in batch[0]])
    		multi_rel_probs.append([rel_prob for batch in probs for rel_prob in batch[1]])
    		multi_tokens_to_keep.append([weight for batch in probs for weight in batch[2]])
    elif (sum_type == "score"):
    	for probs in multi_probs:
    		# Turns list of tuples of tensors into list of matrices
    		multi_arc_probs.append([arc_prob for batch in probs for arc_prob in batch[3]])
    		multi_rel_probs.append([rel_prob for batch in probs for rel_prob in batch[4]])
    		multi_tokens_to_keep.append([weight for batch in probs for weight in batch[2]])
    tokens = [sent for batch in sents for sent in batch]
    
    if sum_weights is None:
    	sum_weights = [1.0] * len(multi_arc_probs)
    with codecs.open(output_file, 'w', encoding='utf-8', errors='ignore') as f:
      j = 0
      for i in inv_idxs:
        #sent, arc_prob, rel_prob, weights = tokens[i], arc_probs[i], rel_probs[i], tokens_to_keep[i]
        weights = multi_tokens_to_keep[0][i]
        sent = tokens[i]
        sent = zip(*sent)
        sequence_length = int(np.sum(weights))+1
        arc_prob = multi_arc_probs[0][i][:sequence_length][:,:sequence_length]
        for n in range(1,len(multi_arc_probs)):
        	#print("adding for arc sent {},shape:{}".format(i,arc_prob.shape))
        	arc_prob += sum_weights[n] * multi_arc_probs[n][i][:sequence_length][:,:sequence_length]

        #arc_preds = np.argmax(arc_prob, axis=1)
        if (sum_type == "score"):
        	# softmax axis = 1
        	arc_prob = np.exp(arc_prob - np.max(arc_prob, axis = 1).reshape(arc_prob.shape[0],1))
        	arc_prob = arc_prob / np.sum(arc_prob, axis = 1).reshape(arc_prob.shape[0],1)
        arc_preds = nonprojective(arc_prob)
        rel_prob = multi_rel_probs[0][i]
        for n in range(1, len(multi_rel_probs)):
        	#print ("adding for rel sent {},shape:{}".format(i,rel_prob.shape))
        	rel_prob += sum_weights[n] * multi_rel_probs[n][i]

        arc_preds_one_hot = np.zeros([rel_prob.shape[0], rel_prob.shape[2]])
        arc_preds_one_hot[np.arange(len(arc_preds)), arc_preds] = 1.
        rel_preds = np.argmax(np.einsum('nrb,nb->nr', rel_prob, arc_preds_one_hot), axis=1)
        merge_line = merge_lines[j]
        for token, arc_pred, rel_pred, weight in zip(sent, arc_preds[1:], rel_preds[1:], weights[1:]):
          token = list(token)
          # add the merge line
          if (int(token[0]) in merge_line.keys()):
          	f.write(merge_line[int(token[0])]+'\n')
          token.insert(5, '_')
          token.append('_')
          token.append('_')
          token[6] = self.vocabs['heads'][arc_pred]
          token[7] = self.vocabs['rels'][rel_pred]
          f.write('\t'.join(token)+'\n')
        j += 1
        if j < len(inv_idxs):
          f.write('\n')
    return

  #=============================================================
  @property
  def train_keys(self):
    return ('n_tokens', 'n_seqs', 'loss', 'n_rel_correct', 'n_arc_correct', 'n_correct', 'n_seqs_correct')
  
  #=============================================================
  @property
  def hinge_keys(self):
    return ('arc_probs', 'tokens_to_keep', 'n_tokens', 'n_seqs', 'loss', 'n_rel_correct', 'n_arc_correct', 'n_correct', 'n_seqs_correct')

  #=============================================================
  @property
  def valid_keys(self):
    return ('arc_preds', 'rel_preds')

  #=============================================================
  @property
  def parse_keys(self):
    return ('arc_probs', 'rel_probs', 'tokens_to_keep')

  #=============================================================
  @property
  def ensemble_keys(self):
    return ('arc_probs', 'rel_probs', 'tokens_to_keep', 'arc_logits', 'rel_logits')
