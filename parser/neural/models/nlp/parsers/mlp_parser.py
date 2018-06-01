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

from parser.neural.models.nlp.parsers.base_parser import BaseParser

#***************************************************************
class MLPParser(BaseParser):
  """"""
  
  #=============================================================
  def __call__(self, vocabs, ts_lstm=None, arc_placeholder=None, stacked_cnn=None, moving_params=None):
    if self.data_form == 'tree':
      outputs = self.parse4tree(vocabs, ts_lstm, arc_placeholder, stacked_cnn, moving_params)
    elif self.data_form == 'graph':
      outputs = self.parse4graph(vocabs, ts_lstm, arc_placeholder, stacked_cnn, moving_params)
    return outputs

  #=============================================================
  def parse4tree(self, vocabs, ts_lstm=None, arc_placeholder=None, stacked_cnn=None, moving_params=None):
    
    top_rep = super(MLPParser, self).__call__(vocabs, ts_lstm, stacked_cnn, moving_params=moving_params)
    int_tokens_to_keep = tf.to_int32(self.tokens_to_keep)
    
    with tf.variable_scope('MLP'):
      dep_mlp, head_mlp = self.MLP(top_rep, self.arc_mlp_size + self.rel_mlp_size,
                                   n_splits=2)
      arc_dep_mlp, rel_dep_mlp = tf.split(dep_mlp, [self.arc_mlp_size, self.rel_mlp_size], axis=2)
      arc_head_mlp, rel_head_mlp = tf.split(head_mlp, [self.arc_mlp_size, self.rel_mlp_size], axis=2)
    
    with tf.variable_scope('Arc'):
      # (n x b x d) * (d x 1 x d) * (n x b x d).T -> (n x b x b)
      arc_logits = self.bilinear(arc_dep_mlp, arc_head_mlp, 1, add_bias2=False)
      # (n x b x b)
      arc_probs = tf.nn.softmax(arc_logits)
      # (n x b)
      arc_targets = self.vocabs['heads'].placeholder

      if arc_placeholder is not None:
        # (n x b)
        #arc_preds = tf.to_int32(tf.argmax(arc_logits, axis=-1))
        arc_preds = arc_placeholder
        # (n x b)
        arc_correct = tf.to_int32(tf.equal(arc_preds, arc_targets))*int_tokens_to_keep
        # (n x b) self.tokens_to_keep: (n x b)
        masked_arc_preds = tf.multiply(arc_preds, int_tokens_to_keep)
        masked_arc_targets = tf.multiply(arc_targets, int_tokens_to_keep)
        # (n x b x b)
        arc_preds_onehot = tf.one_hot(masked_arc_preds, self.bucket_size, on_value = True, off_value = False, dtype = tf.bool)
        arc_targets_onehot = tf.one_hot(masked_arc_targets, self.bucket_size, on_value = True, off_value = False, dtype = tf.bool)
        # (n x b)
        arc_preds_scores = tf.reshape(tf.boolean_mask(arc_logits, arc_preds_onehot), [self.batch_size, self.bucket_size])
        #arc_preds_scores = tf.reduce_max(arc_logits, axis = 2)
        arc_targets_scores = tf.reshape(tf.boolean_mask(arc_logits, arc_targets_onehot), [self.batch_size, self.bucket_size])
        #arc_targets_scores = tf.reduce_sum(tf.multiply(arc_logits, tf.to_float(arc_targets_onehot)), 2)
        # (n)
        arc_pred_scores = tf.reduce_sum(arc_preds_scores, 1)
        arc_target_scores = tf.reduce_sum(arc_targets_scores, 1)
        # (n)
        arc_losses = tf.subtract(arc_pred_scores, arc_target_scores)
        # ()
        arc_loss = tf.reduce_sum(arc_losses)
        # (n x b)
        #masked_margin = tf.to_float(tf.not_equal(masked_arc_preds, masked_arc_targets))
        # (n)
        #margin = tf.reduce_sum(masked_margin, axis = 1)
        # (n)
        #arc_losses += margin
        #arc_losses = np.add(arc_losses, 1)
      else:
        # (n x b)
        arc_preds = tf.to_int32(tf.argmax(arc_logits, axis=-1))
        # (n x b)
        arc_correct = tf.to_int32(tf.equal(arc_preds, arc_targets))*int_tokens_to_keep
        # ()
        arc_loss = tf.losses.sparse_softmax_cross_entropy(arc_targets, arc_logits, self.tokens_to_keep)
      
    with tf.variable_scope('Rel'):
      # (n x b x d) * (d x r x d) * (n x b x d).T -> (n x b x r x b)
      rel_logits = self.bilinear(rel_dep_mlp, rel_head_mlp, len(self.vocabs['rels']))
      # (n x b x r x b)
      rel_probs = tf.nn.softmax(rel_logits, dim=2)

      if arc_placeholder is not None:
        # (n x b x b) -> (n x b x b x 1)
        #arc_preds_onehot = tf.expand_dims(tf.to_float(arc_preds_onehot), axis=3)
        arc_targets_onehot = tf.expand_dims(tf.to_float(arc_targets_onehot), axis=3)
        # (n x b x r x b) * (n x b x b x 1) -> (n x b x r x 1)
        #select_rel_logits_preds = tf.matmul(rel_logits, arc_preds_onehot)
        select_rel_logits_targets = tf.matmul(rel_logits, arc_targets_onehot)
        # (n x b x r x 1) -> (n x b x r)
        #select_rel_logits_preds = tf.squeeze(select_rel_logits_preds, axis=3)
        select_rel_logits_targets = tf.squeeze(select_rel_logits_targets, axis=3)
        # (n x b)
        #rel_preds = tf.to_int32(tf.argmax(select_rel_logits_preds, axis=-1))
        # choose 1st and 2nd label scores from gold tree
        # (n x b x 2)
        rel_scores_top2, rel_preds_top2 = tf.nn.top_k(select_rel_logits_targets, 2)
        rel_preds_1st = rel_preds_top2[:,:,0]
        #rel_preds_2nd = rel_preds_top2[:,:,1]
        rel_preds = rel_preds_1st
        # (n x b)
        rel_targets = self.vocabs['rels'].placeholder
        # (n x b)
        rel_correct = tf.to_int32(tf.equal(rel_preds, rel_targets))*int_tokens_to_keep
        n_rels = tf.shape(select_rel_logits_targets)[2]
        # (n x b x r)
        #rel_preds_onehot = tf.one_hot(rel_preds, n_rels, on_value = True, off_value = False, dtype = tf.bool)
        rel_targets_onehot = tf.one_hot(rel_targets, n_rels, on_value = True, off_value = False, dtype = tf.bool)
        # (n x b)
        unequal_mask = tf.to_float(np.not_equal(rel_preds_1st, rel_targets))
        equal_mask = tf.subtract(1.0, unequal_mask)
        # (n x b x 2)
        rel_mask_top2 = tf.stack([unequal_mask, equal_mask])
        # (n x b)
        wrong_pred_scores = tf.reduce_sum(tf.multiply(rel_scores_top2, rel_mask_top2), 2)
        # (n x b)
        #rel_preds_scores = tf.reshape(tf.boolean_mask(select_rel_logits_preds, rel_preds_onehot), [self.batch_size, self.bucket_size])
        #rel_preds_scores = tf.reduce_max(select_rel_logits_preds, axis = 2)
        rel_targets_scores = tf.reshape(tf.boolean_mask(select_rel_logits_targets, rel_targets_onehot), [self.batch_size, self.bucket_size])
        # (n x b)
        rel_loss_scores = tf.subtract(wrong_pred_scores, rel_targets_scores)
        z = tf.zeros(tf.shape(rel_loss_scores))
        rel_loss_scores = tf.where(tf.greater(rel_loss_scores, -1.0), rel_loss_scores, z)
        # (n)
        #rel_losses = tf.reduce_sum(tf.subtract(rel_preds_scores, rel_targets_scores), 1)
        rel_losses = tf.reduce_sum(rel_loss_scores, 1)
        # ()
        rel_loss = tf.reduce_sum(rel_losses)
      else:
        # (n x b x b)
        one_hot = tf.one_hot(arc_preds if moving_params is not None else arc_targets, self.bucket_size)
        # (n x b x b) -> (n x b x b x 1)
        one_hot = tf.expand_dims(one_hot, axis=3)
        # (n x b x r x b) * (n x b x b x 1) -> (n x b x r x 1)
        select_rel_logits = tf.matmul(rel_logits, one_hot)
        # (n x b x r x 1) -> (n x b x r)
        select_rel_logits = tf.squeeze(select_rel_logits, axis=3)
        # (n x b)
        rel_preds = tf.to_int32(tf.argmax(select_rel_logits, axis=-1))
        # (n x b)
        rel_targets = self.vocabs['rels'].placeholder
        # (n x b)
        rel_correct = tf.to_int32(tf.equal(rel_preds, rel_targets))*int_tokens_to_keep
        # ()
        rel_loss = tf.losses.sparse_softmax_cross_entropy(rel_targets, select_rel_logits, self.tokens_to_keep)
    
    n_arc_correct = tf.reduce_sum(arc_correct)
    n_rel_correct = tf.reduce_sum(rel_correct)
    correct = arc_correct * rel_correct
    n_correct = tf.reduce_sum(correct)
    n_seqs_correct = tf.reduce_sum(tf.to_int32(tf.equal(tf.reduce_sum(correct, axis=1), self.sequence_lengths-1)))

    if arc_placeholder is not None:
      # (n)
      losses = tf.add(arc_losses, rel_losses)
      #loss = tf.reduce_sum(tf.maximum(losses, 0))
      loss = tf.reduce_sum(losses)
      tf.losses.add_loss(loss)
    else:
      loss = arc_loss + rel_loss
    if self.l2_norm:
      #print ('\n'.join([w.name for w in tf.get_collection('Weights')]))
      l2_losses = [tf.nn.l2_loss(w) for w in tf.get_collection('Weights')]
      #print (len(l2_losses))
      tf.losses.add_loss(self.l2_rate * tf.add_n(l2_losses))

    outputs = {
      'arc_logits': arc_logits,
      'arc_probs': arc_probs,
      'arc_preds': arc_preds,
      'arc_targets': arc_targets,
      'arc_correct': arc_correct,
      'arc_loss': arc_loss,
      'n_arc_correct': n_arc_correct,
      
      'rel_logits': rel_logits,
      'rel_probs': rel_probs,
      'rel_preds': rel_preds,
      'rel_targets': rel_targets,
      'rel_correct': rel_correct,
      'rel_loss': rel_loss,
      'n_rel_correct': n_rel_correct,
      
      'n_tokens': self.n_tokens,
      'n_seqs': self.batch_size,
      'tokens_to_keep': self.tokens_to_keep,
      'n_correct': n_correct,
      'n_seqs_correct': n_seqs_correct,
      'loss': loss
    }
    """
    if arc_placeholder is not None:
      outputs['arc_pred_scores'] = arc_preds_scores
      outputs['arc_target_scores'] = arc_targets_scores
      outputs['rel_pred_scores'] = arc_preds
      outputs['rel_target_scores'] = arc_preds
      outputs['arc_losses'] = arc_logits
    """
    return outputs

  #=============================================================
  # dep_mlp/head_mlp : (n x b x d)
  def score(self, dep_mlp, head_mlp, output=1):

    # (n x b x d) => (n x b x 1 x d)
    dep_mlp_ = tf.expand_dims(dep_mlp, axis=2)
    # (n x b x b x d)
    #deps = tf.concat([dep_mlp_] * self.bucket_size, 2)
    deps = tf.tile(dep_mlp_, [1,1,self.bucket_size,1])
    # (n x b x d) => (n x 1 x b x d)
    head_mlp_ = tf.expand_dims(head_mlp, axis=1)
    # (n x b x b x d)
    #heads = tf.concat([head_mlp_] * self.bucket_size, 1)
    heads = tf.tile(head_mlp_, [1,self.bucket_size,1,1])
    # (n x b x b x 2d)
    reps = tf.concat([deps, heads], 3)
    # (n x b x b x output)
    logits = self.linear(reps, output)

    return logits

  #=============================================================
  def parse4graph(self, vocabs, ts_lstm=None, arc_placeholder=None, stacked_cnn=None, moving_params=None):
    
    top_rep = super(MLPParser, self).__call__(vocabs, ts_lstm, stacked_cnn, moving_params=moving_params)
    # (n x b)
    int_tokens_to_keep = tf.to_int32(self.tokens_to_keep)
    
    with tf.variable_scope('MLP'):
      dep_mlp, head_mlp = self.MLP(top_rep, self.arc_mlp_size + self.rel_mlp_size,
                                   n_splits=2)
      arc_dep_mlp, rel_dep_mlp = tf.split(dep_mlp, [self.arc_mlp_size, self.rel_mlp_size], axis=2)
      arc_head_mlp, rel_head_mlp = tf.split(head_mlp, [self.arc_mlp_size, self.rel_mlp_size], axis=2)
    
    with tf.variable_scope('Arc'):
      # (n x b x d) * (d x 1 x d) * (n x b x d).T -> (n x b x b)
      #arc_logits = self.bilinear(arc_dep_mlp, arc_head_mlp, 1, add_bias2=False)
      arc_logits = self.score(arc_dep_mlp, arc_head_mlp)

      # (n x b x b) of 0/1
      arc_targets = self.vocabs['heads'].placeholder
      n_arc_targets = tf.reduce_sum(tf.reduce_sum(arc_targets, axis=2) * int_tokens_to_keep)

      #arc_preds_scores = tf.reduce_sum(arc_logits * tf.to_float(arc_preds), axis=2) * self.tokens_to_keep
      arc_target_scores_ = tf.reduce_sum(arc_logits * tf.to_float(arc_targets), axis=2) * self.tokens_to_keep
      # (n)
      arc_target_scores = tf.reduce_sum(arc_target_scores_, axis=1)
      # ()
      #arc_loss = tf.reduce_sum(arc_losses)
      # (n x b)
      #masked_margin = tf.to_float(tf.not_equal(masked_arc_preds, masked_arc_targets))
      # (n)
      #margin = tf.reduce_sum(masked_margin, axis = 1)
      # (n)
      #arc_losses += margin
      #arc_losses = np.add(arc_losses, 1)
     
    with tf.variable_scope('Rel'):
      # (n x b x d) * (d x r x d) * (n x b x d).T -> (n x b x r x b)
      #rel_logits = self.bilinear(rel_dep_mlp, rel_head_mlp, len(self.vocabs['rels']))
      # (n x b x b x r)
      rel_logits = self.score(rel_dep_mlp, rel_head_mlp, output=len(self.vocabs['rels']))
      # (n x b x r x b) => (n x b x b x r)
      #rel_logits = tf.transpose(rel_logits, [0, 1, 3, 2])
      # (n x b x b x r)
      #rel_probs = tf.nn.softmax(rel_logits, dim=3)

      # Sum up scores of the arc and best label on it
      # (n x b x b)
      sum_logits = arc_logits + tf.reduce_max(rel_logits, axis=3)
      # (n x b x b)
      #arc_preds = arc_placeholder
      arc_preds = tf.to_int32(tf.greater(sum_logits, 0))
      n_arc_preds = tf.reduce_sum(tf.reduce_sum(arc_preds, axis=2) * int_tokens_to_keep)
      # (n x b x b)
      arc_correct_ = tf.to_int32(tf.equal(arc_preds, arc_targets)) * arc_targets
      # (n x b)
      arc_correct = tf.reduce_sum(arc_correct_, axis=2) * int_tokens_to_keep

      # Max label index of each possible arc (n x b x b)
      rel_idxs = tf.to_int32(tf.argmax(rel_logits, axis=3))
      # (n x b x b)
      rel_preds = arc_preds * rel_idxs
      # (n x b x b)
      rel_targets = self.vocabs['rels'].placeholder
      # (n x b x b)
      rel_correct_ = tf.to_int32(tf.equal(rel_preds, rel_targets)) * arc_targets
      # (n x b)
      rel_correct = tf.reduce_sum(rel_correct_, axis=2) * int_tokens_to_keep

      n_rels = tf.shape(rel_logits)[3]
      # (n x b x b x r)
      onehot = tf.one_hot(rel_targets, n_rels)
      # (n x b)
      rel_target_scores_ = tf.reduce_sum(tf.reduce_sum(rel_logits * onehot, axis=3), axis=2) * self.tokens_to_keep
      # (n)
      rel_target_scores = tf.reduce_sum(rel_target_scores_, axis=1)
      # (n x b)
      pred_scores_ = tf.reduce_sum(sum_logits * tf.to_float(arc_preds), axis=2) * self.tokens_to_keep
      # (n)
      pred_scores = tf.reduce_sum(pred_scores_, axis=1)
      target_scores_ = arc_target_scores_ + rel_target_scores_
      target_scores = tf.reduce_sum(target_scores_, axis=1)
    
    n_arc_correct = tf.reduce_sum(arc_correct)
    n_rel_correct = tf.reduce_sum(rel_correct)
    # (n x b)
    correct = tf.reduce_sum(arc_correct_ * rel_correct_, axis=2) * int_tokens_to_keep
    n_correct = tf.reduce_sum(correct)
    #n_seqs_correct = tf.reduce_sum(tf.to_int32(tf.equal(tf.reduce_sum(correct, axis=1), self.sequence_lengths-1)))

    # (n)
    losses = pred_scores - target_scores + 1
    loss = tf.reduce_sum(tf.maximum(losses, 0))
    #loss = tf.reduce_sum(losses)
    tf.losses.add_loss(loss)

    if self.l2_norm:
      #print ('\n'.join([w.name for w in tf.get_collection('Weights')]))
      l2_losses = [tf.nn.l2_loss(w) for w in tf.get_collection('Weights')]
      #print (len(l2_losses))
      tf.losses.add_loss(self.l2_rate * tf.add_n(l2_losses))

    outputs = {
      'pred_scores_': pred_scores_,
      'target_scores_': target_scores_,
      'sum_logits': sum_logits,
      'arc_logits': arc_logits,
      'arc_preds': arc_preds,
      'arc_targets': arc_targets,
      'arc_correct': arc_correct,
      #'arc_loss': arc_loss,
      'n_arc_correct': n_arc_correct,
      'n_arc_targets': n_arc_targets,
      'n_arc_preds': n_arc_preds,
      
      'rel_logits': rel_logits,
      'rel_preds': rel_preds,
      'rel_targets': rel_targets,
      'rel_correct': rel_correct,
      #'rel_loss': rel_loss,
      'n_rel_correct': n_rel_correct,
      
      'n_tokens': self.n_tokens,
      'n_seqs': self.batch_size,
      'tokens_to_keep': self.tokens_to_keep,
      'n_correct': n_correct,
      #'n_seqs_correct': n_seqs_correct,
      'loss': loss
    }
    """
    if arc_placeholder is not None:
      outputs['arc_pred_scores'] = arc_preds_scores
      outputs['arc_target_scores'] = arc_targets_scores
      outputs['rel_pred_scores'] = arc_preds
      outputs['rel_target_scores'] = arc_preds
      outputs['arc_losses'] = arc_logits
    """
    return outputs