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
from collections import Counter

import numpy as np
import tensorflow as tf

from parser import Configurable, Multibucket
from parser.vocabs.base_vocab import BaseVocab
from parser.misc.bucketer import Bucketer
from parser.misc.mst import nonprojective

__all__ = ['Trainset', 'Parseset']

#***************************************************************
class Dataset(Configurable):
  """"""
  
  #=============================================================
  def __init__(self, vocabs, load_file, use_hinge_loss, ts_lstm, stacked_cnn, *args, **kwargs):
    """"""
    # nlp_model = Parser
    nlp_model = kwargs.pop('nlp_model', None)
    #print ("---dataset.py---\n",nlp_model)
    super(Dataset, self).__init__(*args, **kwargs)
    
    self._vocabs = vocabs
    self._multibuckets = [Multibucket.from_configurable(vocab, name='%s-%s'%(self.name, vocab.name)) for vocab in self.vocabs]
    
    if nlp_model is not None:
      self._nlp_model = nlp_model.from_configurable(self, name=self.name)
    else:
      self._nlp_model = None

    self._ts_lstm = ts_lstm

    self._stacked_cnn = stacked_cnn

    self._use_hinge_loss = use_hinge_loss
    if self.use_hinge_loss:
      self._arc_placeholder = tf.placeholder(tf.int32, shape=[None, None], name=self.name)
    else:
      self._arc_placeholder = None

    self._load_file = load_file
    if not self.load_file:
      print ("### {} not load file! ###".format(self.name))
      return

    #print ("---dataset.py---after\n",nlp_model)
    if self.data_form == 'tree':
      self.bucket4tree()
    elif self.data_form == 'graph':
      self.bucket4graph()

    return

  #=============================================================
  def __call__(self, moving_params=None):
    """"""
    #print ("dataset.py(__call__):/n",self.vocabs)
    # Here nlp_model = Parser
    return self._nlp_model(self.vocabs, ts_lstm=self.ts_lstm, arc_placeholder=self.arc_placeholder,
                            stacked_cnn=self.stacked_cnn ,moving_params=moving_params)
  
  #=============================================================
  def bucket4tree(self):
    with Bucketer.from_configurable(self, self.n_buckets, name='bucketer-%s'%self.name) as bucketer:
      splits = bucketer.compute_splits(len(sent) for sent in self.iterfiles4tree())
      for i in xrange(len(splits)):
        splits[i] += 1
    for multibucket, vocab in self.iteritems():
      multibucket.open(splits, depth=vocab.depth)
    for sid, sent in enumerate(self.iterfiles4tree()):
      for i in xrange(len(self)):
        multibucket, vocab = self.get_item(i)
        tokens = [line[vocab.conll_idx] for line in sent]
        if vocab.name == 'words':
          idxs = [vocab.ROOT] + [vocab.index(token, self.name+"-"+str(sid)+"-"+str(wid)) for wid, token in enumerate(tokens)]
        else:
          idxs = [vocab.ROOT] + [vocab.index(token) for token in tokens]
        multibucket.add(idxs, tokens)
    for multibucket in self:
      multibucket.close()
    self._multibucket = Multibucket.from_dataset(self, data_form=self.data_form)
    return

  #=============================================================
  def bucket4graph(self):
    with Bucketer.from_configurable(self, self.n_buckets, name='bucketer-%s'%self.name) as bucketer:
      splits = bucketer.compute_splits(len(sent) for sent in self.iterfiles4graph())
      for i in xrange(len(splits)):
        splits[i] += 1
    for multibucket, vocab in self.iteritems():
        multibucket.open(splits, depth=vocab.depth, vocab_name=vocab.name)
    for sid, sent in enumerate(self.iterfiles4graph()):
      #print (sent)
      for i in xrange(len(self)):
        multibucket, vocab = self.get_item(i)
        tokens = [line[vocab.conll_idx] for line in sent]
        if vocab.name == 'words':
          idxs = [vocab.ROOT] + [vocab.index(token, self.name+"-"+str(sid)+"-"+str(wid)) for wid, token in enumerate(tokens)]
        elif vocab.name == 'heads':
          idxs = [[vocab.ROOT]] + [vocab.index(token) for token in tokens]
        elif vocab.name == 'rels':
          idxs = [[(0, vocab.ROOT)]] + [vocab.index(token) for token in tokens]
        else:
          idxs = [vocab.ROOT] + [vocab.index(token) for token in tokens]
        #print (vocab.name, idxs)
        multibucket.add(idxs, tokens)
    for multibucket in self:
      multibucket.close()
    self._multibucket = Multibucket.from_dataset(self, data_form=self.data_form)
    return

  #=============================================================
  def feed_arc(self, arc_probs, tokens_to_keep):
    max_len = arc_probs.shape[1]
    arc_preds = []
    for arc_prob, weights in zip(arc_probs, tokens_to_keep):
    #for arc_logit, arc_prob, weights in zip(arc_logits, arc_probs, tokens_to_keep):
      sequence_length = int(np.sum(weights))+1
      arc_prob = arc_prob[:sequence_length][:,:sequence_length]
      arc_pred = nonprojective(arc_prob)
      arc_pred = np.pad(arc_pred, (0, max_len - sequence_length),'constant')
      arc_preds.append(arc_pred)
    arc_preds = np.stack(arc_preds)
    feed_dict = {self.arc_placeholder:arc_preds}
    return feed_dict

  #=============================================================
  def iterfiles4tree(self):
    """"""
    if self.name == "parseset":
      self._merge_lines = []
      for data_file in self.data_files:
        with codecs.open(data_file, encoding='utf-8', errors='ignore') as f:
          buff = []
          merge_line = {}
          for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
              if re.match('[0-9]+-[0-9]+', line):
                merge_line[int(line.split('\t')[0].split('-')[0])] = line
              if not re.match('[0-9]+[-.][0-9]+', line):
                buff.append(line.split('\t'))
            elif buff:
              self.merge_lines.append(merge_line)
              merge_line = {}
              yield buff
              buff = []
          self.merge_lines.append(merge_line)
          yield buff
    else:
      for data_file in self.data_files:
        with codecs.open(data_file, encoding='utf-8', errors='ignore') as f:
          buff = []
          for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
              if not re.match('[0-9]+[-.][0-9]+', line):
                buff.append(line.split('\t'))
            elif buff:
              yield buff
              buff = []
          yield buff
  
  #=============================================================
  # There is no [0-9]+[-.][0-9]+ in graphs
  def iterfiles4graph(self):
    """"""
    self._merge_lines = []
    for data_file in self.data_files:
      with codecs.open(data_file, encoding='utf-8', errors='ignore') as f:
        buff = []
        id = 0
        for line in f:
          line = line.strip()
          if line and not line.startswith('#'):
            items = line.split('\t')
            if not re.match('[0-9]+[-.][0-9]+', line):
              if int(items[0]) == id:
                buff[-1][6].append(items[6])
                buff[-1][7].append((items[6], items[7]))
              else:
                items[7] = [(items[6], items[7])]
                items[6] = [items[6]]
                buff.append(items)
                id = int(items[0])
          elif buff:
            id = 0
            self.merge_lines.append({})
            yield buff
            buff = []
        self.merge_lines.append({})
        yield buff

  #=============================================================
  def iterbatches(self, shuffle=True, return_check=False):
    """"""
    
    batch_size = self.batch_size
    batch_by = self.batch_by 
    batches = []
    for bkt_idx, bucket in enumerate(self.multibucket):
      if batch_size == 0:
        n_splits = 1
      else:
        if batch_by == 'tokens':
          n_tokens = bucket.indices.shape[0] * bucket.indices.shape[1]
          n_splits = max(n_tokens // batch_size, 1)
        elif batch_by == 'seqs':
          n_seqs = bucket.indices.shape[0]
          n_splits = max(n_seqs // batch_size, 1)
      """
      if shuffle:
        range_func = np.random.permutation
      else:
        range_func = np.arange
      splits = np.array_split(range_func(bucket.indices.shape[0])[1:], n_splits)
      """
      splits = np.array_split(np.arange(bucket.indices.shape[0])[1:], n_splits)
      if shuffle:
        np.random.shuffle(splits)
      for split in splits:
        batches.append( (bkt_idx, split) )
    if shuffle:
      np.random.shuffle(batches)

    for bkt_idx, batch in batches:
      feed_dict = {}
      tokens = []
      for multibucket, vocab in self.iteritems():
        bucket = multibucket[bkt_idx]
        indices = bucket.indices[batch]
        vocab.set_feed_dict(indices, feed_dict)
        if vocab.name == 'positions': continue
        if return_check:
          if len(indices.shape) == 2:
            tokens.append(vocab[indices])
          elif len(indices.shape) == 3:
            #tokens.extend([subvocab[indices[:,:,i]] for i, subvocab in enumerate(vocab)])
            # TODO This is super hacky
            #if hasattr(subvocab, 'idx2tok'):
              #tokens[-1] = [[subvocab.idx2tok.get(idx, subvocab[subvocab.PAD]) for idx in idxs] for idxs in indices[:,:,-1]]

            for i, subvocab in enumerate(vocab):
              # subtoken vocab
              if hasattr(subvocab, 'idx2tok'):
                tokens.append([[subvocab.idx2tok.get(idx, subvocab[subvocab.PAD]) for idx in idxs] for idxs in indices[:,:,i]])
              else:
                tokens.append(subvocab[indices[:,:,i]])
                
        elif not shuffle:
          tokens.append(bucket.get_tokens(batch))
      if not shuffle or return_check:
        yield feed_dict, zip(*tokens)
      else:
        yield feed_dict
  
  #=============================================================
  def iteritems(self):
    for i in xrange(len(self)):
      yield (self[i], self._vocabs[i])

  def get_item(self, i):
    return (self[i], self._vocabs[i])
  
  #=============================================================
  def update_history(self, history, accumulators):
    return self._nlp_model.update_history(history, accumulators)
  
  def print_accuracy(self, accumulators, time):
    return self._nlp_model.print_accuracy(accumulators, time, prefix=self.PREFIX.title())
  
  def write_probs(self, sents, output_file, probs):
    return self._nlp_model.write_probs(sents, output_file, probs, self.multibucket.inv_idxs(), self.merge_lines)

  def write_probs_ensemble(self, sents, output_file, multi_probs, sum_type, sum_weights=None):
    return self._nlp_model.write_probs_ensemble(sents, output_file, multi_probs, self.multibucket.inv_idxs(), sum_type, self.merge_lines, sum_weights)

  def check(self, preds, sents, fileobj):
    return self._nlp_model.check(preds, sents, fileobj)
  
  def plot(self, history):
    return self._nlp_model.plot(history)
  
  #=============================================================
  @property
  def data_files(self):
    return getattr(self, '{0}_files'.format(self.PREFIX.lower()))
  @property
  def multibucket(self):
    return self._multibucket
  @property
  def vocabs(self):
    return self._vocabs
  @property
  def train_keys(self):
    return self._nlp_model.train_keys
  @property
  def valid_keys(self):
    return self._nlp_model.valid_keys
  @property
  def parse_keys(self):
    return self._nlp_model.parse_keys
  @property
  def ensemble_keys(self):
    return self._nlp_model.ensemble_keys
  # for merge lines like 1-2 it's _ _ ...
  @property
  def merge_lines(self):
    return self._merge_lines
  @property
  def load_file(self):
    return self._load_file
  @property
  def ts_lstm(self):
    return self._ts_lstm
  @property
  def stacked_cnn(self):
    return self._stacked_cnn
  @property
  def use_hinge_loss(self):
    return self._use_hinge_loss
  @property
  def arc_placeholder(self):
    return self._arc_placeholder
  
  #=============================================================
  def __len__(self):
    return len(self._multibuckets)
  def __iter__(self):
    return (multibucket for multibucket in self._multibuckets)
  def __getitem__(self, key):
    return self._multibuckets[key]

#***************************************************************
class Trainset(Dataset):
  PREFIX = 'train'
class Parseset(Dataset):
  PREFIX = 'parse'

#***************************************************************
if __name__ == '__main__':
  """"""
  
  from parser.vocabs import *
  from parser.dataset import Trainset
  
  configurable = Configurable()
  dep_vocab = DepVocab.from_configurable(configurable)
  word_vocab = WordVocab.from_configurable(configurable)
  lemma_vocab = LemmaVocab.from_configurable(configurable)
  #pretrained_vocab = PretrainedVocab.from_vocab(word_vocab)
  char_vocab = NgramMultivocab.from_vocab(word_vocab)
  word_multivocab = Multivocab.from_configurable(configurable, [word_vocab, char_vocab], name='words')
  tag_vocab = TagVocab.from_configurable(configurable)
  xtag_vocab = XTagVocab.from_configurable(configurable)
  head_vocab = HeadVocab.from_configurable(configurable)
  rel_vocab = RelVocab.from_configurable(configurable)
  trainset = Trainset.from_configurable(configurable, [dep_vocab, word_multivocab, lemma_vocab, tag_vocab, xtag_vocab, head_vocab, rel_vocab],
                            True, False, None, None)
  trainset()
  print('Dataset passes')
  
