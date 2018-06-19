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
import time
import codecs
import cPickle as pkl
from collections import defaultdict
import os.path as op

import numpy as np
import tensorflow as tf

from parser import Configurable
from parser.vocabs import *
from parser.dataset import *
from parser.misc.colors import ctext
from parser.neural.optimizers import RadamOptimizer
from parser.neural.models import TS_LSTM

#***************************************************************
class Network(Configurable):
  """"""
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
    is_eval = ('is_evaluation' in kwargs and kwargs['is_evaluation'])
    files = kwargs.pop('files') if is_eval else None
    elmo_file = kwargs["elmo_file"] if 'elmo_file' in kwargs else None
    if elmo_file:
      del kwargs['elmo_file']

    super(Network, self).__init__(*args, **kwargs)
    # hacky!
    #hacky_train_files = op.join(self.save_dir, op.basename(self.get("train_files")))
    #self._config.set('Configurable', 'train_files', hacky_train_files)
    
    # TODO make this more flexible, maybe specify it in config?
    temp_nlp_model = self.nlp_model.from_configurable(self)
    if temp_nlp_model.input_vocabs == ['tags']:
      #print ("model")
      word_vocab = WordVocab.from_configurable(self)
      word_multivocab = Multivocab.from_configurable(self, [word_vocab], name=word_vocab.name)
      tag_vocab = TagVocab.from_configurable(self, initialize_zero=False)
    else:
      print ("### Loading word vocab ###")
      word_vocab = WordVocab.from_configurable(self)
      word_vocabs = [word_vocab]
      #print ("word_vocab len: ", word_vocab.counts)
      if (self.use_pretrained):
        print ("### Loading pretrained vocab ###")
        pretrained_vocab = PretrainedVocab.from_vocab(word_vocab)
        word_vocabs.append(pretrained_vocab)
      
      if (self.use_elmo):
        print ("### Loading ELMo vocab ###")
        if (is_eval):
          if elmo_file:
            elmo_vocab = ElmoVocab.from_vocab(word_vocab, elmo_file, files)
          else:
            print ("### Error: ELMo file for test set not provided! ###")
            exit()
        else:
          elmo_vocab = ElmoVocab.from_vocab(word_vocab, None, None)
        word_vocabs.append(elmo_vocab)
        self._elmo_vocab = elmo_vocab
        #word_multivocab = Multivocab.from_configurable(self, [word_vocab, pretrained_vocab, elmo_vocab, subtoken_vocab], name=word_vocab.name)
      #else:
        #word_multivocab = Multivocab.from_configurable(self, [word_vocab, pretrained_vocab, subtoken_vocab], name=word_vocab.name)

      if (self.use_char):
        print ("### Loading subtoken vocab ###")
        #self.subtoken_vocab here = CharVocab
        subtoken_vocab = self.subtoken_vocab.from_vocab(word_vocab)
        word_vocabs.append(subtoken_vocab)

      word_multivocab = Multivocab.from_configurable(self, word_vocabs, name=word_vocab.name)
      #word_multivocab = Multivocab.from_configurable(self, [word_vocab, pretrained_vocab], name=word_vocab.name)
      tag_vocab = TagVocab.from_configurable(self)
    print ("### Loading dep vocab ###")
    dep_vocab = DepVocab.from_configurable(self)
    lemma_vocab = LemmaVocab.from_configurable(self)
    xtag_vocab = XTagVocab.from_configurable(self)
    head_vocab = HeadVocab.from_configurable(self)
    rel_vocab = RelVocab.from_configurable(self)
    self._vocabs = [dep_vocab, word_multivocab, lemma_vocab, tag_vocab, xtag_vocab, head_vocab, rel_vocab]
    self._global_step = tf.Variable(0., trainable=False, name='global_step')
    self._global_epoch = tf.Variable(0., trainable=False, name='global_epoch')
    self._optimizer = RadamOptimizer.from_configurable(self, global_step=self.global_step)
    return
  
  #=============================================================
  def add_file_vocabs(self, conll_files):
    """"""
    
    # TODO don't depend on hasattr
    for vocab in self.vocabs:
      if hasattr(vocab, 'add_files'):
        vocab.add_files(conll_files)
    for vocab in self.vocabs:
      if hasattr(vocab, 'index_tokens'):
        vocab.index_tokens()
    return
  
  #=============================================================
  def train(self, load=False):
    """"""
    
    # prep the configurables
    self.add_file_vocabs(self.parse_files)
    trainset = Trainset.from_configurable(self, self.vocabs, True, nlp_model=self.nlp_model)
    with tf.variable_scope(self.name.title()):
      train_tensors = trainset()
    train = self.optimizer(tf.losses.get_total_loss())
    train_outputs = [train_tensors[train_key] for train_key in trainset.train_keys]
    saver = tf.train.Saver(self.save_vars, max_to_keep=1)
    validset = Parseset.from_configurable(self, self.vocabs, True, nlp_model=self.nlp_model)
    with tf.variable_scope(self.name.title(), reuse=True):
      valid_tensors = validset(moving_params=self.optimizer)
    valid_outputs = [valid_tensors[train_key] for train_key in validset.train_keys]
    valid_outputs2 = [valid_tensors[valid_key] for valid_key in validset.valid_keys]
    current_acc = 0
    best_acc = 0
    n_iters_since_improvement = 0
    n_iters_in_epoch = 0
    
    # calling these properties is inefficient so we save them in separate variables
    min_train_iters = self.min_train_iters
    max_train_iters = self.max_train_iters
    min_save_iters = self.min_save_iters
    validate_every = self.validate_every
    save_every = self.save_every
    verbose = self.verbose
    quit_after_n_iters_without_improvement = self.quit_after_n_iters_without_improvement
    cpu_count = self.cpu_count
    
    # load or prep the history
    if load:
      self.history = pkl.load(open(os.path.join(self.save_dir, 'history.pkl')))
    else:
      self.history = {'train': defaultdict(list), 'valid': defaultdict(list)}
    
    # start up the session
    config_proto = tf.ConfigProto(device_count={"CPU": cpu_count})
    if self.per_process_gpu_memory_fraction == -1:
      config_proto.gpu_options.allow_growth = True
    else:
      config_proto.gpu_options.per_process_gpu_memory_fraction = self.per_process_gpu_memory_fraction
    with tf.Session(config=config_proto) as sess:
      if self.use_elmo:
        sess.run(tf.global_variables_initializer(), feed_dict={self.elmo_vocab.embed_placeholder:self.elmo_vocab.elmo_embeddings})
      else:
        sess.run(tf.global_variables_initializer())
      if load:
        saver.restore(sess, tf.train.latest_checkpoint(self.save_dir))
      total_train_iters = sess.run(self.global_step)
      train_accumulators = np.zeros(len(train_outputs))
      train_time = 0
      # training loop
      while total_train_iters < max_train_iters:
        for feed_dict in trainset.iterbatches():
          start_time = time.time()
          batch_values = sess.run(train_outputs + [train], feed_dict=feed_dict)[:-1]
          batch_time = time.time() - start_time
          # update accumulators
          total_train_iters += 1
          n_iters_since_improvement += 1
          train_accumulators += batch_values
          train_time += batch_time
          # possibly validate
          if total_train_iters == 1 or (total_train_iters % validate_every == 0):
            valid_accumulators = np.zeros(len(train_outputs))
            valid_time = 0
            with codecs.open(os.path.join(self.save_dir, 'sanity_check'), 'w', encoding='utf-8', errors='ignore') as f:
              for feed_dict, sents in validset.iterbatches(return_check=True):
                start_time = time.time()
                batch_values = sess.run(valid_outputs+valid_outputs2, feed_dict=feed_dict)
                batch_time = time.time() - start_time
                # update accumulators
                valid_accumulators += batch_values[:len(valid_outputs)]
                valid_preds = batch_values[len(valid_outputs):]
                valid_time += batch_time
                validset.check(valid_preds, sents, f)
            # update history
            trainset.update_history(self.history['train'], train_accumulators)
            current_acc = validset.update_history(self.history['valid'], valid_accumulators)
            # print
            if verbose:
              print('{0:6d}'.format(int(total_train_iters))+')') 
              trainset.print_accuracy(train_accumulators, train_time)
              validset.print_accuracy(valid_accumulators, valid_time)
            train_accumulators = np.zeros(len(train_outputs))
            train_time = 0
            if current_acc > best_acc and total_train_iters > min_save_iters:
              if verbose:
                print('Saving model...')
              best_acc = current_acc
              n_iters_since_improvement = 0
              saver.save(sess, os.path.join(self.save_dir, self.name.lower()),
                         #global_step=self.global_epoch,
                         write_meta_graph=False)
              with open(os.path.join(self.save_dir, 'history.pkl'), 'w') as f:
                pkl.dump(dict(self.history), f)
            elif n_iters_since_improvement >= quit_after_n_iters_without_improvement and total_train_iters > min_train_iters:
              break
        else:
          # We've completed one epoch
          if total_train_iters <= min_train_iters:
            saver.save(sess, os.path.join(self.save_dir, self.name.lower()),
                       #global_step=self.global_epoch,
                       write_meta_graph=False)
            with open(os.path.join(self.save_dir, 'history.pkl'), 'w') as f:
              pkl.dump(dict(self.history), f)
          continue
        break
    print ("### Finish Training! ###")
    return
  
  #=============================================================
  def parse(self, input_files, output_dir=None, output_file=None):
    """"""
    
    if not isinstance(input_files, (tuple, list)):
      input_files = [input_files]
    if len(input_files) > 1 and output_file is not None:
      raise ValueError('Cannot provide a value for --output_file when parsing multiple files')
    self.add_file_vocabs(input_files)
    
    # load the model and prep the parse set
    trainset = Trainset.from_configurable(self, self.vocabs, False, nlp_model=self.nlp_model)
    with tf.variable_scope(self.name.title()):
      train_tensors = trainset()
    train_outputs = [train_tensors[train_key] for train_key in trainset.train_keys]

    saver = tf.train.Saver(self.save_vars, max_to_keep=1)
    config_proto = tf.ConfigProto(device_count={"CPU": self.cpu_count})
    if self.per_process_gpu_memory_fraction == -1:
      config_proto.gpu_options.allow_growth = True
    else:
      config_proto.gpu_options.per_process_gpu_memory_fraction = self.per_process_gpu_memory_fraction
    with tf.Session(config=config_proto) as sess:
      for var in self.non_save_vars:
        if var.name.startswith('Elmo'):
          sess.run(var.initializer, feed_dict={self.elmo_vocab.embed_placeholder:self.elmo_vocab.elmo_embeddings})
        else:
          sess.run(var.initializer)
      saver.restore(sess, tf.train.latest_checkpoint(self.save_dir))
      
      # Iterate through files and batches
      for input_file in input_files:
        #parseset = Parseset.from_configurable(trainset, self.vocabs, parse_files=input_file, nlp_model=self.nlp_model)
        parseset = Parseset.from_configurable(trainset, self.vocabs, True, parse_files=input_file, nlp_model=self.nlp_model)
        with tf.variable_scope(self.name.title(), reuse=True):
          parse_tensors = parseset(moving_params=self.optimizer)
        parse_outputs = [parse_tensors[parse_key] for parse_key in parseset.parse_keys]

        input_dir, input_file = os.path.split(input_file)
        if output_dir is None and output_file is None:
          output_dir = self.save_dir
        if output_dir == input_dir and output_file is None:
          output_path = os.path.join(input_dir, 'parsed-'+input_file)
        elif output_file is None:
          output_path = os.path.join(output_dir, input_file)
        else:
          output_path = output_file
        
        start_time = time.time()
        probs = []
        sents = []
        for feed_dict, tokens in parseset.iterbatches(shuffle=False):
          probs.append(sess.run(parse_outputs, feed_dict=feed_dict))
          sents.append(tokens)
        parseset.write_probs(sents, output_path, probs)
    if self.verbose:
      print('Parsing {0} file(s) took {1} seconds'.format(len(input_files), time.time()-start_time))
    return
  
  #=============================================================
  def ensemble(self, input_files, other_save_dirs, sum_type="prob", sum_weights=None, output_dir=None, output_file=None):
    """"""
    
    if not isinstance(input_files, (tuple, list)):
      input_files = [input_files]
    if len(input_files) > 1 and output_file is not None:
      raise ValueError('Cannot provide a value for --output_file when parsing multiple files')
    self.add_file_vocabs(input_files)
    
    # load the model and prep the parse set
    trainset = Trainset.from_configurable(self, self.vocabs, False, nlp_model=self.nlp_model)
    with tf.variable_scope(self.name.title()):
      train_tensors = trainset()
    train_outputs = [train_tensors[train_key] for train_key in trainset.train_keys]

    saver = tf.train.Saver(self.save_vars, max_to_keep=1)
    config_proto = tf.ConfigProto(device_count={"CPU": self.cpu_count})
    if self.per_process_gpu_memory_fraction == -1:
      config_proto.gpu_options.allow_growth = True
    else:
      config_proto.gpu_options.per_process_gpu_memory_fraction = self.per_process_gpu_memory_fraction
    with tf.Session(config=config_proto) as sess:
      for var in self.non_save_vars:
        if var.name.startswith('Elmo'):
          sess.run(var.initializer, feed_dict={self.elmo_vocab.embed_placeholder:self.elmo_vocab.elmo_embeddings})
        else:
          sess.run(var.initializer)
 
      # Iterate through files and batches
      for input_file in input_files:
        parseset = Parseset.from_configurable(trainset, self.vocabs, True, parse_files=input_file, nlp_model=self.nlp_model)
        with tf.variable_scope(self.name.title(), reuse=True):
          parse_tensors = parseset(moving_params=self.optimizer)
        ensemble_outputs = [parse_tensors[ensemble_key] for ensemble_key in parseset.ensemble_keys]

        input_dir, input_file = os.path.split(input_file)
        if output_dir is None and output_file is None:
          output_dir = self.save_dir
        if output_dir == input_dir and output_file is None:
          output_path = os.path.join(input_dir, 'parsed-'+input_file)
        elif output_file is None:
          output_path = os.path.join(output_dir, input_file)
        else:
          output_path = output_file
        
        multi_probs = []
        start_time = time.time()
        probs = []
        sents = []
        saver.restore(sess, tf.train.latest_checkpoint(self.save_dir))

        for feed_dict, tokens in parseset.iterbatches(shuffle=False):
          probs.append(sess.run(ensemble_outputs, feed_dict=feed_dict))
          sents.append(tokens)
        multi_probs.append(probs)

        for other_dir in other_save_dirs:
          probs = []
          sents = []
          print("Loading from model ",other_dir)
          saver.restore(sess, tf.train.latest_checkpoint(other_dir))
          for feed_dict, tokens in parseset.iterbatches(shuffle=False):
            probs.append(sess.run(ensemble_outputs, feed_dict=feed_dict))
            sents.append(tokens)
          multi_probs.append(probs)

        #parseset.write_probs(sents, output_path, probs)
        parseset.write_probs_ensemble(sents, output_path, multi_probs, sum_type, sum_weights=sum_weights)
    if self.verbose:
      print('Parsing {0} file(s) took {1} seconds'.format(len(input_files), time.time()-start_time))
    return

  #=============================================================
  @property
  def vocabs(self):
    return self._vocabs
  @property
  def datasets(self):
    return self._datasets
  @property
  def optimizer(self):
    return self._optimizer
  @property
  def save_vars(self):
    return filter(lambda x: u'Pretrained/Embeddings:0' != x.name and u'Elmo/Embeddings:0' != x.name, tf.global_variables())
  @property
  def non_save_vars(self):
    return filter(lambda x: u'Pretrained/Embeddings:0' == x.name or u'Elmo/Embeddings:0' == x.name, tf.global_variables())
  @property
  def global_step(self):
    return self._global_step
  @property
  def global_epoch(self):
    return self._global_epoch
  @property
  def elmo_vocab(self):
    return self._elmo_vocab

#***************************************************************
if __name__ == '__main__':
  """"""
  
  from parser import Network
  configurable = Configurable()
  network = Network.from_configurable(configurable)
  network.train()
