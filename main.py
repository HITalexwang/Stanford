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
import os
import sys
import codecs
from argparse import ArgumentParser

from parser import Configurable
from parser import Network

# TODO make the pretrained vocab names a list given to TokenVocab
#***************************************************************
# Set up the argparser
argparser = ArgumentParser('Network')
argparser.add_argument('--save_dir', required=True)
subparsers = argparser.add_subparsers()
section_names = set()
# --section_name opt1=value1 opt2=value2 opt3=value3
with codecs.open('config/defaults.cfg') as f:
  section_regex = re.compile('\[(.*)\]')
  for line in f:
    match = section_regex.match(line)
    if match:
      section_names.add(match.group(1).lower().replace(' ', '_'))
#===============================================================
# Train
#---------------------------------------------------------------
def train(save_dir, **kwargs):
  """"""
  
  load = kwargs.pop('load')
  try:
    if not load and os.path.isdir(save_dir):
      raw_input('Save directory already exists. Press <Enter> to continue or <Ctrl-c> to abort.')
      if os.path.isfile(os.path.join(save_dir, 'config.cfg')):
        os.remove(os.path.join(save_dir, 'config.cfg'))
  except KeyboardInterrupt:
    sys.exit(0)
  
  #print (kwargs)
  #print ("train files:",kwargs['train_files'])
  print ("### Initializing ###")
  network = Network(**kwargs)
  print ("### Initialized ###")
  network.train(load=load)
  return
#---------------------------------------------------------------

train_parser = subparsers.add_parser('train')
train_parser.set_defaults(action=train)
train_parser.add_argument('--load', action='store_true')
train_parser.add_argument('--config_file')
for section_name in section_names:
  train_parser.add_argument('--'+section_name, nargs='+')

#===============================================================
# Parse
#---------------------------------------------------------------
def parse(save_dir, **kwargs):
  """"""
  
  kwargs['config_file'] = os.path.join(save_dir, 'config.cfg')
  #files = kwargs.pop('files')
  # this will be removed in Network.__init__
  files = kwargs['files']
  output_file = kwargs.pop('output_file', None)
  output_dir = kwargs.pop('output_dir', None)
  if len(files) > 1 and output_file is not None:
    raise ValueError('Cannot provide a value for --output_file when parsing multiple files')
  kwargs['is_evaluation'] = True

  network = Network(**kwargs)
  network.parse(files, output_file=output_file, output_dir=output_dir)
  return
#---------------------------------------------------------------

parse_parser = subparsers.add_parser('parse')
parse_parser.set_defaults(action=parse)
parse_parser.add_argument('files', nargs='+')
for section_name in section_names:
  parse_parser.add_argument('--'+section_name, nargs='+')
parse_parser.add_argument('--output_file')
parse_parser.add_argument('--output_dir')
parse_parser.add_argument('--elmo_file')

#===============================================================
# Ensemble
#---------------------------------------------------------------
def ensemble(save_dir, **kwargs):

  #print("save_dir:",save_dir)
  other_save_dirs = kwargs.pop('other_save_dirs', None)
  if other_save_dirs is None:
    raise ValueError('### \'--other_save_dirs\' must be provided for ensemble! ###')
  #print("other_save_dirs",other_save_dirs)
  sum_type = kwargs.pop('sum_type', None)
  if sum_type is None:
    sum_type = 'prob'
  if sum_type not in ['prob', 'score']:
    raise ValueError('### \'--sum_type\' is not \'prob\' or \'score\'! ###')
  print ('sum_type:', sum_type)
  sum_weights = kwargs.pop('sum_weights', None)
  if sum_weights is not None:
  	sum_weights = [float(w) for w in sum_weights]
  	for w in sum_weights:
  		assert(w > 0 and w <= 1.0)
  	print ('sum_weights:', sum_weights)

  ts_lstms = kwargs.pop('ts_lstms', None)
  if ts_lstms is not None:
    tslstm_option = [(int(w), int(t)) for w, t in [ts.strip().split(',') for ts in ts_lstms.strip().split('/')]]
  else:
    tslstm_option = None
  print ('TS-LSTM:', tslstm_option)

  kwargs['config_file'] = os.path.join(save_dir, 'config.cfg')
  #files = kwargs.pop('files')
  # this will be removed in Network.__init__
  files = kwargs['files']
  output_file = kwargs.pop('output_file', None)
  output_dir = kwargs.pop('output_dir', None)
  if len(files) > 1 and output_file is not None:
    raise ValueError('Cannot provide a value for --output_file when parsing multiple files')
  kwargs['is_evaluation'] = True
  #print("ensemble config:\nfiles:{}\noutput_file:{}\noutput_dir:{}\n".format(files, output_file, output_dir))
  network = Network(**kwargs)
  network.ensemble(files, other_save_dirs, sum_type, sum_weights=sum_weights, tslstm_option=tslstm_option,
                    output_file=output_file, output_dir=output_dir)
  return
#---------------------------------------------------------------

ens_parser = subparsers.add_parser('ensemble')
ens_parser.set_defaults(action=ensemble)
ens_parser.add_argument('files', nargs='+')
for section_name in section_names:
  ens_parser.add_argument('--'+section_name, nargs='+')
ens_parser.add_argument('--other_save_dirs', nargs='+')
ens_parser.add_argument('--sum_type')
ens_parser.add_argument('--sum_weights', nargs='+')
ens_parser.add_argument('--output_file')
ens_parser.add_argument('--output_dir')
ens_parser.add_argument('--elmo_file')
ens_parser.add_argument('--ts_lstms') # 4,4;8,8;16,16

#***************************************************************
# Parse the arguments
kwargs = vars(argparser.parse_args())
action = kwargs.pop('action')
save_dir = kwargs.pop('save_dir')
kwargs = {key: value for key, value in kwargs.iteritems() if value is not None}
for section, values in kwargs.iteritems():
  if section in section_names:
    values = [value.split('=', 1) for value in values]
    kwargs[section] = {opt: value for opt, value in values}
if 'default' not in kwargs:
  kwargs['default'] = {}
kwargs['default']['save_dir'] = save_dir
action(save_dir, **kwargs)  


