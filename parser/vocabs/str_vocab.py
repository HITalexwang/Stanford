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
import sys
from collections import Counter

import numpy as np
import tensorflow as tf

from parser import Configurable

__all__ = ['FeatVocab']

#***************************************************************
class StrVocab(Configurable):
  """"""
  
  ROOT = 0
  
  #=============================================================
  def __init__(self, *args, **kwargs):
    """"""
    
    super(StrVocab, self).__init__(*args, **kwargs)
    self._name = 'strs'
  
  #=============================================================
  def generate_placeholder(self):
    """"""
    return
  
  #=============================================================
  def set_feed_dict(self, data, feed_dict):
    """"""
    return
  
  #=============================================================
  def index(self, token):
    return 0
  
  #=============================================================
  @property
  def depth(self):
    return None
  @property
  def conll_idx(self):
    return self._conll_idx
  @property
  def name(self):
    return self._name
  

  #=============================================================
  def __getitem__(self, key):
    if isinstance(key, basestring):
      return key
    elif hasattr(key, '__iter__'):
      return [self[k] for k in key]
    else:
      raise ValueError('key to StrVocab.__getitem__ must be string')
    return

#***************************************************************
class FeatVocab(StrVocab):
  _conll_idx = 5