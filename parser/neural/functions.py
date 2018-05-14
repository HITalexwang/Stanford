#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

#***************************************************************
sig_const = np.arctanh(1/3)
tanh_const = np.arctanh(np.sqrt(1/3))

def gate(x):
  return tf.nn.sigmoid(2*x)

def tanh(x):
  return tf.nn.tanh(x)

def gated_tanh(x):
  if isinstance(x, list):
    cell_act, gate_act = x
  else:
    dim = len(x.get_shape().as_list())-1
    cell_act, gate_act = tf.split(x, 2, dim)
  return gate(gate_act) * tanh(cell_act)

def identity(x):
  return tf.identity(x)

def gated_identity(x):
  if isinstance(x, list):
    cell_act, gate_act = x
  else:
    dim = len(x.get_shape().as_list())-1
    cell_act, gate_act = tf.split(x, 2, dim)
  return gate(gate_act) * identity(cell_act)

def gated_relu(x):
  if isinstance(x, list):
    cell_act, gate_act = x
  else:
    dim = len(x.get_shape().as_list())-1
    cell_act, gate_act = tf.split(x, 2, dim)
  return gate(gate_act) * relu(cell_act)

def gated_leaky_relu(x):
  if isinstance(x, list):
    cell_act, gate_act = x
  else:
    dim = len(x.get_shape().as_list())-1
    cell_act, gate_act = tf.split(x, 2, dim)
  return gate(gate_act) * leaky_relu(cell_act)

def softplus(x):
  return tf.softplus(2*x)/2

def elu(x):
  return tf.nn.elu(x)

def relu(x):
  return tf.nn.relu(x)

def leaky_relu(x):
  return tf.maximum(.1*x, x)