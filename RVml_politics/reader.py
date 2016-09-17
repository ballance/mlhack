# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
#from datashape.coretypes import int32


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import tensorflow as tf


def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename_trump, filename_hillary):
  data = _read_words(filename_trump) + _read_words(filename_hillary);

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data]
  
#def _file_to_word_ids(filename, word_to_id):
#  data = _read_words(filename)
#  data_sub = data[0:5000]
#  return [word_to_id[word] for word in data_sub]


def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path_trump = "/Users/pal004/PoliticPredictors/PregenData/TrainData/protrump.t1andh3.filtered.shuf.n26k.syntaxnet.smallTest.txt";
  train_path_hillary = "/Users/pal004/PoliticPredictors/PregenData/TrainData/prohillary.h1andt3.filtered.shuf.n26k.syntaxnet.smallTest.txt"
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path_trump, train_path_hillary)
  train_data_trump = _file_to_word_ids(train_path_trump, word_to_id)
  train_data_hillary = _file_to_word_ids(train_path_hillary, word_to_id)
  valid_data = ""
  test_data = ""
   
#  valid_data = _file_to_word_ids(valid_path, word_to_id)
#  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data_trump, train_data_hillary, valid_data, test_data, vocabulary


def ptb_iterator(raw_data_trump, raw_data_hillary, batch_size, num_steps):
  """Iterate on the raw PTB data.

  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.

  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.

  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  raw_data_trump = np.array(raw_data_trump, dtype=np.int32)
  raw_data_hillary = np.array(raw_data_hillary, dtype=np.int32)

  data_len = min (len(raw_data_trump), len(raw_data_hillary))
  batch_len = data_len // batch_size
  data_trump = np.zeros([batch_size, batch_len], dtype=np.int32)
  data_hillary = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data_trump[i] = raw_data_trump[batch_len * i:batch_len * (i + 1)]
    data_hillary[i] = raw_data_hillary[batch_len * i:batch_len * (i + 1)]

  epoch_size = (batch_len - 1) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x_trump = data_trump[:, i*num_steps:(i+1)*num_steps]
#    y_trump = data_trump[:, i*num_steps+1:(i+1)*num_steps+1]
    y_trump = np.array ([np.array ([[0, 1]] * x_trump.shape[1], dtype=np.int32) for ii in range(x_trump.shape[0])], dtype=np.int32)
    x_hillary = data_hillary[:, i*num_steps:(i+1)*num_steps]
#    y_hillary = data_hillary[:, i*num_steps+1:(i+1)*num_steps+1]
    y_hillary = np.array ([np.array ([[1, 0]] * x_hillary.shape[1], dtype=np.int32) for ii in range(x_hillary.shape[0])], dtype=np.int32)
    yield (x_trump, y_trump, x_hillary, y_hillary)
