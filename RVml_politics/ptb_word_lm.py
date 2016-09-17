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

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from tensorflow.models.rnn import rnn

import time

import numpy as np
import tensorflow as tf

import subprocess

#from tensorflow.models.rnn.ptb import reader
import reader as reader
import os
import json
import datetime
from twitter import Api
from time import sleep

from subprocess import call
import random;

# Either specify a set of keys here or use os.getenv('CONSUMER_KEY') style
# assignment:

CONSUMER_KEY = 'Nj6hl39zp3gDxejVuW1mglI5p' 
# CONSUMER_KEY = os.getenv("CONSUMER_KEY", None)
CONSUMER_SECRET = '7TivplTj4eWL0zRq5EwTxpd5E4bdWH35fCDCJIXZazfvDB0YOE'
# CONSUMER_SECRET = os.getenv("CONSUMER_SECRET", None)
ACCESS_TOKEN = '720371915336998912-q9erL7k9EqV85g9AoPev3gx1H23GodL'
# ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", None)
ACCESS_TOKEN_SECRET = 'PwnA7AXTbnDpb10d7If55XtCQzm3GAzaja5shvhBakRTb'
# ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET", None)

# Users to watch for should be a list. This will be joined by Twitter and the
# data returned will be for any tweet mentioning:
# @twitter *OR* @twitterapi *OR* @support.
         
H1 = ['#ImWithHer','#ilikehillary','#momsdemandhillary', '#StrongerTogether'] #Hillary Good
H2 = ['#Hillary'] #Hillary Neutral 
H3 = ['#NeverHillary','#CrookedHillary'] #Hillary Bad
T1 = ['#TrumpTrain','#MAGA', 'TrumpPence16','#Trump2016','#TrumpArmy'] #Trump Good
T2 = ['#Trump'] #Trump Neutral
T3 = ['#NeverTrump', '#DumpTrump','#StopTrump','#TrumpIsAJoke'] #Trump Bad
controlTweet = ['#poetry'] #
mixed = ['']

# Since we're going to be using a streaming endpoint, there is no need to worry
# about rate limits.
api = Api(CONSUMER_KEY,
          CONSUMER_SECRET,
          ACCESS_TOKEN,
          ACCESS_TOKEN_SECRET)

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")

FLAGS = flags.FLAGS


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps, 2])

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size])
      inputs = tf.nn.embedding_lookup(embedding, self._input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
#    inputs = [tf.squeeze(input_, [1])
#              for input_ in tf.split(1, tf.shape (self._input_data)[1], inputs)]
#    outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
#     
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
#      for time_step in tf.range(tf.shape(self._input_data)[1]):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    
    softmax_w_2 = tf.get_variable("softmax_w_2", [size, 2])
    softmax_b_2 = tf.get_variable("softmax_b_2", [2])
    logits_2 = self._logits_2 = tf.matmul(output, softmax_w_2) + softmax_b_2
    loss_2 = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits(logits_2, tf.reshape(tf.to_float (self._targets), [self.batch_size * self.num_steps, 2]))); 
    self._testvar = tf.reshape(tf.to_float (self._targets), [self.batch_size * self.num_steps, 2]);
#    loss_2 = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits(logits_2, tf.to_float (tf.transpose ([tf.reshape(self._targets, [-1])]))));
#    loss_2 = tf.reduce_mean (tf.nn.cross_entropy_with_logits(logits_2, tf.to_float (tf.transpose ([tf.reshape(self._targets, [-1])]))));
    self._softmax_2 = tf.nn.softmax(logits_2)
    self._cost = cost = loss_2 / batch_size
    self._final_state = state
    
#    softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
#    softmax_b = tf.get_variable("softmax_b", [vocab_size])
#    logits = tf.matmul(output, softmax_w) + softmax_b
#    loss = tf.nn.seq2seq.sequence_loss_by_example(
#        [logits],
#        [tf.reshape(self._targets, [-1])],
#        [tf.ones([batch_size * num_steps])])
#    self._cost = cost = tf.reduce_sum(loss) / batch_size
#    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def logits_2(self):
    return self._logits_2

  @property
  def softmax_2(self):
    return self._softmax_2

  @property
  def testvar(self):
    return self._testvar

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
#  max_max_epoch = 3
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


def run_epoch(session, m, data_trump, data_hillary, eval_op, verbose=False):
  """Runs the model on the given data."""
  epoch_size = ((min (len(data_trump),len(data_hillary)) // m.batch_size) - 1) // m.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = m.initial_state.eval()
  for step, (x_trump, y_trump, x_hillary, y_hillary) in enumerate(reader.ptb_iterator(data_trump, data_hillary, 
                                                    m.batch_size,
                                                    m.num_steps)):
    cost1, logits_2_trump, softmax_2_trump, testvar_trump, state, _ = session.run([m.cost, m.logits_2, m.softmax_2, m.testvar, m.final_state, eval_op],
                                 {
                                  m.input_data: x_trump,
                                  m.targets: y_trump,
                                  m.initial_state: state})
    costs += cost1
    iters += m.num_steps
    
    cost2, logits_2_hillary, softmax_2_hillary, testvar_hillary, state, _ = session.run([m.cost, m.logits_2, m.softmax_2, m.testvar, m.final_state, eval_op],
                                 {
                                  m.input_data: x_hillary,
                                  m.targets: y_hillary,
                                  m.initial_state: state})
    costs += cost2
    iters += m.num_steps

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * m.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data_trump, train_data_hillary, valid_data, test_data, _ = raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
#  eval_config.num_steps = 1
  eval_config.num_steps = 20

  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
      m = PTBModel(is_training=True, config=config)
    # the syntaxnet output is ready, lets run this 
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config)

    tf.initialize_all_variables().run()
    saver = tf.train.Saver()

    
#    for i in range(config.max_max_epoch):
#      lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
#      m.assign_lr(session, config.learning_rate * lr_decay)
#
#      print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
#      train_perplexity = run_epoch(session, m, train_data_trump, train_data_hillary, 
#                                   m.train_op,
#                                   verbose=True)
#      saver.save (session, "savedTrainedModel")
#      print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
##      valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
##      print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
##    test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
##    print("Test Perplexity: %.3f" % test_perplexity)
    
    
    saver.restore(session, "savedTrainedModel")
#    test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
#    print("Test Perplexity: %.3f" % test_perplexity)


    count = 0
    for line in api.GetStreamFilter(track=['#Hillary', '#Trump']):
        with open('/Users/pal004/PoliticPredictors/input.json', 'w') as f:
            f.write(json.dumps(line))
            f.flush ();
            count = count + 1
            print (count)
            
        filtered_output = subprocess.Popen(["/usr/bin/Rscript", "/Users/pal004/PoliticPredictors/filter_tweet.R", "/Users/pal004/PoliticPredictors/input.json"], stdout=subprocess.PIPE).communicate()[0]
        myarray = filtered_output.split('\n');
        print ("Input: %s" % (filtered_output));
       
        # if long is note null, lets continue 
        if (len (myarray) == 3 and  myarray[1] != "0" and myarray[2] != "0"):
            if os.path.isfile('/Users/pal004/PoliticPredictors/syntaxnetoutput.formodel.txt'):
                os.remove ('/Users/pal004/PoliticPredictors/syntaxnetoutput.formodel.txt');
                
            with open('/Users/pal004/PoliticPredictors/forsyntaxnet.json', 'w') as f: # put filtered tweet in file
                myoutput = "%s" % (myarray[0]);
                f.write (myoutput);
                f.flush ();
                
            while (True):
                if os.path.isfile('/Users/pal004/PoliticPredictors/syntaxnetoutput.formodel.txt'):
                    break;
                
            mtest = PTBModel(is_training=False, config=eval_config)
            #    test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
                
#            call(["/bin/bash", "/Users/pal004/PoliticPredictors/runsyntaxnet.sh"])
#            notused_output = subprocess.Popen(["/bin/bash", "/Users/pal004/PoliticPredictors/runsyntaxnet.sh"], stdout=subprocess.PIPE).communicate()[0]
#            os.system ("/bin/bash /Users/pal004/PoliticPredictors/runsyntaxnet.sh");
#            subprocess.call(['/Users/pal004/PoliticPredictors/runsyntaxnet.sh'], shell=True)

            # pull from above model instead.. uncomment above
            myrandom = random.uniform (0,1);
            
            with open('/Users/pal004/PoliticPredictors/output.json', 'w') as f:
                myoutput = "%s\n%s\n%s\n%s\n%s" % (myarray[0], myarray[1], myarray[2], str (myrandom), "EOL");
                f.write (myoutput);
                f.flush ();
                print (myoutput);
#            exit ("hi")
        


if __name__ == "__main__":
  tf.app.run()
