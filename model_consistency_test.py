# Make sure that first few predictions match expectations
# (where "expectations" = first few predicted results from original implementation)

import pytest
import numpy as np
import tensorflow as tf
import lm1b.utils.vocab as vocab_util
import lm1b.model.vocab_nodes as vocab_nodes
import lm1b.model.model_nodes as model_nodes
import lm1b.utils.util as util
from lm1b.utils.util import merge
import lm1b.hparams
import os
import sys

import numpy as np
from six.moves import xrange
import tensorflow as tf

from google.protobuf import text_format
import lm1b.data_utils as data_utils

FLAGS = tf.flags.FLAGS
# General flags.
tf.flags.DEFINE_string('mode', 'eval',
                       'One of [sample, eval, dump_emb, dump_lstm_emb]. '
                       '"sample" mode samples future word predictions, using '
                       'FLAGS.prefix as prefix (prefix could be left empty). '
                       '"eval" mode calculates perplexity of the '
                       'FLAGS.input_data. '
                       '"dump_emb" mode dumps word and softmax embeddings to '
                       'FLAGS.save_dir. embeddings are dumped in the same '
                       'order as words in vocabulary. All words in vocabulary '
                       'are dumped.'
                       'dump_lstm_emb dumps lstm embeddings of FLAGS.sentence '
                       'to FLAGS.save_dir.')

tf.flags.DEFINE_string('pbtxt', '',
                       'GraphDef proto text file used to construct model '
                       'structure.')

tf.flags.DEFINE_string('ckpt', '',
                       'Checkpoint directory used to fill model values.')
tf.flags.DEFINE_string('vocab_file', '', 'Vocabulary file.')
tf.flags.DEFINE_string('save_dir_0', '',
                       'Used for "dump_emb" mode to save word embeddings.')
tf.flags.DEFINE_string('save_dir_1', '',
                       'Used for "dump_emb" mode to save word embeddings.')
tf.flags.DEFINE_string('save_dir', '',
                       'Used for "dump_emb" mode to save word embeddings.')
# sample mode flags.
tf.flags.DEFINE_string('prefix', '',
                       'Used for "sample" mode to predict next words.')
tf.flags.DEFINE_integer('max_sample_words', 100,
                        'Sampling stops either when </S> is met or this number '
                        'of steps has passed.')
tf.flags.DEFINE_integer('num_samples', 3,
                        'Number of samples to generate for the prefix.')
# dump_lstm_emb mode flags.
tf.flags.DEFINE_string('sentence', '',
                       'Used as input for "dump_lstm_emb" mode.')
# eval mode flags.
tf.flags.DEFINE_string('input_data', '',
                       'Input data files for eval model.')
tf.flags.DEFINE_integer('max_eval_steps', 1000000,
                        'Maximum mumber of steps to run "eval" mode.')


# For saving demo resources, use batch size 1 and step 1.
BATCH_SIZE = 1
NUM_TIMESTEPS = 1
MAX_WORD_LEN = 50

run_config = util.load_config("config.json")

hparams = lm1b.hparams.get_default_hparams()

hparams.sequence_length = 1
hparams.max_word_length = 50
hparams.chars_padding_id = 4

def _SampleSoftmax(softmax):
  return min(np.sum(np.cumsum(softmax) < np.random.rand()), len(softmax) - 1)

def _SampleModel(prefix_words, vocab):
  """Predict next words using the given prefix words.
  Args:
    prefix_words: Prefix words.
    vocab: Vocabulary. Contains max word chard id length and converts between
        words and ids.
  """
  targets = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
  weights = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)

  sess, t, input = _LoadModel()

  if prefix_words.find('<S>') != 0:
    prefix_words = '<S> ' + prefix_words

  prefix = [vocab.word_to_id(w) for w in prefix_words.split()]
  prefix_char_ids = [vocab.word_to_char_ids(w) for w in prefix_words.split()]
  for _ in xrange(FLAGS.num_samples):
    inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
    char_ids_inputs = np.zeros(
        [BATCH_SIZE, NUM_TIMESTEPS, vocab.max_word_length], np.int32)
    samples = prefix[:]
    char_ids_samples = prefix_char_ids[:]
    sent = ''
    while True:
      inputs[0, 0] = samples[0]
      char_ids_inputs[0, 0, :] = char_ids_samples[0]
      samples = samples[1:]
      char_ids_samples = char_ids_samples[1:]

      softmax = sess.run(t['softmax'],
                         feed_dict={input: char_ids_inputs})

      sample = _SampleSoftmax(softmax[0])
      sample_char_ids = vocab.word_to_char_ids(vocab.id_to_word(sample))

      if not samples:
        samples = [sample]
        char_ids_samples = [sample_char_ids]
      sent += vocab.id_to_word(samples[0]) + ' '
      sys.stderr.write('%s\n' % sent)

      if (vocab.id_to_word(samples[0]) == '</S>' or
          len(sent) > FLAGS.max_sample_words):
        break


def _DumpSentenceEmbedding(sentence, vocab):
  """Predict next words using the given prefix words.
  Args:
    sentence: Sentence words.
    vocab: Vocabulary. Contains max word chard id length and converts between
        words and ids.
  """
  LSTM_hidden_0 = {}
  LSTM_hidden_1 = {}
  LSTM_memory_0 = {}
  LSTM_memory_1 = {}

  print(sentence)
  sentence = sentence.encode("utf-8").decode()
  targets = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
  weights = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)

  sess, t, input = _LoadModel()
  print("t:",t)
  if sentence.find('<S>') != 0:
    sentence = '<S> ' + sentence

  word_ids = [vocab.word_to_id(w) for w in sentence.split()]
  char_ids = [vocab.word_to_char_ids(w) for w in sentence.split()]

  inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
  char_ids_inputs = np.zeros(
      [BATCH_SIZE, NUM_TIMESTEPS, vocab.max_word_length], np.int32)
  for i in xrange(len(word_ids)):
    inputs[0, 0] = word_ids[i]
    char_ids_inputs[0, 0, :] = char_ids[i]
    print(i,sentence.split()[i])
    # Add 'lstm/lstm_0/control_dependency' if you want to dump previous layer
    # LSTM.
    cell_out, cell_state = sess.run([t['cell_out_all_layers'],t['cell_state_all_layers']],
                        feed_dict={input: char_ids_inputs,})

    LSTM_hidden_0[i] = cell_state[0].h
    LSTM_hidden_1[i] = cell_state[1].h

    LSTM_memory_0[i] = cell_state[0].c
    LSTM_memory_1[i] = cell_state[1].c

  fname = os.path.join(FLAGS.save_dir, 'lstm_hidden_0')
  np.save(fname,LSTM_hidden_0)

  fname = os.path.join(FLAGS.save_dir, 'lstm_hidden_1')
  np.save(fname, LSTM_hidden_1)

  fname = os.path.join(FLAGS.save_dir, 'lstm_memory_0')
  np.save(fname, LSTM_memory_0)

  fname = os.path.join(FLAGS.save_dir, 'lstm_memory_1')
  np.save(fname, LSTM_memory_1)


def _LoadModel():
  graph = tf.Graph()
  with graph.as_default():
    graph_nodes = {}
    # Attach word / character lookup tables
    graph_nodes = merge(graph_nodes, vocab_nodes.attach_vocab_nodes(run_config['vocab_path'], hparams=hparams))


    # placeholder for input sentences (encoded as character arrays)
    input_seqs = tf.placeholder(dtype=tf.int64, shape=(BATCH_SIZE,hparams.sequence_length, hparams.max_word_length))
    # attach the model itself
    graph_nodes = merge(graph_nodes, model_nodes.attach_inference_nodes(input_seqs, hparams=hparams))
    # attach a helper to lookup top k predictions
    graph_nodes = merge(graph_nodes,
                        model_nodes.attach_predicted_word_nodes(graph_nodes['logits'],
                                                                graph_nodes['lookup_id_to_word'],
                                                                k=10,
                                                                hparams=hparams))

    sess = tf.Session(graph=graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    # Restore the original pre-trained model into this graph
    model_nodes.restore_original_lm1b(sess, run_config=run_config)


  return sess, graph_nodes , input_seqs


def main(unused_argv):
  if len(unused_argv) != 1:  # prints a message if you've entered flags incorrectly
    raise Exception("Problem with flags: %s" % unused_argv)

  vocab = data_utils.CharsVocabulary(FLAGS.vocab_file, MAX_WORD_LEN)

  if FLAGS.mode == 'eval':
    dataset = data_utils.LM1BDataset(FLAGS.input_data, vocab)
    _EvalModel(dataset)
  elif FLAGS.mode == 'sample':
    _SampleModel(FLAGS.prefix, vocab)
  elif FLAGS.mode == 'dump_emb':
    _DumpEmb(vocab)
  elif FLAGS.mode == 'dump_lstm_emb':
    _DumpSentenceEmbedding(FLAGS.sentence, vocab)
  else:
    raise Exception('Mode not supported.')


if __name__ == '__main__':
  tf.app.run()



