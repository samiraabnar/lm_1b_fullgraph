# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Eval pre-trained 1 billion word language model.
"""
import os
import sys

import numpy as np
from six.moves import xrange
import tensorflow as tf

from google.protobuf import text_format
import data_utils

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

import numpy as np
import tensorflow as tf
import lm1b.utils.vocab as vocab_util
import lm1b.model.vocab_nodes as vocab_nodes
import lm1b.model.model_nodes as model_nodes
import lm1b.utils.util as util
from lm1b.utils.util import merge
import lm1b.hparams

run_config= util.load_config("config.json")

hparams= lm1b.hparams.get_default_hparams()

hparams.sequence_length = 1
hparams.max_word_length = 50
hparams.chars_padding_id = 4


graph= tf.Graph()
with graph.as_default():
    graph_nodes={}
    # Attach word / character lookup tables
    graph_nodes=merge( graph_nodes, vocab_nodes.attach_vocab_nodes(run_config['vocab_path'], hparams=hparams) )
    char_to_id_lookup_table = graph_nodes['lookup_char_to_id']
    word_to_id_lookup_table = graph_nodes['lookup_word_to_id']

    # placeholder for input sentences (encoded as character arrays)
    input_seqs=tf.placeholder(dtype=tf.int64, shape=(hparams.sequence_length, hparams.max_word_length))
    # attach the model itself
    graph_nodes=merge( graph_nodes, model_nodes.attach_inference_nodes(input_seqs, hparams=hparams))
    # attach a helper to lookup top k predictions
    graph_nodes=merge( graph_nodes,
                       model_nodes.attach_predicted_word_nodes(graph_nodes['logits'],
                                                               graph_nodes['lookup_id_to_word'],
                                                               k=10,
                                                               hparams=hparams))



    sess= tf.Session(graph=graph)
    sess.run( tf.global_variables_initializer() )
    sess.run( tf.tables_initializer() )

    # Restore the original pre-trained model into this graph
    model_nodes.restore_original_lm1b(sess, run_config=run_config)
    current_step = model_nodes['global_step'].eval(session=sess)
    sys.stderr.write('Loaded step %d.\n' % current_step)


