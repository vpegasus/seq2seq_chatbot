#!/usr/bin/env python  
# -*- coding: utf-8 -*-
"""
@author: Prince
@file: seq2seq.py 
@time: 2018-1-17 14: 35
@license: Apache License
@contact: pegasus.wenjia@foxmail.com 
"""
import tensorflow as tf
from tensorflow.python.util import nest


class Seq2Seq(object):

    def __init__(self, num_units,
                 num_layers,
                 vocab_size,
                 embedding_size,
                 beam_size,
                 use_attention,
                 use_beam_search,
                 start_token_idx,
                 end_token_idx,
                 max_gradient_norm):
        """
        This is the whole model for seq2seq chatbot.

        :param num_units: rnn size
        :param num_layers: the number of RNNs stacked in encoder and decoder
        :param vocab_size: the number of words in the dictionary
        :param embedding_size: the length of word vector
        :param beam_size: the width of beam search
        :param use_attention: switch to attention mechanism
        :param use_beam_search: switch to beam search
        :param start_token_idx: the '<go>'s index: default 1
        :param end_token_idx: the '<eos>'s index: default 2
        :param max_gradient_norm: the max gradient to update in back propagation process
        """

        self.num_units = num_units
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.special_word_idx = {'<go>': start_token_idx, '<eos>': end_token_idx}
        self.beam_width = beam_size
        self.use_attention = use_attention
        self.use_beam_search = use_beam_search
        self.batch_size = None
        self.max_gradient_norm = max_gradient_norm

        self.embedding = self._embeddiing_matrix()

        self.saver = tf.train.Saver(tf.global_variables())

    def _embeddiing_matrix(self):
        """
        use to intialize embedding matrix
        :return: (variable): embedding matrix
        """
        with tf.variable_scope('embeddingMatrix'):
            return tf.get_variable(name='embedding_matrix', shape=[self.vocab_size, self.embedding_size])

    def embedding_lookup(self, inputs):
        """
        a wrapper for embedding look up.

        :param inputs: tensor (or dataset slice): the word ids matrix: [batch_size,seq length]
        :return: embeding tensor: [batch_size, seq_length,embedding_size]
        """
        return tf.nn.embedding_lookup(self.embedding, inputs)

    def encoder(self, inputs, input_length, keep_probs):
        """
        enocoder, used to encode input tensors in to a C vector

        :param inputs: embedding tensor: [batch_size, seq_length,embedding_size]
        :param input_length: seq length of each sentence before padding.
        :param keep_probs: the probability to keep in dropout operation
        :return:
         encoder outputs: tensor: [batch_size, seq_length, embedding_size]
         encoder states: this will be a tuple having the corresponding shapes. If cells are `LSTMCells`
        `state` will be a tuple containing a `LSTMStateTuple` for each cell.
         in my instance, enocoder states:

          (LSTMStateTuple(c=<tf.Tensor 'encoder/rnn/while/Exit_2:0' shape=(?, 256)dtype=float32>,
                            h=<tf.Tensor 'encoder/rnn/while/Exit_3:0' shape=(?, 256) dtype=float32>),
          LSTMStateTuple(c=<tf.Tensor 'encoder/rnn/while/Exit_4:0' shape=(?, 256) dtype=float32>,
                            h=<tf.Tensor 'encoder/rnn/while/Exit_5:0' shape=(?, 256) dtype=float32>))

         the number 256 is the num units

        """
        with tf.variable_scope('encoder'):
            encoder_cell = _rnn_cell(self.num_units, self.num_layers, keep_probs)
            encode_outputs, encode_states = tf.nn.dynamic_rnn(encoder_cell, inputs,
                                                              sequence_length=input_length,
                                                              dtype=tf.float32)
        return encode_outputs, encode_states

    def decoder(self, encoder_output,
                encoder_state,
                source_length,
                decoder_targets,
                target_length, mode, keep_probs):
        """
        decoder used to decode the C vector into a embedding tensor, then the final response.
        :param encoder_output:
        :param encoder_state:
        :param source_length: list of each source sentence(eg. question sentence in Q&A) length before padding
        :param decoder_targets: the final correct sentence( eg. answers sentences in Q&A)
        :param target_length:
        :param mode: swith for 'train' and 'inference'
        :param keep_probs: the probability to keep for weights in drop out operation
        :return: decoder_outputs
        in my instance:
             BasicDecoderOutput(rnn_output=<tf.Tensor 'decoder/decoder/transpose:0'
                                                shape=(256, ?, 24643) dtype=float32>,
                                sample_id=<tf.Tensor 'decoder/decoder/transpose_1:0'
                                                shape=(256, ?) dtype=int32>)

            the number 24643 is the number of words in dictionary, 256 is the num_units
        """
        with tf.variable_scope('decoder'):

            # zero_state_size just the batch_size in 'train' mode, but batch_size *beam_size in 'inference' mode
            zero_state_size = self.batch_size
            if self.use_beam_search and mode == 'inference':
                encoder_output, encoder_state, source_length, zero_state_size = self._beam_search(encoder_output,
                                                                                                  encoder_state,
                                                                                                  source_length,
                                                                                                  self.beam_width)

            decoder_cell = _rnn_cell(self.num_units, self.num_layers, keep_probs)

            if self.use_attention:
                decoder_cell = self._attention(decoder_cell, encoder_output, source_length)

            # take the final state of encoder as the initial state of decoder
            decoder_initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=zero_state_size)
            decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)

            # a components used in decoder wrapper,
            # note in decoder: output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
            # `tf.layers.Dense`. Optional layer to apply to the RNN output prior
            #  to storing the result or sampling.
            output_layer = tf.layers.Dense(self.vocab_size,
                                           kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            if mode == 'train':
                # switch for 'impute_finished' in dynamic decode
                # impute_finished: Python boolean.  If `True`, then states for batch
                # entries which are marked as finished get copied through and the
                # corresponding outputs get zeroed out.  This causes some slowdown at
                # each time step, but ensures that the final state and outputs have
                # the correct values and that backprop ignores time steps that were
                # marked as finished.
                switch = True
                max_target_length = tf.reduce_max(target_length, name='max_target_length')

                # delete <end> at the ending of target sequences,
                # and add a label <go> at the beginning of each sequence.
                # please note that in 'train' mode, decoder_input is used as correct labels, see original paper.
                decoder_input = tf.strided_slice(decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])
                decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.special_word_idx['<go>']), decoder_input],
                                          1)
                decoder_inputs_embedded = self.embedding_lookup(decoder_input)

                # below two statements appear as a conventional way.
                train_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embedded,
                                                                 sequence_length=target_length,
                                                                 time_major=False,
                                                                 name='train_helper')
                decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                          helper=train_helper,
                                                          initial_state=decoder_initial_state,
                                                          output_layer=output_layer)

            elif mode == 'inference':

                # as the data we use has the max length of 10, so here, we simply set it to the
                # max length.
                switch = False
                max_target_length = 10
                start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.special_word_idx['<go>']
                end_token = self.special_word_idx['<eos>']
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell,
                                                               embedding=self.embedding,
                                                               start_tokens=start_tokens,
                                                               end_token=end_token,
                                                               initial_state=decoder_initial_state,
                                                               beam_width=self.beam_width,
                                                               output_layer=output_layer)
            else:
                raise AttributeError('Unrecognized mode {}'.format(mode))

            # dynamic decode output:decoder_outputs: namedtuple(rnn_outputs, sample_id)
            decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                                      impute_finished=switch,
                                                                      maximum_iterations=max_target_length)
            return decoder_outputs

    def _beam_search(self, encoder_output,
                     encoder_state,
                     source_length, beam_width):
        """
        a wrapper for the inputs of  beam search

        :param encoder_output:
        :param encoder_state:
        :param source_length:
        :param beam_width:
        :return:
            encoder_output,
            encoder_state,
            source_length,
            tile_size: the batch_size * beam_width, a parameters required for beam search wrapper.
        """

        def tile_batch(tensor):
            """
            a wrapper for tile batch

            Tile the batch dimension of a (possibly nested structure of) tensor(s) t.

            :param tensor: `Tensor` shaped `[batch_size, ...]`.
                   multiplier: beam_width: times to copy batch_size tensors
            :return:  A (possibly nested structure of) `Tensor` shaped
                     `[batch_size * multiplier, ...]`
            """
            return tf.contrib.seq2seq.tile_batch(tensor, multiplier=beam_width)

        encoder_output = tile_batch(encoder_output)
        encoder_state = nest.map_structure(tile_batch, encoder_state)
        source_length = tile_batch(source_length)

        tile_size = self.batch_size * self.beam_width
        return encoder_output, encoder_state, source_length, tile_size

    def _attention(self, cell, encoder_output, source_length):
        """
        a wrapper for attention mechanism.
        :param cell: decoder(cells)
        :param encoder_output:
        :param source_length: list of each sentence length before padding
        :return:
        """
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.num_units,
                                                                   memory=encoder_output,
                                                                   memory_sequence_length=source_length)
        cell = tf.contrib.seq2seq.AttentionWrapper(cell=cell,
                                                   attention_mechanism=attention_mechanism,
                                                   attention_layer_size=self.num_units,
                                                   name='attention_wrapper')
        return cell

    def model(self, source_input,
              target_input=None,
              source_length=None,
              target_length=None,
              mode='inference',
              batch_size=1,
              keep_probs=1.0):
        """
        a wrapper for the mode three main layers: embedding_layer, encoder layer and decoder layer.
        and the params share the same meaning as mentioned above.

        :param source_input:
        :param target_input:
        :param source_length:
        :param target_length:
        :param mode:
        :param batch_size:
        :param keep_probs:
        :return:
        """
        source_input = self.embedding_lookup(source_input)

        # initialize batch_size
        self.batch_size = batch_size

        encoder_output, encoder_state = self.encoder(source_input, source_length, keep_probs)
        decoder_outputs = self.decoder(encoder_output,
                                       encoder_state,
                                       source_length,
                                       target_input,
                                       target_length,
                                       mode,
                                       keep_probs)
        return decoder_outputs

    def train(self, decode_outputs,
              targets,
              target_length,
              learning_rate):
        """
        a wrapper for training.

        :param decode_outputs:
        :param targets:
        :param target_length:
        :param learning_rate:
        :return:
        """
        max_target_length = tf.reduce_max(target_length, name='max_target_len')

        # the following statement do a mask operation, the note cited from original comments
        # Returns a mask tensor representing the first N positions of each cell.
        #
        #   If `lengths` has shape `[d_1, d_2, ..., d_n]` the resulting tensor `mask` has
        #   dtype `dtype` and shape `[d_1, d_2, ..., d_n, maxlen]`, with
        #
        #   ```
        #   mask[i_1, i_2, ..., i_n, j] = (j < lengths[i_1, i_2, ..., i_n])
        #   ```
        #
        #   Examples:
        #
        #   ```python
        #   tf.sequence_mask([1, 3, 2], 5)  # [[True, False, False, False, False],
        #                                   #  [True, True, True, False, False],
        #                                   #  [True, True, False, False, False]]
        #
        #   tf.sequence_mask([[1, 3],[2,0]])  # [[[True, False, False],
        #                                     #   [True, True, True]],
        #                                     #  [[True, True, False],
        #                                     #   [False, False, False]]]
        #   ```
        mask = tf.sequence_mask(target_length, max_target_length, dtype=tf.float32,
                                name='masks')
        logits = tf.identity(decode_outputs.rnn_output)
        predict_logits = tf.argmax(logits, axis=-1, name='train_pred')
        loss = tf.contrib.seq2seq.sequence_loss(logits=logits,
                                                targets=targets,
                                                weights=mask)
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

        for grad, var in zip(clip_gradients, trainable_params):
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))
        summary_merge = tf.summary.merge_all()
        return train_op, loss, summary_merge, predict_logits


def _rnn_cell(num_units,
              num_layers,
              keep_prob=1.0,
              cell_type='lstm'):
    """
    a wraper for rnn cells

    :param num_units:
    :param num_layers:
    :param keep_prob:
    :param cell_type: which type of rnn you use, you can expand this mode as you like.
    :return:
    """
    if cell_type == 'lstm':
        def lstm_cell(rnn_size, keepprob):
            rnn = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
            return tf.contrib.rnn.DropoutWrapper(rnn, output_keep_prob=keepprob)

        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(num_units, keep_prob) for _ in range(num_layers)])
    elif cell_type == 'gru':
        def gru_cell(rnn_size, keepprob):
            rnn = tf.nn.rnn_cell.GRUCell(rnn_size)
            return tf.contrib.rnn.DropoutWrapper(rnn, output_keep_prob=keepprob)

        cell = tf.nn.rnn_cell.MultiRNNCell([gru_cell(num_units, keep_prob) for _ in range(num_layers)])
    else:
        raise AttributeError('Unrecognized type {}'.format(cell_type))
    return cell
