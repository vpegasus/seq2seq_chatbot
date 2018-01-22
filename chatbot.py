#!/usr/bin/env python  
# -*- coding: utf-8 -*-
"""
@author: Prince
@file: chatbot.py 
@time: 2018-1-21 17: 42
@license: Apache License
@contact: pegasus.wenjia@foxmail.com 
"""
import tensorflow as tf
import nltk
from data_process import data, DIR
import random

from train import parameters
from seq2seq import Seq2Seq

word2id, id2word, _, _, _, _ = data(DIR, 1, 1)


def chatbot():
    """
    you can use this to chat with your own chat bot
    :return:
    """
    questionholder = tf.placeholder(shape=[None, None], dtype=tf.int32)
    quelengthholder = tf.placeholder(shape=[None], dtype=tf.int32)
    params = parameters()
    model_param_dict = {'num_units': int(params.get('modelparam', 'num_units')),
                        'num_layers': int(params.get('modelparam', 'num_layers')),
                        'vocab_size': int(params.get('modelparam', 'vocab_size')),
                        'embedding_size': int(params.get('modelparam', 'embedding_size')),
                        'beam_size': int(params.get('modelparam', 'beam_size')),
                        'use_attention': bool(params.get('modelparam', 'use_attention')),
                        'use_beam_search': bool(params.get('modelparam', 'use_beam_search')),
                        'start_token_idx': int(params.get('modelparam', 'start_token_idx')),
                        'end_token_idx': int(params.get('modelparam', 'end_token_idx')),
                        'max_gradient_norm': float(params.get('modelparam', 'max_gradient_norm'))}

    modelsaved_dir = params.get('trainparam', 'checkpoint_dir')
    seq2seq = Seq2Seq(**model_param_dict)
    decode_outputs = seq2seq.model(source_input=questionholder,
                                   source_length=quelengthholder,
                                   mode='inference',
                                   batch_size=1, keep_probs=1.0)
    predict_ids = decode_outputs.predicted_ids

    ckpt = tf.train.get_checkpoint_state(modelsaved_dir)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            saver.restore(sess, tf.train.latest_checkpoint(modelsaved_dir))
        else:
            raise ValueError('There is no chatbot baby in {}'.format(modelsaved_dir))
        question = 'start'
        print('Hello, I\'m ibot, nice to meet you!')
        while question:
            question = input(':: ')
            questionbatch, question_length = sentence2ids(question)
            answer_ids = sess.run(predict_ids, feed_dict={questionholder: questionbatch,
                                                          quelengthholder: question_length})
            answer = ids2sentence(answer_ids, model_param_dict['beam_size'])


def sentence2ids(question):
    """
    used to convert sentence that is inputed by user to id list of words in sentence.

    :param question:
    :return:
    """
    ids = []

    if question == '':
        return 'hello, let\'s chat?'
    question = nltk.word_tokenize(question)
    if len(question) > 20:
        return 'sorry, I\'m still a baby, couldn\'t understand that long sentence. =^_^='

    for word in question:
        ids.append(word2id.get(word, 3))  # 3 is the id of 'unknown' token
    ids_length = len(ids) if len(ids) <= 10 else 10
    if len(ids) > 10:
        ids = ids[:10]
    else:
        ids = ids + [0] * (10 - len(ids))
    return [ids], [ids_length]


def ids2sentence(predict_ids, beam_width):
    """
    convert ids output by chatbot into words and connect to make a complete sentence.

    :param predict_ids:
    :param beam_width:
    :return:
    """

    answers = []
    for i in range(beam_width):
        predict_seq = [id2word[idx] for idx in predict_ids[0, :, i]]
        answers.append(" ".join(predict_seq))
    final = answers[random.choice(range(beam_width))]
    sen = ''
    for i in final:
        if i not in ['?','!','.']:
            sen +=i
        else:
            break

    print('ibot: ',sen)


if __name__ == '__main__':
    chatbot()
