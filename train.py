#!/usr/bin/env python  
# -*- coding: utf-8 -*-
"""
@author: Prince
@file: train.py
@time: 2018-1-20 23: 01
@license: Apache License
@contact: pegasus.wenjia@foxmail.com 
"""
from configparser import ConfigParser
import tensorflow as tf
import os

from seq2seq import Seq2Seq
from data_process import data, DIR
import math

def parameters():
    cf = ConfigParser()
    cf.read('config.ini')
    return cf


def training():
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

    batch_size = int(params.get('trainparam', 'batch_size'))
    learning_rate = float(params.get('trainparam', 'learning_rate'))
    keep_prob = float(params.get('trainparam', 'keep_prob'))
    epochs = int(params.get('trainparam', 'epochs'))
    modelsaved_dir = params.get('trainparam', 'checkpoint_dir')
    savedname = params.get('trainparam', 'checkpoint_name')
    _, _, questionbatch, answerbatch, qlengthbatch, alengthbatch = data(DIR, epochs, batch_size)

    seq2seq = Seq2Seq(**model_param_dict)
    decode_outputs = seq2seq.model(questionbatch, answerbatch, qlengthbatch, alengthbatch, 'train',
                                   batch_size, keep_prob)
    train_op, loss, summary_merge, predicts = seq2seq.train(decode_outputs, answerbatch, alengthbatch, learning_rate)

    ckpt = tf.train.get_checkpoint_state(modelsaved_dir)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            saver.restore(sess, tf.train.latest_checkpoint(modelsaved_dir))
        else:
            print('Create model from scratch..')
            sess.run(tf.global_variables_initializer())
        step = 0
        summary_writer = tf.summary.FileWriter(modelsaved_dir, graph=sess.graph)
        while True:
            try:
                step += 1
                _, temploss, tempsummary = sess.run([train_op, loss, summary_merge])
                # temploss should not be named as loss, as the name has been used in the model, or, will raise error:
                # eg: 'Fetch argument 10.112038 has invalid type <class 'numpy.float32'>, must be a string or Tensor.
                # (Can not convert a float32 into a Tensor or Operation.)'
                print('run step: ', step,end='\r')
                if step % int(params.get('trainparam', 'steps_per_checkpoint')) == 0:
                    perplexity = math.exp(float(temploss)) if temploss < 300 else float('inf')
                    print('save at step: ', step,'perplexity: ',perplexity)
                    summary_writer.add_summary(tempsummary, step)
                    checkpoint_path = os.path.join(modelsaved_dir, savedname)
                    saver.save(sess, checkpoint_path, global_step=step)
            except tf.errors.OutOfRangeError:
                print('done')
                break



if __name__ == '__main__':
    training()
