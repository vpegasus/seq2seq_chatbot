#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Prince
@file: data_process.py
@time: 2018-1-20 19: 05
@license: Apache License
@contact: pegasus.wenjia@foxmail.com
"""
import numpy as np
import tensorflow as tf
import pickle

DIR = '/home/prince/PycharmProjects/lc222seq2seq/data/dataset-cornell-length10-filter1-vocabSize40000.pkl'


def data(DIR, epochs, batch_size):
    with open(DIR, 'rb') as d:
        data = pickle.load(d)

    word2id = data['word2id']
    id2word = data['id2word']
    trainingSamples = data['trainingSamples']

    questions, answers = zip(*trainingSamples)

    qlength = [x for x in map(len, questions)]
    alength = [x for x in map(len, answers)]

    def reverse(x): return x[::-1]

    questions = [x for x in map(reverse, questions)]  # reverse questions
    answers = list(answers)
    max_length = 10

    for i in range(len(qlength)):
        questions[i] = [0] * (10 - qlength[i]) + questions[i]
        answers[i] = answers[i] + [0] * (10 - alength[i])

    # dtype =np.int32 should not be changed as this type is in according with dtype in seq2seq model
    answers = np.array(answers, dtype=np.int32)
    questions = np.array(questions, dtype=np.int32)
    alength = np.array(alength, dtype=np.int32)
    qlength = np.array(qlength, dtype=np.int32)

    answers = tf.data.Dataset.from_tensor_slices(answers)
    questions = tf.data.Dataset.from_tensor_slices(questions)
    alength = tf.data.Dataset.from_tensor_slices(alength)
    qlength = tf.data.Dataset.from_tensor_slices((qlength))

    data = tf.data.Dataset.zip((questions, answers, qlength, alength))
    data = data.repeat(epochs)
    data = data.shuffle(buffer_size=10000)
    data = data.batch(batch_size=batch_size)
    iterator = data.make_one_shot_iterator()
    questionbatch, answerbatch, qlengthbatch, alengthbatch = iterator.get_next()
    return word2id, id2word, questionbatch, answerbatch, qlengthbatch, alengthbatch


# with tf.python_io.TFRecordWriter('qadataset.tfrecord') as writer:
#     for i in range(len(qlength)):
#
#         example = tf.train.Example(features =  tf.train.Features(
#             feature = {'question': tf.train.Feature(int64_list = tf.train.Int64List(value = questions[i])),
#                        'answer': tf.train.Feature(int64_list = tf.train.Int64List(value = answers[i])),
#                        'questionlength':tf.train.Feature(int64_list = tf.train.Int64List(value = [qlength[i]])),
#                        'answerlength':tf.train.Feature(int64_list = tf.train.Int64List(value = [alength[i]]))}))
#         serialized = example.SerializeToString()
#         writer.write(serialized)
#     print('done')

if __name__ == '__main__':
    word2id, id2word, questionbatch, answerbatch, qlengthbatch, alengthbatch = data(DIR, 30, 128)
    s = tf.Session().run
    c = s(questionbatch)
