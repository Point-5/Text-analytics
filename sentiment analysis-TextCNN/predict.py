#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import csv
import pandas as pd
from tensorflow.contrib import learn

# Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "D:\\bsdz\\predict\\model_saved\\checkpoints\\", "Checkpoint directory path")
FLAGS = tf.flags.FLAGS

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

# Put your own data here
# D:\bsdz\predict\data
# x_raw = ["a masterpiece four years in the making", "everything is off."]

def test(content):

    x_raw = content

    # Map data into vocabulary
    vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

    # Prediction
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            saver = tf.train.import_meta_graph("D:\\bsdz\\predict\\model_saved\\checkpoints\\model-7000.meta")
            saver.restore(sess, checkpoint_file)
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            batches = batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    # print(all_predictions)
    return all_predictions

all_content = []
all_label = []
all_time = []
with open("D:\\bsdz\\predict\\data\\sorted_new1.csv", "r", encoding='ISO-8859-1') as csvfile:
    index = 0
    data_128 = []
    time_128 = []
    reader = csv.reader(csvfile)
    for line in reader:
        time_128.append(line[0])
        data_128.append(line[1])
        index += 1
        print(index)
        if index == 533:
            label_128 = test(data_128)
            for data in data_128:
                all_content.append(data)
            for label in label_128:
                print(label)
                all_label.append(label)
            for time in time_128:
                all_time.append(time)
            index = 0
            data_128 = []
            time_128 = []

        # content = line[1]
        # label = test(content)
        # label = label[0]
        # print(label)
        # all_content.append(content)
        # all_label.append(label)


sub = pd.DataFrame({'datetime': all_time, 'content': all_content, 'label': all_label})
sub = sub[['datetime', 'content', 'label']]
sub.to_csv('result_new2.csv', header=None, index=False)