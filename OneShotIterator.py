# Demonstration of making use of One Shot Iterator pipeline

import tensorflow as tf
from tqdm import tqdm
from Data import X_train, y_train
from Model import Model

epochs = 10
batch_size = 64
iterations = len(y_train) * epochs

tf.reset_default_graph()

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# Generate the complete Dataset required in the pipeline
dataset = dataset.repeat(epochs).batch(batch_size)
iterator = dataset.make_one_shot_iterator()

data_X, data_y = iterator.get_next()
data_y = tf.cast(data_y, tf.int32)
model = Model(data_X, data_y)

with tf.Session() as sess, tqdm(total = iterations) as pbar:
    sess.run(tf.global_variables_initializer())

    tot_accuracy = 0
    try:
        while True:
            accuracy, _ = sess.run([model.accuracy, model.optimizer])
            tot_accuracy += accuracy
            pbar.update(batch_size)
    except tf.errors.OutOfRangeError:
        pass

print('\nAverage training accuracy: {:.4f}'.format(tot_accuracy / iterations))