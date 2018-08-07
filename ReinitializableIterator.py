# Demonstration of Reinitializable Iterator pipeline

import tensorflow as tf
from tqdm import tqdm
from Data import X_train, y_train, X_val, y_val
from Model import Model

def map_fn(x, y):
    # Do transformations here
    return x, y

epochs = 10
batch_size = 64

tf.reset_default_graph()
placeholder_X = tf.placeholder(tf.float32, shape = [None, 28, 28, 1])
placeholder_y = tf.placeholder(tf.int32, shape = [None])

# Create separate Datasets for training and validation
train_dataset = tf.data.Dataset.from_tensor_slices((placeholder_X, placeholder_y))
train_dataset = train_dataset.batch(batch_size).map(lambda x, y: map_fn(x, y))
val_dataset = tf.data.Dataset.from_tensor_slices((placeholder_X, placeholder_y))
val_dataset = val_dataset.batch(batch_size)

# Iterator has to have same output types across all Datasets to be used
iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
data_X, data_y = iterator.get_next()
data_y = tf.cast(data_y, tf.int32)
model = Model(data_X, data_y)

# Initialize with required Datasets
train_iterator = iterator.make_initializer(train_dataset)
val_iterator = iterator.make_initializer(val_dataset)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_no in range(epochs):
        train_loss, train_accuracy = 0, 0
        val_loss, val_accuracy = 0, 0

        # Start train iterator
        sess.run(train_iterator, feed_dict = {placeholder_X: X_train, placeholder_y: y_train})
        try:
            with tqdm(total = len(y_train)) as pbar:
                while True:
                    _, acc, loss = sess.run([model.optimizer, model.accuracy, model.loss])
                    train_loss += loss
                    train_accuracy += acc
                    pbar.update(batch_size)
        except tf.errors.OutOfRangeError:
            pass

        # Start validation iterator
        sess.run(val_iterator, feed_dict = {placeholder_X: X_val, placeholder_y: y_val})
        try:
            while True:
                acc, loss = sess.run([model.accuracy, model.loss])
                val_loss += loss
                val_accuracy += acc
        except tf.errors.OutOfRangeError:
            pass

        print('\nEpoch: {}'.format(epoch_no + 1))
        print('Train accuracy: {:.4f}, loss: {:.4f}'.format(train_accuracy / len(y_train),
                                                             train_loss / len(y_train)))
        print('Val accuracy: {:.4f}, loss: {:.4f}\n'.format(val_accuracy / len(y_val), 
                                                            val_loss / len(y_val)))