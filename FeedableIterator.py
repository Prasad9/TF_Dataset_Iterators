# Demonstration of making use of Feedable Iterator pipeline

import tensorflow as tf
from tqdm import tqdm
from Data import X_train, y_train, X_val, y_val, X_test, y_test
from Model import Model
import numpy as np

def map_fn(x, y):
    # Do transformations here
    return x, y

epochs = 10
batch_size = 64

tf.reset_default_graph()
placeholder_X = tf.placeholder(tf.float32, shape = [None, 28, 28, 1])
placeholder_y = tf.placeholder(tf.int32, shape = [None])

# Create separate Datasets for training, validation and testing
train_dataset = tf.data.Dataset.from_tensor_slices((placeholder_X, placeholder_y))
train_dataset = train_dataset.batch(batch_size).map(lambda x, y: map_fn(x, y))

val_dataset = tf.data.Dataset.from_tensor_slices((placeholder_X, placeholder_y))
val_dataset = val_dataset.batch(batch_size)

y_test = np.array(y_test, dtype = np.int32)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(batch_size)

# Feedable iterator assigns each iterator a unique string handle it is going to work on 
handle = tf.placeholder(tf.string, shape = [])
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
data_X, data_y = iterator.get_next()
data_y = tf.cast(data_y, tf.int32)
model = Model(data_X, data_y)

# Create Reinitializable iterator for Train and Validation, one shot iterator for Test
train_val_iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
train_iterator = train_val_iterator.make_initializer(train_dataset)
val_iterator = train_val_iterator.make_initializer(val_dataset)
test_iterator = test_dataset.make_one_shot_iterator()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Create string handles for above reinitializable and one shot iterators.
    train_val_string = sess.run(train_val_iterator.string_handle())
    test_string = sess.run(test_iterator.string_handle())

    for epoch_no in range(epochs):
        train_loss, train_accuracy = 0, 0
        val_loss, val_accuracy = 0, 0

        # Start reinitializable's train iterator
        sess.run(train_iterator, feed_dict = {placeholder_X: X_train, placeholder_y: y_train})
        try:
            with tqdm(total = len(y_train)) as pbar:
                while True:
                    # Feed to feedable iterator the string handle of reinitializable iterator
                    _, loss, acc = sess.run([model.optimizer, model.loss, model.accuracy], \
                                                feed_dict = {handle: train_val_string})
                    train_loss += loss
                    train_accuracy += acc
                    pbar.update(batch_size)
        except tf.errors.OutOfRangeError:
            pass
      
        # Start reinitializable's validation iterator
        sess.run(val_iterator, feed_dict = {placeholder_X: X_val, placeholder_y: y_val})
        try:
            while True:
                loss, acc = sess.run([model.loss, model.accuracy], \
                                        feed_dict = {handle: train_val_string})
                val_loss += loss
                val_accuracy += acc
        except tf.errors.OutOfRangeError:
            pass
    
        print('\nEpoch: {}'.format(epoch_no + 1))
        print('Training accuracy: {:.4f}, loss: {:.4f}'.format(train_accuracy / len(y_train), 
                                                                train_loss / len(y_train)))
        print('Val accuaracy: {:.4f}, loss: {:.4f}\n'.format(val_accuracy / len(y_val), 
                                                                val_loss / len(y_val)))
    
    test_loss, test_accuracy = 0, 0
    try:
        while True:
            # Feed to feedable iterator the string handle of one shot iterator
            loss, acc = sess.run([model.loss, model.accuracy], feed_dict = {handle: test_string})
            test_loss += loss
            test_accuracy += acc
    except tf.errors.OutOfRangeError:
        pass

print('\nTest accuracy: {:.4f}, loss: {:.4f}'.format(test_accuracy / len(y_test), test_loss / len(y_test)))