# Demonstration of Initializable Iterator pipeline

import tensorflow as tf
from tqdm import tqdm
from Data import X_train, y_train, X_val, y_val
from Model import Model

epochs = 10
batch_size = 64

tf.reset_default_graph()
placeholder_X = tf.placeholder(tf.float32, [None, 28, 28, 1])
placeholder_y = tf.placeholder(tf.int32, [None])

dataset = tf.data.Dataset.from_tensor_slices((placeholder_X, placeholder_y))
dataset = dataset.batch(batch_size)
iterator = dataset.make_initializable_iterator()

data_X, data_y = iterator.get_next()
data_y = tf.cast(data_y, tf.int32)
model = Model(data_X, data_y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch_no in range(epochs):
        train_loss, train_accuracy = 0, 0
        val_loss, val_accuracy = 0, 0

        # Initialize iterator with training Data
        sess.run(iterator.initializer, feed_dict = {placeholder_X: X_train, placeholder_y: y_train})
        try:
            with tqdm(total = len(y_train)) as pbar:
                while True:
                    _, loss, acc = sess.run([model.optimizer, model.loss, model.accuracy])
                    train_loss += loss 
                    train_accuracy += acc
                    pbar.update(batch_size)
        except tf.errors.OutOfRangeError:
            pass
    
        # Initialize iterator with validation Data
        sess.run(iterator.initializer, feed_dict = {placeholder_X: X_val, placeholder_y: y_val})
        try:
            while True:
                loss, acc = sess.run([model.loss, model.accuracy])
                val_loss += loss 
                val_accuracy += acc
        except tf.errors.OutOfRangeError:
            pass
    
        print('\nEpoch No: {}'.format(epoch_no + 1))
        print('Train accuracy = {:.4f}, loss = {:.4f}'.format(train_accuracy / len(y_train), 
                                                        train_loss / len(y_train)))
        print('Val accuracy = {:.4f}, loss = {:.4f}'.format(val_accuracy / len(y_val), 
                                                        val_loss / len(y_val)))