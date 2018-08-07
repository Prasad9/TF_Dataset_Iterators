# Demonstration of how to create Dataset with various supporting APIs

import tensorflow as tf
import numpy as np

# Making use of from_tensor_slices
# Assume batch size is 1
dataset1 = tf.data.Dataset.from_tensor_slices(tf.range(10, 15))
# Emits data of 10, 11, 12, 13, 14, (One element at a time)

dataset2 = tf.data.Dataset.from_tensor_slices((tf.range(30, 45, 3), np.arange(60, 70, 2)))
# Emits data of (30, 60), (33, 62), (36, 64), (39, 66), (42, 68)
# Emits one tuple at a time

try:
    dataset3 = tf.data.Dataset.from_tensor_slices((tf.range(10), np.arange(5)))
    # Dataset not possible as zeroth dimenion is different at 10 and 5
except:
    pass


# Making use of from_tensors
dataset4 = tf.data.Dataset.from_tensors(tf.range(10, 15))
# Emits data of [10, 11, 12, 13, 14]
# Holds entire list as one element

dataset5 = tf.data.Dataset.from_tensors((tf.range(30, 45, 3), np.arange(60, 70, 2)))
# Emits data of ([30, 33, 36, 39, 42], [60, 62, 64, 66, 68])
# Holds entire tuple as one element

dataset6 = tf.data.Dataset.from_tensors((tf.range(10), np.arange(5)))
# Possible with from_tensors, regardless of zeroth dimension mismatch of constituent elements.
# Emits data of ([1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4])
# Holds entire tuple as one element


# Making use of from_generators
# Assume batch size is 1
def generator(sequence_type):
    if sequence_type == 1:
        for i in range(5):
            yield 10 + i
    elif sequence_type == 2:
        for i in range(5):
            yield (30 + 3 * i, 60 + 2 * i)
    elif sequence_type == 3:
        for i in range(1, 4):
            yield (i, ['Hi'] * i)

dataset7 = tf.data.Dataset.from_generator(generator, (tf.int32), args = ([1]))
# Emits data of 10, 11, 12, 13, 14, (One element at a time)

dataset8 = tf.data.Dataset.from_generator(generator, (tf.int32, tf.int32), args = ([2]))
# Emits data of (30, 60), (33, 62), (36, 64), (39, 66), (42, 68)
# Emits one tuple at a time

dataset9 = tf.data.Dataset.from_generator(generator, (tf.int32, tf.string), args = ([3]))
# Emits data of (1, ['Hi']), (2, ['Hi', 'Hi']), (3, ['Hi', 'Hi', 'Hi'])
# Emits one tuple at a time