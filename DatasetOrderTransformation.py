# Demonstration of Dataset transformation operations highlighting ordering of transformations

import tensorflow as tf


# Ordering #1
dataset1 = tf.data.Dataset.from_tensor_slices(tf.range(10))
# Dataset: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

dataset1 = dataset1.batch(4)
# Dataset: [0, 1, 2, 3], [4, 5, 6, 7], [8, 9]

dataset1 = dataset1.repeat(2)
# Dataset: [0, 1, 2, 3], [4, 5, 6, 7], [8, 9], [0, 1, 2, 3], [4, 5, 6, 7], [8, 9]
# Notice a 2 element batch in between

dataset1 = dataset1.shuffle(4)
# Shuffles at batch level.
# Dataset: [0, 1, 2, 3], [4, 5, 6, 7], [8, 9], [8, 9], [0, 1, 2, 3], [4, 5, 6, 7]

# Ordering #2
dataset2 = tf.data.Dataset.from_tensor_slices(tf.range(10))
# Dataset: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

dataset2 = dataset2.shuffle(4)
# Dataset: [3, 1, 0, 4, 5, 8, 6, 9, 7, 2]

dataset2 = dataset2.repeat(2)
# Dataset: [3, 1, 0, 4, 5, 8, 6, 9, 7, 2, 3, 1, 0, 4, 5, 8, 6, 9, 7, 2]
 
dataset2 = dataset2.batch(4)
# Dataset: [3, 1, 0, 4], [5, 8, 6, 9], [7, 2, 3, 1], [0, 4, 5, 8], [6, 9, 7, 2]
