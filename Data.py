from tensorflow.examples.tutorials.mnist import input_data

# Extracting MNIST data
mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train = mnist.train.images, mnist.train.labels
X_val, y_val     = mnist.validation.images, mnist.validation.labels
X_test, y_test   = mnist.test.images, mnist.test.labels