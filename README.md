# Demonstration of TensorFlow's Dataset and Iterators 
This repository is an illustration of how to use Tensorflow's Datasets and Iterators with MNIST image training pipelines.

You can read the full detailed explanation about the code used in this repository in [my Medium post](https://medium.com/ymedialabs-innovation/how-to-use-dataset-and-iterators-in-tensorflow-with-code-samples-3bb98b6b74ab).

You may go through each of the file for the specific code samples listed below.

### Dataset Creation from Numpy/Tensor example
Code showing how to create a Dataset from Numpy/Tensor. No output to be displayed in this sample.
```
python DatasetExample.py
```

### Dataset Transformation example
Code showing how to perform transformations on Dataset. No output to be displayed in this sample.
```
python DatasetTransformation.py
```

### Impact of ordering of Dataset Transformation example
Code showing how the ordering of transformation on Dataset can change the manner of data coming in Dataset.
```
python DatasetOrderTransformation.py
```

### One Shot Iterator example
Code showing training of MNIST digit images with LeNet-5 model using One Shot Iterator.
```
python OneShotIterator.py
```

### Initializable Iterator example
Code showing training and validation of MNIST digit images with LeNet-5 model using Initializable Iterator.
```
python InitializableIterator.py
```

### Reinitializable Iterator example
Code showing training and validation of MNIST digit images with LeNet-5 model using Reinitializable Iterator.
```
python ReinitializableIterator.py
```

### Feedable Iterator example
Code showing training, validation and testing of MNIST digit images with LeNet-5 model using Feedable Iterator.
```
python FeedableIterator.py
```

If you wish to see the running code completely, you may also check it out in [Google Colab notebook](https://colab.research.google.com/drive/1FHS7lLJdX-l858sGzYQ0Nq3rjS7-_JL2).
