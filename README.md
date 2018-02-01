# Oil Spill Challenge
The code implements Backpropagation on a feedforward neural network using Stochastic Gradient Descent for classification on MNIST dataset. Initial experiments were done with vanilla SGD, but the process was optimized by using following 'tricks of the trade' from Yann LeCun's paper on Efficient Backpropagation from 1998.
* Input normalization
* Use of Nesterov Momentum Accelerated Gradient
* Efficient Weights initialization

Inputs were **normalized** by dividing them by 127.5 and subtracting 1. This was done so that inputs are in the range -1 to 1, centered around 0. Also, the input data was split in 80:20 ratio into training/validation dataset.

**Nesterov Momentum Accelerated Gradient** is an improvement over simple momentum, in the sense that it makes use of 'Look-Ahead Weights' and calculates gradient using them to update the weights. This speeds up convergence by a huge margin, as was observed in the trials conducted for this task.

Weights were initialized randomly from **Normal distribution** with mean 0 and variance equal to 1/(*fan-in*), where *fan-in* is the number of inputs to a neuron.

The architecture used has 2 hidden layers, 100 neurons in the first hidden layer and 50 neurons in second hidden layer. Both hidden layers have *ReLU* activation functions, as specified in the task. The output layer has 10 neurons, each corresponding to 1 digit in MNIST. Softmax activation has been used for output layer, and a cross-entropy loss function has been used to observe model's behavior.

Various experiments were conducted, changing the architecture of network - number of neurons in both the hidden layers. Early stopping method was used to determine the epoch at which validation accuracy is maximum, and those weights were used to test the model.

## Libraries used
* NumPy
* mnist (to import MNIST data)

## Implementation
The value of learning rate is taken as 0.01, which was chosen by hit-and-trial. And the number of epochs is set to 800. Also, since Nesterov Momentum method is used, the value of momentum factor (alpha) is set to 0.9. This is just an empirical value which is used as it is. The current batch size being used is 256. To run the code, follow these steps:

1. Make sure the MNIST data files ('train-images-idx3-ubyte','train-labels-idx1-ubyte','t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte') are present in the same directory as the code.

2. Make sure that library 'mnist' is installed on the system. Use the command
```
pip install python-mnist
```
to install the library

3. To run the code, use following command
```
python backprop_nn.py
```

## Results
For the values of hyperparameters mentioned above, and for the architecture mentioned above, the results obtained were:

Early Stopping Epoch : 553

Training Accuracy : 93.9958%

Testing Accuracy : 92.83%

Validation Accuracy : 92.9166%

Note: This is nowhere close to state-of-the-art methods, but the code implementation is correct, with additional optimizations implemented that mainly affect the training speed. Proper tuning of the model was not performed due to huge computation time and limited resources with me, but with better architecture, greater accuracies can be achieved. One such method is to use 2 hidden layers with 64 neurons each, and with *tanh* activation function.

## Technicalities and Brief Code Description
The code currently doesn't support inputs from command prompt, since architecture was specified initially. But the code is modular enough that by changing arguments to the `FeedForward` function invoked in `main`, one can change the number of hidden layers and neurons, and activation function. Following are the arguments to Feedforward function and their roles:

* **X\_train\_** : This takes in the entire training input data (split is performed inside) as a NumPy matrix.


* **Y\_train\_** :  This takes in the output labels for training data, in one-hot encoded form as a NumPy matrix.


* **X_test** :  This takes in the entire testin input data as a NumPy matrix.


* **Y_test** :  This takes in the output labels for testing data, in one-hot encoded form as a NumPy matrix.


* **hidden** :  This takes in a non-empty list as argument, where length of list specifies number of hidden layers, and each element specifies number of neurons in that layer.


* **activations** :  This takes in a non-empty list  as argument, where (length of list - 1) specifies number of hidden layers, and each element specifies activation function for that layer. Last element of the list specifies activation function for output layer. Currently that value is fixed as *softmax*.


* **activations_** :  This takes in a non-empty list  as argument, where length of list specifies number of hidden layers, and each element specifies derivative of activation function for that layer. This is essentially a duplicate of the *activations* argument, in the sense that for a hidden layer, if the passed function is *ReLu*, the corresponding argument in this would be *ReLu_*.


* **eta_** :  This is the constant learning rate for all the layers. Default value is 0.1.


* **batch_size** :  This is the batch size for stochastic gradient descent. Default value is 128.


* **epochs** :  This is the number of epochs for which the model is to train. Default value is 150.


* **holdoutRatio** :  This is the percent of training data (specified in decimals) that is to be kept as validation data. Default value is 0.2.


* **momentum** :  This is the momentum factor (alpha) to be used in case a momentum gradient descent method is decided to be implemented. Default value is 0 i.e. no momentum.
