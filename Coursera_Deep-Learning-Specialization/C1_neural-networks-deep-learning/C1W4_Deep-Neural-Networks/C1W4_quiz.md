QUIZ
==========

> 1. What is the "cache" used for in our implementation of forward propagation and backward propagation?<br><br>
_**We use it to pass variables computed during forward propagation to the corresponding backward propagation step. It contains useful values for backward propagation to compute derivatives.**_

> 2. Among the following, which ones are "hyperparameters"? (Check all that apply.)<br><br>
_**number of iterations**_
_**learning rate α**_
_**number of layers L in the neural network**_
_**size of the hidden layers n^[l]**_

> 3. Which of the following statements is true?<br><br>
_**The deeper layers of a neural network are typically computing more complex features of the input than the earlier layers.**_

> 4. Vectorization allows you to compute forward propagation in an L-layer neural network without an explicit for-loop (or any other explicit iterative loop) over the layers l=1, 2, …,L. True/False?<br><br>
_**False**_

> 5. Assume we store the values for n^[l] in an array called layer_dims, as follows: layer_dims = [n_x, 4,3,2,1]. So layer 1 has four hidden units, layer 2 has 3 hidden units and so on. Which of the following for-loops will allow you to initialize the parameters for the model?<br><br>
```python
for i in range(1, len(layer_dims)):
	parameter['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.01
	parameter['b' + str(i)] = np.random.randn(layer_dims[i], 1) * 0.01
```

> 6. Consider the following neural network.<br>
(그림)<br>
How many layers does this network have?<br><br>
_**The number of layers L is 4. The number of hidden layers is 3.**_

> 7. During forward propagation, in the forward function for a layer l you need to know what is the activation function in a layer (Sigmoid, tanh, ReLU, etc.). During backpropagation, the corresponding backward function also needs to know what is the activation function for layer l, since the gradient depends on it. True/False?<br><br>
_**True**_

> 8. There are certain functions with the following properties:<br>
(i) To compute the function using a shallow network circuit, you will need a large network (where we measure size by the number of logic gates in the network), but (ii) To compute it using a deep network circuit, you need only an exponentially smaller network. True/False?<br><br>
_**True**_

> 9. Consider the following 2 hidden layer neural network:<br>
(그림)<br>
Which of the following statements are True? (Check all that apply).<br><br>
_**W^[1] will have shape (4, 4)**_
_**W^[2] will have shape (3, 4)**_
_**W^[3] will have shape (1, 3)**_
_**b^[1] will have shape (4, 1)**_
_**b^[2] will have shape (3, 1)**_
_**b^[3] will have shape (1, 1)**_

> 10. Whereas the previous question used a specific network, in the general case what is the dimension of W^{[l]}, the weight matrix associated with layer l?<br><br>
_**W^[l] has shape (n^[l], n^[l-1])**_