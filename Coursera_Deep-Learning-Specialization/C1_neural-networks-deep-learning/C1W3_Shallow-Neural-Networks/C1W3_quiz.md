QUIZ
==========

> 1. Which of the following are true? (Check all that apply.)<br><br>
_**X is a matrix in which each column is one training example.**_
_**a_4^[2] is the activation output by the 4th neuron of the 2nd layer**_
_**a^[2](12) denotes the activation vector of the 2nd layer for the 12th training example.**_
_**a^[2] denotes the activation vector of the 2nd layer.**_

> 2. The tanh activation is not always better than sigmoid activation function for hidden units because the mean of its output is closer to zero, and so it centers the data, making learning complex for the next layer. True/False?<br><br>
**False**

> 3. Which of these is a correct vectorized implementation of forward propagation for layer ll, where 1 ≤ l ≤ L?<br><br>
**Z^[l] = W^[l]A^l-1] + b^[l]
A^[l] = g^[l](Z^[l])**

> 4. You are building a binary classifier for recognizing cucumbers (y=1) vs. watermelons (y=0). Which one of these activation functions would you recommend using for the output layer?<br><br>
_**sigmoid**_

> 5. Consider the following code:
 A = np.random.randn(4,3)B = np.sum(A, axis = 1, keepdims = True) 
What will be B.shape? (If you’re not sure, feel free to run this in python to find out).<br><br>
_**(4, 1)**_

> 6. Suppose you have built a neural network. You decide to initialize the weights and biases to be zero. Which of the following statements is true?<br><br>
_**Each neuron in the first hidden layer will perform the same computation. So even after multiple iterations of gradient descent each neuron in the layer will be computing the same thing as other neurons.**_

> 7. Logistic regression’s weights w should be initialized randomly rather than to all zeros, because if you initialize to all zeros, then logistic regression will fail to learn a useful decision boundary because it will fail to “break symmetry”, True/False?<br><br>
_**False**_

> 8. You have built a network using the tanh activation for all the hidden units. You initialize the weights to relative large values, using np.random.randn(..,..)*1000. What will happen?<br><br>
_**This will cause the inputs of the tanh to also be very large, thus causing gradients to be close to zero. The optimization algorithm will thus become slow.**_

> 9. Consider the following 1 hidden layer neural network:<br>
(그림)<br>
Which of the following statements are True? (Check all that apply).<br><br>
_**W^[1] will have shape (4, 2)**_<br>
_**b^[1] will have shape (4, 1)**_<br>
_**W^[2] will have shape (1, 4)**_<br>
_**b^[2] will have shape (1, 1)**_

(그림)

> 10. In the same network as the previous question, what are the dimensions of Z^[1] and A^[1]?<br><br>
_**Z^[1] and A^[1] are (4, m)**_