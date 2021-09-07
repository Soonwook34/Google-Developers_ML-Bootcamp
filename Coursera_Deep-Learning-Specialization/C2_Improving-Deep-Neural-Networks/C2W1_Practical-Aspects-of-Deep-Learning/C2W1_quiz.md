QUIZ
==========

> 1. If you have 10,000,000 examples, how would you split the train/dev/test set?<br><br>
_**98% train . 1% dev . 1% test**_

> 2. The dev and test set should:<br><br>
_**Come from the same distribution**_

> 3. If your Neural Network model seems to have high variance, what of the following would be promising things to try?<br><br>
_**Add regularization**_
_**Get more training data**_

> 4. You are working on an automated check-out kiosk for a supermarket, and are building a classifier for apples, bananas and oranges. Suppose your classifier obtains a training set error of 0.5%, and a dev set error of 7%. Which of the following are promising things to try to improve your classifier? (Check all that apply.)<br><br>
_**Increase the regularization parameter lambda**_
_**Get more training data**_

> 5. What is weight decay?<br><br>
_**A regularization technique (such as L2 regularization) that results in gradient descent shrinking the weights on every iteration.**_

> 6. What happens when you increase the regularization hyperparameter lambda?<br><br>
_**Weights are pushed toward becoming smaller (closer to 0)**_

> 7. With the inverted dropout technique, at test time:<br><br>
_**You do not apply dropout (do not randomly eliminate units), but keep the 1/keep_prob factor in the calculations used in training.**_

> 8. Increasing the parameter keep_prob from (say) 0.5 to 0.6 will likely cause the following: (Check the two that apply)<br><br>
_**Reducing the regularization effect**_
_**Causing the neural network to end up with a lower training set error**_

> 9. Which of these techniques are useful for reducing variance (reducing overfitting)? (Check all that apply.)<br><br>
_**Data augmentation**_
_**L2 regularization**_
_**Dropout**_

> 10. Why do we normalize the inputs $x$?<br><br>
_**It makes the cost function faster to optimize**_