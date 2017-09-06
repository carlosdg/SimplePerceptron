# Simple Perceptron
<h2> Brief description: </h2>
<p>
    I want to implement a neural network from scratch. I am not trying to reinvent the wheel,
    I want to see if I really understand neural networks by implementing one. Though before
    implementing a network I started with this simple perceptron as an introduction.
</p>


<h2> Author: </h2>
<p>	Carlos Domínguez García </p>


<h2> Tested on: </h2>
<ul>
  <li>	Google Chrome - 60.0.3112.113  </li>
</ul>


<h2> About perceptrons </h2>
<p>
    There is plenty of information about perceptrons and neural networks on the web, so I won't
    go very deep explaining what it is.
</p>
<p>
    In a very abstract level a perceptron is just a mathematical model that tries to describe
    how real single neuron behaves. In this model a neuron receives various inputs (the features
    of an element), does some operation with those (a weighted sum) and if the result of that
    pass some threshold, it outputs the final result. This last part is the activation function,
    it seems that real neurons doesn't always propagates an output, only when the weighted sum
    pass some threshold. So the activation function in the model could be something like:    
</p>
<p>
    f(x) = { 0 if x < some_threshold;  1 if x >= some_threshold; }
</p>
<p>
    However this isn't convenient because the function is not continuous. And the derivative is always
    zero where it is continuous. So we cannot use gradient descent for updating the weights to minimize
    some error, (that error function would depend on the output of the perceptron that depends on that function).
    So the alternative I chose is the sigmoid function s(x) = 1 / (1 + exp(-x)). Although there are a lot
    of functions that can be used as activation functions, I chose this one because it is very similar
    to the previous function but it is continuous. Also it is very simple and the derivative is very easy
    to compute s'(x) = s(x) * (1 - s(x)).
</p>
<p>
    However, there is a problem left, the threshold, if you take a look
    at the sigmoid, it looks like f(x) but, in the sigmoid the threshold is always set to zero.
    For solving this, we can imagine that to f(x), instead of passing (x) and comparing
    to (some_threshold). We could pass (x - some_threshold) and compare it with zero. This way the threshold variable
    is "eliminated" from the function by adding a bias to the input. That is the reason why in the code we have weights and a constant weight
    that is not multiplied with the inputs but summed directly. Another point of view for the necessity of this
    bias term is that if we don't add it, we would never have a complete polynomyal, so it would be harder to
    to approximate the function that we want.
</p>
