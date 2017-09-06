var Neuron = (function () {
    // This class Neuron was designed thinking in implementing later a fully connected
    // neural network that is why it distinguish between an input neuron and the rest.
    function Neuron(numberOfInputs, isInputNeuron_, outputVal_) {
        if (isInputNeuron_ === void 0) { isInputNeuron_ = false; }
        if (outputVal_ === void 0) { outputVal_ = Math.random(); }
        this.isInputNeuron_ = isInputNeuron_;
        this.outputVal_ = outputVal_;
        this.weights_ = [];
        if (!this.isInputNeuron_)
            this.derivedOutputVal_ = Math.random();
        // The last weight is the constant summand, the threshold
        for (var i_1 = 0; i_1 <= numberOfInputs; ++i_1)
            this.weights_.push(Math.random());
    }
    // Getters
    Neuron.prototype.getOutput = function () { return this.outputVal_; };
    Neuron.prototype.getDerivedOutput = function () { return this.derivedOutputVal_; };
    Neuron.prototype.getWeights = function () { return this.weights_; };
    // Sigmoid activation function
    Neuron.activationFunction = function (input) {
        return (1 / (1 + Math.exp(-input)));
    };
    Neuron.rateOfChangeOfActivationFunction = function (activatedInput) {
        return ((1 - activatedInput) * activatedInput);
    };
    // @param inputFeatures: vector of inputs, only the real ones, there is no need
    //                       to add a constant one for the bias, the neuron handles that.
    // @return: This object is return so we can chain methods when using a neuron.
    Neuron.prototype.feed = function (inputFeatures) {
        // If this is an input neuron we don't have to do anything
        if (this.isInputNeuron_)
            return this;
        // Checking that the dot product can be done.
        // Remember that this.weights_ has an extra element
        // which would be the constant summand, the one
        // not multiplied by inputs.
        var length = inputFeatures.length;
        if (length !== (this.weights_.length - 1))
            throw Error("Cannot perform dot product. Vector sizes don't match");
        // Weighted sum
        // Instead of initializing the sum at 0, we initialize
        // it directly by the value of the bias.
        this.outputVal_ = this.weights_[length];
        for (var i_2 = 0; i_2 < length; ++i_2) {
            this.outputVal_ += inputFeatures[i_2] * this.weights_[i_2];
        }
        // Applying activation function
        this.outputVal_ = Neuron.activationFunction(this.outputVal_);
        // Storing derivative
        this.derivedOutputVal_ = Neuron.rateOfChangeOfActivationFunction(this.outputVal_);
        return this;
    };
    return Neuron;
}());
var SimplePerceptronClassifier = (function () {
    // In training instead of updating the weights until all predictions matches the real outputs,
    // we limit the training by a number of iterations. Because of simplicity and I think that
    // training until the function fits perfectly the data could be very costly and because of the overfit risk
    function SimplePerceptronClassifier(learningRate_, numIterationsGD_) {
        this.learningRate_ = learningRate_;
        this.numIterationsGD_ = numIterationsGD_;
    }
    // @return: This is returned so user can chain methods
    // @param X: Inputs as rows, features as columns
    // @param Y: Labels associated with each input, note that each input can be associated
    //           only with one value because a simple perceptron only uses a single neuron, not a network
    SimplePerceptronClassifier.prototype.train = function (X, Y, numFeatures) {
        if (numFeatures === void 0) { numFeatures = X[0].length; }
        this.perceptron_ = new Neuron(numFeatures);
        for (var i_3 = 0, numIts = this.numIterationsGD_; i_3 < numIts; ++i_3) {
            var _loop_1 = function (j, numInputs) {
                // Calculate prediction for the current input
                var yPred = this_1.perceptron_.feed(X[j]).getOutput();
                var yPredDerivative = this_1.perceptron_.getDerivedOutput();
                // Get weights
                var weights = this_1.perceptron_.getWeights();
                // Error derivative function respect to "weightChanging"
                // The error function is [ 1/2 * (Ypred - Yreal)^2 ]
                // So the derivative with respect to some weight is
                // [ (Ypred - Yreal) * DERIVATIVE_OF (Ypred) WITH_RESPECT_TO (some weight)]
                var calcErrorDerivative = function (weightChanging) {
                    if (weightChanging < numFeatures)
                        // Applying chain rule we have that the derivative of Ypred is the derivative
                        // of it with respect to its input. That times the derivative of its inputs
                        // with respect to the weightChanging
                        return ((yPred - Y[j]) * yPredDerivative * X[j][weightChanging]);
                    else
                        // In case of the bias term, the derivative of Ypred input's with respect
                        // to the bias, is 1.
                        return ((yPred - Y[j]) * yPredDerivative);
                };
                // Update weights. Because all variables needed for this are already calculated,
                // the update of one weight doesn't depend on other weights. Otherwise we would have
                // a problem in which the update depends on other weights and those could be already
                // updated or not, in that case we would have to use an auxiliar vector of weights
                // so we can update all weights at once.
                for (var w = 0, numWeights = weights.length; w < numWeights; ++w) {
                    weights[w] -= (this_1.learningRate_ * calcErrorDerivative(w));
                }
            };
            var this_1 = this;
            // Iterate through the input data
            for (var j = 0, numInputs = X.length; j < numInputs; ++j) {
                _loop_1(j, numInputs);
            }
        }
        return this;
    };
    SimplePerceptronClassifier.prototype.predict = function (X) {
        return (this.perceptron_.feed(X).getOutput());
    };
    return SimplePerceptronClassifier;
}());
// Create sample data
var X = [], Y = [];
for (var i = -100; i < 100; ++i) {
    X.push([i, i]);
    Y.push(1);
    X.push([10 + i, i]);
    Y.push(0);
    // Logic OR sample data
    // X.push([0,0]); Y.push(0);
    // X.push([1,0]); Y.push(1);
    // X.push([0,1]); Y.push(1);
    // X.push([1,1]); Y.push(1);
}
console.log(X, Y);
var perceptron = new SimplePerceptronClassifier(0.01, 10000);
perceptron.train(X, Y);
console.log(perceptron);
