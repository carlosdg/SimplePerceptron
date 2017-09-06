class Neuron
{
    // Suppose that the activation function is a sigmoid
    //      s(x) = 1 / (1 + exp(-x))
    // derivedOutputVal_ stores the derivative with respect to x,
    //      derivedOutputVal_ = s'(x) = s(x) * (1 - s(x))
    private derivedOutputVal_: number;
    private weights_: number[] = [];

    // This class Neuron was designed thinking in implementing later a fully connected
    // neural network that is why it distinguish between an input neuron and the rest.
    constructor( numberOfInputs: number,
                 private isInputNeuron_: boolean = false,
                 private outputVal_: number = Math.random()
    ){
        if ( !this.isInputNeuron_ )
            this.derivedOutputVal_ = Math.random();

        // The last weight is the constant summand, the threshold
        for (let i = 0; i <= numberOfInputs; ++i)
            this.weights_.push( Math.random() );
    }


    // Getters
    public getOutput(): number { return this.outputVal_; }
    public getDerivedOutput(): number { return this.derivedOutputVal_; }
    public getWeights(): number[] { return this.weights_; }

    // Sigmoid activation function
    public static activationFunction( input: number ): number{
        return (  1 / (1 + Math.exp(-input))  );
    }
    public static rateOfChangeOfActivationFunction( activatedInput: number ): number{
        return ( (1-activatedInput) * activatedInput );
    }

    // @param inputFeatures: vector of inputs, only the real ones, there is no need
    //                       to add a constant one for the bias, the neuron handles that.
    // @return: This object is return so we can chain methods when using a neuron.
    public feed( inputFeatures: number[] ): Neuron{
        // If this is an input neuron we don't have to do anything
        if (this.isInputNeuron_)
            return this;

        // Checking that the dot product can be done.
        // Remember that this.weights_ has an extra element
        // which would be the constant summand, the one
        // not multiplied by inputs.
        let length = inputFeatures.length;
        if ( length !==  (this.weights_.length - 1) )
            throw Error( "Cannot perform dot product. Vector sizes don't match" );

        // Weighted sum
        // Instead of initializing the sum at 0, we initialize
        // it directly by the value of the bias.
        this.outputVal_ = this.weights_[length];
        for (let i = 0; i < length; ++i){
            this.outputVal_ += inputFeatures[i] * this.weights_[i];
        }

        // Applying activation function
        this.outputVal_ = Neuron.activationFunction( this.outputVal_ );

        // Storing derivative
        this.derivedOutputVal_ = Neuron.rateOfChangeOfActivationFunction( this.outputVal_ );

        return this;
    }

}

class SimplePerceptronClassifier
{
    private perceptron_: Neuron;

    // In training instead of updating the weights until all predictions matches the real outputs,
    // we limit the training by a number of iterations. Because of simplicity and I think that
    // training until the function fits perfectly the data could be very costly and because of the overfit risk
    constructor( private learningRate_: number,
                 private numIterationsGD_: number){}

    // @return: This is returned so user can chain methods
    // @param X: Inputs as rows, features as columns
    // @param Y: Labels associated with each input, note that each input can be associated
    //           only with one value because a simple perceptron only uses a single neuron, not a network
    public train( X: number[][], Y: number[], numFeatures = X[0].length ): SimplePerceptronClassifier{

        this.perceptron_ = new Neuron( numFeatures );

        for (let i = 0, numIts = this.numIterationsGD_; i < numIts; ++i){
            // Iterate through the input data
            for (let j = 0, numInputs = X.length; j < numInputs; ++j){

                // Calculate prediction for the current input
                let yPred: number = this.perceptron_.feed(X[j]).getOutput();
                let yPredDerivative: number = this.perceptron_.getDerivedOutput();

                // Get weights
                let weights: number[] = this.perceptron_.getWeights();

                // Error derivative function respect to "weightChanging"
                // The error function is [ 1/2 * (Ypred - Yreal)^2 ]
                // So the derivative with respect to some weight is
                // [ (Ypred - Yreal) * DERIVATIVE_OF (Ypred) WITH_RESPECT_TO (some weight)]
                let calcErrorDerivative = function( weightChanging ){
                    if (weightChanging < numFeatures)
                    // Applying chain rule we have that the derivative of Ypred is the derivative
                    // of it with respect to its input. That times the derivative of its inputs
                    // with respect to the weightChanging
                        return ( (yPred - Y[j]) * yPredDerivative * X[j][weightChanging] );
                    else
                        // In case of the bias term, the derivative of Ypred input's with respect
                        // to the bias, is 1.
                        return ( (yPred - Y[j]) * yPredDerivative );
                }

                // Update weights. Because all variables needed for this are already calculated,
                // the update of one weight doesn't depend on other weights. Otherwise we would have
                // a problem in which the update depends on other weights and those could be already
                // updated or not, in that case we would have to use an auxiliar vector of weights
                // so we can update all weights at once.
                for (let w = 0, numWeights = weights.length; w < numWeights; ++w){
                    weights[w] -= (this.learningRate_ * calcErrorDerivative(w));
                }

            }
        }

        return this;
    }

    public predict ( X: number[] ): number{
        return (this.perceptron_.feed( X ).getOutput());
    }

}


// Create sample data
let X: number[][] = [],
    Y: number[]   = [];

for (var i = -100; i < 100; ++i){
    X.push([i,i]); Y.push(1);
    X.push([10+i,i]); Y.push(0);

    // Logic OR sample data
    // X.push([0,0]); Y.push(0);
    // X.push([1,0]); Y.push(1);
    // X.push([0,1]); Y.push(1);
    // X.push([1,1]); Y.push(1);
}


console.log(X, Y);

let perceptron = new SimplePerceptronClassifier( 0.01, 10000 );
perceptron.train(X, Y);
console.log(perceptron);
