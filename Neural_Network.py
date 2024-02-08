from matplotlib import cm
import numpy as np
import nnfs
import math
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data
from nnfs.datasets import sine_data
nnfs.init()
import matplotlib.pyplot as plt
import os
import urllib
import urllib.request
from zipfile import ZipFile
import cv2
import pickle
import copy
import time

class Data_Dict():
    def __init__(self, labels):
        self.labels = labels
        
# Load dataset function
def load_mnist_dataset(dataset, path):
    # Scan directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # Create empty lists for data / label outputs
    X = []
    y = []
    images = []

    # For each label in folder
    for label in labels: 
        # For each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            # Append image and label to lists
            X.append(image)
            y.append(label)
            
    return np.array(X), np.array(y).astype('uint8')

def create_mnist_dataset(path):
    
    # Load training/validation data separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)
    
    return X, y, X_test, y_test

# Hidden trainable layer

class Layer_Dense:
    
    def __init__(self, n_inputs, n_neurons, lambda_l1w=0, lambda_l1b=0, lambda_l2w=0, lambda_l2b=0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # Shaping weights matrix negates need to transpose
        self.biases = np.zeros((1, n_neurons)) # Takes shape tuple ((r,c))
        # Lambda values are used in the concept of regularization and represent the penalty for large weights/biases
        # Arbitrarily large weights/biases indicate networks memorizing their training data
        # Memorization is bad because the model lacks the ability to generalize and will be less accurate on unseen data

        # All lambdas are multiplied with the calculated "penalty" and thus represent the weight of penalization
        # (Lambda = 0 means no penalty for large weights/biases)

        # L1 is a directly linear penalty which sums the absolute values of weights/biases
        self.lambda_l1w = lambda_l1w
        self.lambda_l1b = lambda_l1b
        # L2 is non-linear and sums the squares of weights/biases
        # L2 penalizes arbitrarily large values more harshly than L1 compared to small values
        self.lambda_l2w = lambda_l2w
        self.lambda_l2b = lambda_l2b
        
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases    # Outputs for next neural layer
        
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.lambda_l1w > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.lambda_l1w * dL1
            
        if self.lambda_l2w > 0:
            self.dweights += 2 * self.lambda_l2w * self.weights
        
        if self.lambda_l1b > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.lambda_l1b * dL1
            
        if self.lambda_l2b > 0:
            self.dbiases += 2 * self.lambda_l2b * self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        
    def get_parameters(self):
        return self.weights, self.biases
    
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases
        
# Activation functions

class Activation_ReLU: # Rectified Linear (Unit) Activation Function
    
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs) # Restrict all values to positive -> Negatives = 0
        
    def backward(self, dvalues):
        # Dvalues will be modified so copy first
        self.dinputs = dvalues.copy()
        # Zero gradient where negative inputs become zero
        self.dinputs[self.inputs <= 0] = 0
        
    def predictions(self, outputs):
        return outputs
        
class Activation_Softmax: # Restricts scale of values to be [0,1]
    
    def forward(self, inputs, training):
        self.inputs = inputs
        # Calculate unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # Subtract max from all values and calculate exp(values-max)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) # Normalize probabilities
        self.output = probabilities
        
    def backward(self, dvalues):
        # dvalues are the gradients we're receiving
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian of output
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient, fill array of sample gradients
            self.dinputs[index] = np.dot(jacobian,single_dvalues)
            
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)
   
class Activation_Linear:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs
        
    def backward(self, dvalues):
        # Derivative is 1
        self.dinputs = dvalues.copy()
        
    def predictions(self, outputs):
        return outputs # Returns inputs

class Activation_Softmax_Loss_CategoricalCrossEntropy():
    
    def backward(self, dvalues, y_true):
        
        samples = len(dvalues)
        # If labels are one-hot, convert to discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1) # argmax returns indices of max values in each row (axis = 1)
            
        # Copy to modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples    

class Activation_Sigmoid:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
        
    def predictions(self, outputs):
        return (outputs > 0.5) * 1
    
class Layer_Dropout:
    
    def __init__(self, rate):
        # Rate represents the % of neurons kept
        self.rate = 1 - rate
        
    def forward(self, inputs, training):
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Apply mask to output
        self.output = inputs * self.binary_mask
        
    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask

# Loss functions

class Loss:
    
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization=False):
        
        # Calculate sample loss
        sample_losses = self.forward(output, y)
        
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        
        # Accumulate sum across samples and epochs
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        # If only calculating data loss, return it
        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()
    
    def regularization_loss(self):

        # 0 by default
        regularization_loss = 0
        
        for layer in self.trainable_layers:
            # L1 regularization, weights
            # Calculate only if lambda > 0 (Else there is no linear penalty for large weights)
            if layer.lambda_l1w > 0:
                regularization_loss += layer.lambda_l1w * np.sum(np.abs(layer.weights))
            
            # L1 regularization, biases
            if layer.lambda_l1b > 0:
                regularization_loss += layer.lambda_l1b * np.sum(np.abs(layer.biases))
            
            # L2 regularization, weights
            if layer.lambda_l2w > 0:
                regularization_loss += layer.lambda_l2w * np.sum(layer.weights * layer.weights)
            
            # L2 regularization, biases
            if layer.lambda_l2b > 0:
                regularization_loss += layer.lambda_l2b * np.sum(layer.biases * layer.biases)
            
        return regularization_loss # Parent loss class
    
    def calculate_accumulated(self, *, include_regularization=False):
        
        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count
        
        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()
    
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
class Loss_Categorical_Cross_Entropy(Loss):
    
    # y_pred is neural network prediction values, y_true is target training values
    def forward(self, y_pred, y_true):   
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) # Set bounds to avoid log(0) = inf loss

        # Formats can vary: [1,0] (2 class values, 1 & 0) or [[0,1], [1,0]] (2 One-hot values, also 1 & 0)
        if(len(y_true.shape) == 1): # Scalar values have been passed (Format 1)
            correct_confidences = y_pred_clipped[range(samples), y_true] # Separate outputs for ONLY the correct training values 
            
        elif(len(y_true.shape) == 2): # One-hot values passed
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1) # Separate outputs for ONLY the correct training values
            
        negative_log_likelihoods = -np.log(correct_confidences) # Calculate negative log for CCE
        
        return negative_log_likelihoods
    def backward(self, dvalues, y_true):
        # Identify number of samples
        samples = len(dvalues)
        # Identify number of labels/sample
        labels = len(dvalues[0])
        
        # If labels are sparce, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Clip dvalues to avoid dividing by zero
        dvalues_clipped = np.clip(dvalues, 1e-7, 1 - 1e-7)
        # Calculate gradient
        self.dinputs = -y_true / dvalues_clipped
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        
class Loss_BinaryCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        # Clip data to prevent division by zero
        # Clip both sides to prevent shifting the mean
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        
        return sample_losses
    
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        outputs = len(dvalues[0])
        
        # Clip data to prevent division by 0, both sides to prevent shifting the mean
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        
        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        
class Loss_MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=1)
        return sample_losses
        
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs per sample
        outputs = len(dvalues[0])
        
        # Gradient of values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        
class Loss_MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        samples_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return samples_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        
        outputs = len(dvalues[0])
        
        # Calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Accuracy functions

class Accuracy:
    def calculate(self, predictions, y):
        
        # Compare results
        comparisons = self.compare(predictions, y)
        
        # Calculate accuracy
        accuracy = np.mean(comparisons)
        
        # Add accumulated sum across samples and epochs
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        
        return accuracy # Parent accuracy class
    
    def calculate_accumulated(self):
        
        accuracy = self.accumulated_sum / self.accumulated_count
        
        return accuracy
    
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

class Accuracy_Regression(Accuracy):
    def __init__(self):
        # Create precision property
        self.precision = None

    # Calculate precision value based on true y values
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
            
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision
    
class Accuracy_Categorical(Accuracy):
    
    # No initialization needed
    def init(self, y):
        pass
    
    # Compare predictions to true y values
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
    

# Optimizer functions

class Optimizer_SGD:
    # Initialize, Set Learning Rate
    def __init__(self, learning_rate=1., decay = 0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0
        
    def pre_update_params(self):
        if self.decay:
            # Logarithmic decrease of learning rate to achieve smaller shifts in local minimums
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
            
    def update_params(self, layer):
        if self.momentum:
            # If layer doesn't have momentum arrays, create uninitialized arrays of zeroes
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            # Otherwise update weights/biases using no momentum
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        # Change layer weights/biases regardless of momentum (weight_updates will hold changes)
        layer.weights += weight_updates
        layer.biases += bias_updates
        
    def post_update_params(self):
        self.iterations += 1    
        
class Optimizer_RMSProp:
    # Initialize, Set Learning Rate, decay, and epsilon
    def __init__(self, learning_rate=0.001, decay = 0., epsilon = 1e-7, rho = 0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
        
    def pre_update_params(self):
        if self.decay:
            # Logarithmic decrease of learning rate to achieve smaller shifts in local minimums
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    def update_params(self, layer):

        # If layer doesn't have cache arrays, create uninitialized arrays of zeroes
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
                
        # Update cache with squared layer gradients
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2
            
        # Normal SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
        
    def post_update_params(self):
        self.iterations += 1   
    
class Optimizer_Adagrad:
    # Initialize, Set Learning Rate, decay, and epsilon
    def __init__(self, learning_rate=1., decay = 0., epsilon = 1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        
    def pre_update_params(self):
        if self.decay:
            # Logarithmic decrease of learning rate to achieve smaller shifts in local minimums
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    def update_params(self, layer):

        # If layer doesn't have cache arrays, create uninitialized arrays of zeroes
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
                
        # Update cache with squared layer gradients
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
            
        # Normal SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
        
    def post_update_params(self):
        self.iterations += 1   
        
class Optimizer_Adam:
    # Initialize, Set Learning Rate, decay, and epsilon
    def __init__(self, learning_rate=0.001, decay = 0., epsilon = 1e-7, beta_1 = 0.9, beta_2 = 0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        
    def pre_update_params(self):
        if self.decay:
            # Logarithmic decrease of learning rate to achieve smaller shifts in local minimums
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
            
    def update_params(self, layer):
        # If layer doesn't have cache arrays, create uninitialized arrays of zeroes
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.bias_momentums = np.zeros_like(layer.biases)
                
        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
            
        # Get corrected momentum (self.iterations is 0 for first pass so offset by 1)
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        
        # SGD Parameter update and normalization with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
    def post_update_params(self):
        self.iterations += 1   
        
# Packaged model class

class Layer_Input:
    def forward(self, inputs, training):
        self.output = inputs

class Model:
    def __init__(self, dictionary):
        self.layers = []
        self.softmax_classifier_output = None
        self.dictionary = dictionary
        
    def add(self, layer):
        self.layers.append(layer)

    def evaluate(self, X_val, y_val, *, batch_size=None):
        
        # Default value if batch size is not set
        validation_steps = 1
        
        # Calculate number of steps
        if batch_size is not None:
            validation_steps = len(X_val)//batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
                
        # Reset accumulated accuracy and loss values
        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):
                
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val   
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]
            
            # Forward pass
            output = self.forward(batch_X, training=False)
            
            # Calculate validation loss
            loss = self.loss.calculate(output, batch_y)
            
            # Calculate predictions and accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, batch_y)
            
            validation_loss = self.loss.calculate_accumulated()
            validation_accuracy = self.accuracy.calculate_accumulated()
            
            # Print validation summary
            print(f'validation, ' +
            f'acc: {validation_accuracy:.3f}, ' +
            f'loss: {validation_loss:.3f}')
            return validation_accuracy
        
    # Asterisk means loss and optimizer are required keyword arguments ex. loss=Loss_MeanSquaredError()
    # Now added functionality for importing models so required parameters are None by default
    def set(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        if optimizer is not None:    
            self.optimizer = optimizer
        if accuracy is not None:    
            self.accuracy = accuracy
       
    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        
        # Initialize accuracy precision
        self.accuracy.init(y)
        
        # Default value if batch_size not set
        train_steps = 1
        
        # If validation data is passed, set default step for validation as well
        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data
            
        # Calculate steps
        if batch_size is not None:
            
            train_steps = len(X) // batch_size
            
            # Account for integer division rounding down
            if train_steps * batch_size < len(X):
                train_steps += 1
                
            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1
                    
        # Main training loop
        for epoch in range(1, epochs+1):
            
            # Print epoch number
            print(f'Epoch: {epoch}')
            
            # Reset cumulative values
            self.loss.new_pass()
            self.accuracy.new_pass()
            
            for step in range(train_steps):
                
                # If batch size not set, train with full data set
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                
                # Otherwise slice batch
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]
                    
                # Perform forward pass
                output = self.forward(batch_X, training=True)
                        
                # Calculate loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss
            
                # Calculate predictions and accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)
            
                # Perform backward pass
                self.backward(output, batch_y)
            
                # Optimize
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()
            
                # Print training summary
                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                    f'acc: {accuracy:.3f}, ' +
                    f'loss: {loss:.3f} (' +
                    f'data_loss: {data_loss:.3f}, ' +
                    f'reg_loss: {regularization_loss:.3f}), ' +
                    f'lr: {self.optimizer.current_learning_rate}')
        
            # Calculate and print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()
        
            print(f'training, ' +
            f'acc: {epoch_accuracy:.3f}, ' +
            f'loss: {epoch_loss:.3f} (' +
            f'data_loss: {epoch_data_loss:.3f}, ' +
            f'reg_loss: {epoch_regularization_loss:.3f}), ' +
            f'lr: {self.optimizer.current_learning_rate}')
        
        # Validate data
        if validation_data is not None:
            
            val_acc = self.evaluate(*validation_data, batch_size=batch_size)

        return epoch_accuracy, val_acc
            

    def finalize(self):
        # Set input layer
        self.input_layer = Layer_Input()
        
        # Count number of layers
        layer_count = len(self.layers)
        
        # Create list of trainable layers with weights/biases for backpropogation
        self.trainable_layers = []

        for i in range(layer_count):
            # If it's in the first layer, previous layer was input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
                
            # All layers except first and last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
                
            # Last layer points to loss function
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
                
            # Keep track of trainable layers for backpropogation
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
           
            # Update loss object with trainable layers    
            # Added if statement to ignore IF loading pretrained model without optimizer
            if self.loss is not None:
                self.loss.remember_trainable_layers(self.trainable_layers)
       
        # If output function is Softmax Categorical Cross-Entropy, create combined activation/loss object
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_Categorical_Cross_Entropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossEntropy()
        
                
    def forward(self, X, training):
        # Call forward on input layer to set input_layer.output for next layer
        self.input_layer.forward(X, training)
        # Call forward on every layer
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        # Return output of last layer
        return layer.output
    
    def backward(self, output, y):
        # If softmax classifier
        if self.softmax_classifier_output is not None:
            
            # Call backward method on combined activation/loss function to populate dinputs
            self.softmax_classifier_output.backward(output, y)
            
            # Backward method of last layer will not be called (it was called already in the combined object) - set dinputs manually 
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            
            # Iterate backward through all layers except the softmax activation function
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        
        # Create dinputs property by calling backward on loss function
        self.loss.backward(output, y)
        
        # Pass dinputs backwards through each layer to calculate gradient
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
            
    def get_parameters(self):
        # Iterate through all layers and return parameters
        params = []
        for layer in self.trainable_layers:
            params.append(layer.get_parameters())
        return params
    
    def set_parameters(self, parameters):
        
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)
            
    def save_parameters(self, path):  
        # Open file in binary-write mode
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)
            
    def load_parameters(self, path): 
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))
            
    def save(self, path):
        
        # Deep copy current structure
        model = copy.deepcopy(self)
        
        # Reset accumulated values in loss/acc
        model.loss.new_pass()
        model.accuracy.new_pass()
        
        # Remove data from input layer and gradients from loss object
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)
        
        # Remove these properties from every layer
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)
       
        # Save model to specified file
        with open(path, 'wb') as f:
            pickle.dump(model, f)
            
    @staticmethod
    def load(path):
        
        with open(path, 'rb') as f:
            model = pickle.load(f)
            
        return model
    
    # Predict will return array of probability vectors of confidences for each class
    def predict(self, X, *, batch_size=None):
        
        # Default value if batch_size is None
        prediction_steps = 1
        
        # Calculate number of steps
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1
                
        output = []
        
        for step in range(prediction_steps):
            
            # Divide into batches
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]
        
        # Forward pass
        batch_output = self.forward(batch_X, training=False)
        
        # Append prediction
        output.append(batch_output)
        
        # Stack and return results
        return np.vstack(output)
    
    # Full Predict will argmax probabilities and print the corresponding labels
    def fullpredict(self, X, *, batch_size=None):
        confidences = self.predict(X, batch_size=batch_size)
        
        predictions = self.output_layer_activation.predictions(confidences)
        
        #for prediction in predictions:
        #    print(self.dictionary.labels[prediction])
        if len(X) > 1:
            return self.label(predictions)
        else:
            return self.label(predictions[0])
            
    def label(self, X):
        predictions = []
        if type(X) is list or type(X) is np.ndarray:
            for val in X:
                predictions.append(self.dictionary.labels[val])
        else:
            return self.dictionary.labels[X]
        return predictions
       
def print_predict(X, y, *, images=None, time=0.5, click=False):
    if images is None:
        for inp, label in zip(X, y):
            prediction = model.fullpredict(inp)[0]
            label = model.label(label)
            if prediction is not label:
                print("Incorrect Prediction")
            print("Prediction: " + prediction + " | True Value: " + label)
    else:
        if not click:
            plt.ion()
        for image, inp, label in zip(images, X, y):
            prediction = model.fullpredict(inp)[0]
            label = model.label(label)
            if prediction is not label:
                print("Incorrect Prediction")
            print("Prediction: " + prediction + " | True Value: " + label)
            plt.imshow(image)
            plt.show()
            plt.pause(time)

fashion_mnist_labels = {
0 : 'T-shirt/top' ,
1 : 'Trouser' ,
2 : 'Pullover' ,
3 : 'Dress' ,
4 : 'Coat' ,
5 : 'Sandal' ,
6 : 'Shirt' ,
7 : 'Sneaker' ,
8 : 'Bag' ,
9 : 'Ankle boot'
}

X, y, X_test, y_test = create_mnist_dataset('fashion_mnist_images')
print('Dataset loaded')
fdict = Data_Dict(fashion_mnist_labels)

keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
test_keys = np.array(range(X_test.shape[0]))
np.random.shuffle(test_keys)

X = X[keys]
y = y[keys]
X_test = X_test[test_keys]
y_test = y_test[test_keys]
images_list = X_test
# Normalize pixel values [0, 255] to [-1, 1] (Subtract half of max val (255) and divide by half of max val)
X = (X.astype(np.float32) - 127.5) / 127.5

X_test = (X_test.astype(np.float32) - 127.5) / 127.5

# Currently X's are 2D arrays and must be flattened into vectors to pass through dense layers
# X.shape[0] represents the number of images (samples), -1 will flatten to a vector
X = X.reshape(X.shape[0], -1)

X_test = X_test.reshape(X_test.shape[0], -1)

# We don't want to train using the data in it's current order, because the first s/l (samples/labels) will be the same label
# Therefore, we will generate keys to shuffle both X, y the same way

model = Model.load('fashion_mnist_model')

print_predict(X_test, y_test, images=None)


"""
model = Model(dictionary=fdict)
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128,10))
model.add(Activation_Softmax())

model.set(loss=Loss_Categorical_Cross_Entropy(), 
          optimizer=Optimizer_Adam(decay=1e-4), 
          accuracy=Accuracy_Categorical())

model.finalize()



val_acc, e_acc = model.train(X, y, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)

model.save('fashion_mnist_model')
"""





"""
last_rms, current_rms = None, None
while(current_rms is None or last_rms > current_rms):
    
    val_acc, e_acc = model.train(X, y, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)
    
    if last_rms is None:
        last_rms = math.sqrt((val_acc**2 + e_acc**2)/2)
        last_params = model.get_parameters()
    elif current_rms is None:
        current_rms = math.sqrt((val_acc**2 + e_acc**2)/2)
        current_params = model.get_parameters()
    else:
        last_rms = current_rms
        current_rms = math.sqrt((val_acc**2 + e_acc**2)/2)
        last_params = current_params
        current_params = model.get_parameters
    
with open('fashion_mnist_params', 'wb') as f:
    pickle.dump(last_params, f)
"""

