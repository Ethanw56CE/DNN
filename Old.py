'''                
# Create dataset
#X, y = spiral_data(samples=100, classes=2)
#X_test, y_test = spiral_data(samples=100, classes=2)

X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=1000, classes=3)

# Reshape labels to be list of lists
#y = y.reshape(-1, 1)
#y_test = y_test.reshape(-1, 1)

# Initialize model
model = Model()

# Add layers
model.add(Layer_Dense(2, 512, lambda_l2w=5e-4, lambda_l2b=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(512, 3))
model.add(Activation_Softmax())


# Set loss, optimizer and accuracy functions
model.set(loss=Loss_Categorical_Cross_Entropy(), 
          optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
          accuracy=Accuracy_Categorical())

# Finalize model
model.finalize()

# Train model
model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)    

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].scatter(X_test[:,0], X_test[:,1], c=y, cmap="brg")
ax[0].set_title("Validation data")
ax[1].scatter(model.layers[-1].output[:,0], model.layers[-1].output[:,1], c=y, cmap="brg")
ax[1].set_title("Validation output")
plt.show()
'''
# Sine data regression    
"""
# Regression model training
X, y = sine_data()

# 1 input feature, 64 outputs
dense1 = Layer_Dense(1, 64)

activation1 = Activation_ReLU()

dense2 = Layer_Dense(64, 64)

activation2 = Activation_ReLU()

dense3 = Layer_Dense(64, 1)

activation3 = Activation_Linear()

loss_function = Loss_MeanSquaredError()

optimizer = Optimizer_Adam(learning_rate=0.005, decay=1e-3)

# Accuracy precision for accuracy calculation
# There are no really accuracy factor for regression problem,
# but we can simulate/approximate it. We'll calculate it by checking
# how many values have a difference to their ground truth equivalent
# less than given precision
# We'll calculate this precision as a fraction of standard deviation
# of al the ground truth values
accuracy_precision = np.std(y) / 250
fig, ax = plt.subplots(nrows=2, ncols=5)
plot = 1
row = 0
column = 0

for epoch in range(10001):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    
    # Loss calculation
    data_loss = loss_function.calculate(activation3.output, y)
    regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2) + loss_function.regularization_loss(dense3)
    loss = data_loss + regularization_loss
    
    # Calculate accuracy from output of activation2 and targets
    # To calculate it we're taking absolute difference between
    # predictions and ground truth values and compare if differences
    # are lower than given precision value
    predictions = activation3.output
    accuracy = np.mean(np.absolute(predictions - y) < accuracy_precision)
    
    # Output training updates
    if not epoch % 1000 and epoch != 0:
        print(f'Iteration: {epoch}, ' + f'acc: {100*accuracy:.3f}%, ' + f'loss: {loss:.3f} ' + f'data loss: {data_loss:.3f} ' + f'reg loss: {regularization_loss:.3f} ' + f'lr: {optimizer.current_learning_rate:.5f}')
        ax[row, column].plot(X, y)
        ax[row, column].plot(X, activation3.output)
        ax[row, column].set_title(f'Plot iteration: {epoch}')
        plot += 1
        column += 1
        if(column == 5):
            row = 1
            column = 0
        

    # Backward pass
    loss_function.backward(activation3.output, y)
    activation3.backward(loss_function.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    # Update weight and bias parameters
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()
    

print("Done")
print("Loss: ", loss)
print("Accuracy: ", accuracy * 100, "%")
plt.show()

X_test, y_test = sine_data()
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)

plt.plot(X_test, y_test)
plt.plot(X_test, activation3.output)
plt.show()
"""    

# Adam 3 classes
"""
X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 64, lambda_l2w=5e-4, lambda_l2b=5e-4) #Inputs, Neurons (Outputs)

activation1 = Activation_ReLU() # Recitified linear (Sets negative outputs -> 0)

dropout1 = Layer_Dropout(0.1) # Specifies the % of neurons kept each training pass

dense2 = Layer_Dense(64, 3) #Inputs (Previous # Neurons), Neurons (Ouptuts)

#activation2 = Activation_Softmax() # Normalizes outputs so \sum = 1 (Models probabilities rather than arbitrary weights)

loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy() # Combination Softmax activation and loss function

optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-5)


#epoch = 0
for epoch in range(10001):
#while optimizer.current_learning_rate > 0.01:
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)
    
    # Loss calculation
    data_loss = loss_activation.forward(dense2.output, y)
    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)
    loss = data_loss + regularization_loss

    # Accuracy calculation
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 1000:
        print(f'Iteration: {epoch}, ' + f'acc: {100*accuracy:.3f}%, ' + f'loss: {loss:.3f} ' + f'data loss: {data_loss:.3f} ' + f'reg loss: {regularization_loss:.3f} ' + f'lr: {optimizer.current_learning_rate:.5f}')

    # Backwards pass
    loss_activation.backward(loss_activation.output, y) # Creates loss_activation.dinputs with backward pass
    dense2.backward(loss_activation.dinputs) # Creates dense2.dinputs
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs) # Creates activation1.dinputs
    dense1.backward(activation1.dinputs) # Completes calculation of gradient

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()


print("Done")
print("Loss: ", loss)
print("Accuracy: ", accuracy * 100, "%")

fig, ax = plt.subplots(nrows=2, ncols=2)

ax[0, 0].scatter(X[:,0], X[:,1], c=y, cmap="brg")
ax[0, 0].set_title('Training Input')
ax[0, 1].scatter(loss_activation.output[:,0], loss_activation.output[:,1], c=y, cmap="brg")
ax[0, 1].set_title('Training Output')
# Validation
Xtest, ytest = spiral_data(samples=100, classes=3)

dense1.forward(Xtest)
activation1.forward(dense1.output)
dense2.forward(activation1.output)

loss = loss_activation.forward(dense2.output, ytest)

#data_loss = loss_activation.forward(dense2.output, ytest)
#regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)
#loss = data_loss + regularization_loss

predictions = np.argmax(loss_activation.output, axis=1)
if len(ytest.shape) == 2:
    ytest = np.argmax(ytest, axis=1)
accuracy = np.mean(predictions==ytest)

print(f'validation, acc: {accuracy*100:.3f}, loss: {loss:.3f}')

ax[1, 0].scatter(Xtest[:,0], Xtest[:,1], c=ytest, cmap="brg")
ax[1, 0].set_title('Validation Input')
ax[1, 1].scatter(loss_activation.output[:,0], loss_activation.output[:,1], c=ytest, cmap="brg")
ax[1, 1].set_title('Validation Output')

plt.get_current_fig_manager().window.state('zoomed')
plt.show()
"""

# Binary Logistic Regression (2 classes)
'''

X, y = spiral_data(samples=100, classes=2)

# Reshape data so that it is a list of lists
# Inner list contains one output (either 0 or 1) representing each class, for each neuron
y = y.reshape(-1, 1)

dense1 = Layer_Dense(2, 64, lambda_l2b=5e-4, lambda_l2w=5e-4)

activation1 = Activation_ReLU()

dense2 = Layer_Dense(64, 1)

activation2 = Activation_Sigmoid()

loss_function = Loss_BinaryCrossentropy()

optimizer = Optimizer_Adam(decay=5e-7)

for epoch in range(30001):
    # Forward pass    
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    # Loss calculation
    data_loss = loss_function.calculate(activation2.output, y)
    regularization_loss = loss_function.regularization_loss(dense1) + loss_function.regularization_loss(dense2)
    loss = data_loss + regularization_loss
    
    # Accuracy calculation
    # Converts probability output to zeros/ones representing the two classes
    predictions = (activation2.output > 0.5) * 1
    accuracy = np.mean(predictions==y)
    
    # Output training updates
    if not epoch % 1000:
        print(f'Iteration: {epoch}, ' + f'acc: {100*accuracy:.3f}%, ' + f'loss: {loss:.3f} ' + f'data loss: {data_loss:.3f} ' + f'reg loss: {regularization_loss:.3f} ' + f'lr: {optimizer.current_learning_rate:.5f}')

    # Backward pass
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    # Update weight and bias parameters
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
    
print("Done")
print("Loss: ", loss)
print("Accuracy: ", accuracy * 100, "%")

fig, ax = plt.subplots(nrows=2, ncols=2)

ax[0, 0].scatter(X[:,0], X[:,1], c=y, cmap="brg")
ax[0, 0].set_title('Training Input')
ax[0, 1].scatter(activation2.output[:], activation2.output[:], c=y, cmap="brg")
ax[0, 1].set_title('Training Output')

# Validating the model
X_test, y_test = spiral_data(samples=100, classes=2)

# Reshape
y_test = y_test.reshape(-1, 1)

# Validation forward pass
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

# Loss and Accuracy
loss = loss_function.calculate(activation2.output, y_test)

predictions = (activation2.output > 0.5) * 1
accuracy = np.mean(predictions==y_test)

print(f'Validation, acc: {accuracy:.3f}, loss: {loss:.3f}')

ax[1, 0].scatter(X_test[:,0], X_test[:,1], c=y_test, cmap="brg")
ax[1, 0].set_title('Validation Input')
ax[1, 1].scatter(activation2.output[:], activation2.output[:], c=y_test, cmap="brg")
ax[1, 1].set_title('Validation Output')

plt.get_current_fig_manager().window.state('zoomed')
plt.show()
'''





# Linear Random Backpropogation
"""
bestLoss = loss 
best_dense1_weights = dense1.weights.copy()
best_dense2_weights = dense2.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_biases = dense2.biases.copy()

for i in range(100000):
    dense1.weights += 0.05 * np.random.randn(dense1.weights.shape[0], dense1.weights.shape[1])
    dense2.weights += 0.05 * np.random.randn(dense2.weights.shape[0], dense2.weights.shape[1])
    dense1.biases += 0.05 * np.random.randn(dense1.biases.shape[0], dense1.biases.shape[1])
    dense2.biases += 0.05 * np.random.randn(dense2.biases.shape[0], dense2.biases.shape[1])
    
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    loss = loss_function.calculate(activation2.output, y)
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)
    
    if(loss < bestLoss):
        bestLoss = loss
        print("New Best Loss: ", loss," Iteration: ", i, " Accuracy: ", accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_biases = dense2.biases.copy()
    else:
        dense1.weights = best_dense1_weights.copy()
        dense2.weights = best_dense2_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.biases = best_dense2_biases.copy()
print("Done")
"""

"""
def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points) # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y
"""

"""
weights = [[0.2, 0.8, -0.5, 1.0], 
          [0.5, -0.91, 0.26, -0.5], 
          [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5], 
            [-0.5, 0.12, -0.33], 
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

# Implicit cast weights to numpy array, Transpose (Row 1 becomes Column 1, etc.)
# Structure of .dot return matches first input to dot()

layer1_output = np.dot(inputs, np.array(weights).T) + biases  
layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2

# Inputs first will throw "shape" error
print(layer2_output)
"""

"""
inputs = [1, 2, 3, 2.5]

weights = [
    [0.2, 0.8, -0.5, 1.0], 
    [0.5, -0.91, 0.26, -0.5], 
    [-0.26, -0.27, 0.17, 0.87]
           ]
biases = [2, 3, 0.5]

layer_outputs = []  #Output of current layer
for n_weight, n_bias in zip(weights, biases): # Returns iterable [([W0.1, W0.2, W0.3 ... W0.N], B0), ([Wn.1, Wn.2, Wn.3 ... Wn.N], Bn)]
    n_output = 0    #Output of current neuron
    for n_input, weight in zip(inputs, n_weight):   # Returns iterable [(Input{n}, Weight)] (Each node has a list of weights(Wn.1-Wn.N)
        n_output += n_input*weight  # Iterably adds weighted inputs
    n_output += n_bias  # Introduce bias shift
    layer_outputs.append(n_output)  # Save output in list
print(layer_outputs)
"""