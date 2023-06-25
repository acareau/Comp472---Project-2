# Comp472---Project-2

Question 1 (Java):

a) How many hidden layers have you used?

  None. This is a simple function, two input nodes to one output node.

b) What are the weights and biases of each node?

Weight of each input node is set to 1, as they are both equally important. Bias is set to -1 to ensure that the function returns 0 if only one of the inputs is equal to 1.


Question 2 (Python):

a) How many hidden layers have you used? And why?

Just one, because the training set is pretty small and not very complex. 

b) How many nodes in each hidden layer and why that number of nodes in particular?

5 nodes because there are 5 inputs. 

c) What is the activation function that you used and why? Did you use the same activation function in all layers? Why?

Both. The ReLu is used in the hidden layer and the Sigmoid is used in the output layer. 
We use the ReLu in the hidden layer because it is more efficient, and it reduces overfitting. 
We use the Sigmoid in the output layer because it is better at classifying binary outputs because of the way
it "squashes" values between 0 and 1 (giving a probability). It also gives a clear decision boundary at 0.5.

d) What learning algorithm did you use to train the neural net and why?

Batch gradient descent. In this gradient descent, the whole data set is used in each epoch, this can be expensive 
but in our case it's ok because the data set is small. This method tends to be more accurate than Schotastic 
gradient descent or mini-batch gradient descent as it considers the whole dataset each time. It is not 
used as much due to its high computing cost but in this case, with a small dataset, this method made sense.

e) Can you use one hidden layer only to solve this problem? If yes, how many nodes are you going to have in it? And why?

  Yes. Both datasets can be solved with a simple OR gate on inputs A and C, therefore the hidden layer can have a single node, or there can be no hidden layer at all. The output node can simply be reached by a single function.
  
f) Can we use 5 hidden layers? Is that a good idea? Justify your answer.

  Yes, it's possible to use 5 layers. No, it wouldn't be a good idea. Since the datasets can be solved with a single function, adding layers and nodes needlessly adds time and space complexity to the network.

g) How did the neural net do in classifying the testing set? Comment on how good or bad it learned the function from the training set.

It did good. 
