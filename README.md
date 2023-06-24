# Comp472---Project-2

Question 1 (JAVA):

a) How many hidden layers have you used?

  None. This is a simple function, two input nodes to one output node.

b) Weight of each input node is set to 1, as they are both equally important. Bias is set to -1 to insure that the function returns 0 if only one of the inputs is equal to 1.


Question 2 (Python):

a) How many hidden layers have you used? And why?

b) How many nodes in each hidden layer and why that number of nodes in particular?

c) What is the activation function that you used and why? Did you use the same activation function in all layers? Why?

d) What learning algorithm did you use to train the neural net and why?

e) Can you use one hidden layer only to solve this problem? If yes, how many nodes are you going to have in it? And why?

  Yes. Both datasets can be solved with a simple OR gate on inputs A and C, therefore the hidden layer can have a single node, or there can be no hidden layer at all. The output node can simply be reached by a single function.
  
f) Can we use 5 hidden layers? Is that a good idea? Justify your answer.

  Yes, it's possible to use 5 layers. No, it wouldn't be a good idea. Since the datasets can be solved with a single function, adding layers and nodes needlessly adds time and space complexity to the network.

g) How did the neural net do in classifying the testing set? Comment on how good or bad it learned the function from the training set.
