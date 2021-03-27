import numpy as np
from LinearClassifiers.softmax import *
from LinearClassifiers.svm import *

class LinearClassifier(object):

  def __init__(self): # constructor method
    self.W = None # class attribute

  def train(self, X_train, y_train, learning_rate = 1e-3, reg = 1e-5, num_iters = 100,
            batch_size = 100, verbose = False):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X_train: A numpy array of shape (num_train_features, num_train_samples) containing training data.
    - y_train: A numpy array of shape (num_train_samples, ) containing training data labels.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    num_train_features, num_train_samples = X_train.shape
    num_train_labels = np.max(y_train) + 1 
    if self.W is None:
      # Randomly initialize W
      self.W = 0.01 * np.random.randn(num_train_labels, num_train_features)
      
    # Run batch /stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
      X_train_batch = None
      y_train_batch = None

      #########################################################################
      # TODO:                                                                 #
      # Sample batch_size elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in X_batch and their corresponding labels in           #
      # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
      # and y_batch should have shape (batch_size,)                           #
      #                                                                       #
      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #
      #########################################################################
      batch_indices = np.random.choice(num_train_samples, batch_size, replace = True)
      X_train_batch = X_train[:, batch_indices]
      y_train_batch = y_train[batch_indices]
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################

      # evaluate loss and gradient
      loss, grad = self.lossgradient(X_train_batch, y_train_batch, reg)
      loss_history.append(loss)

      # perform parameter update
      #########################################################################
      # TODO:                                                                 #
      # Update the weights using the gradient and the learning rate.          #
      #########################################################################
      self.W = self.W - learning_rate * grad
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################
      
      # Print loss history once every 100 iterations
      if verbose and (it % 100 == 0):
        print('Iteration: %d / %d, Loss: %f' % (it, num_iters, loss))

    return loss_history

  def predict(self, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - X: A numpy array of shape (num_features, num_samples) containing data.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length num_samples, and each element is an integer giving the predicted
      class.
    """
    y_pred = np.zeros(X.shape[0])
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Store the predicted labels in y_pred.            #
    ###########################################################################
    S = np.dot(self.W, X)
    y_pred = S.argmax(axis = 0)
    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return y_pred
  
  def lossgradient(self, X_train_batch, y_train_batch, reg):
    """
    Compute the loss function and its derivative. 
    Subclasses (for SVM and Softmax) will override this.

    Inputs:
    - X_train_batch: A numpy array of shape (num_train_batch_features, num_train_batch_samples) containing batch training         data.
    - y_train_batch: A numpy array of shape (num_train_batch_samples, ) containing batch training data labels.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to W; an array of the same shape as W
    """
    pass


class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  def loss(self, X_train_batch, y_train_batch, reg):
    return svm_loss_gradient(self.W, X_train_batch, y_train_batch, reg)


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def lossgradient(self, X_train_batch, y_train_batch, reg):
    return softmax_loss_gradient(self.W, X_train_batch, y_train_batch, reg)

