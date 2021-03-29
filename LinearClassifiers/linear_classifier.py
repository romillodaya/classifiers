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
    num_train_labels = np.max(y_train) + 1 # Assuming that the labels are like 0, 1, 2, etc. 
    if self.W is None:
      self.W = np.zeros((num_train_labels, num_train_features)) # bias column initialized to zeros        
      # Randomly initialize weights part of the W matrix
      (self.W)[:, :-1] = 0.01 * np.random.randn(num_train_labels, num_train_features-1)
      
    # Run batch /stochastic gradient descent to optimize W
    loss_history = [] # initialize empty list for storing loss history over iterations
    for it in range(num_iters):
      # Get random sample indices for batch processing
      batch_indices = np.random.choice(num_train_samples, batch_size, replace = False)
      
      # Evaluate loss and gradient on batch samples
      loss, dW = self.loss_gradient(X_train[:, batch_indices], y_train[batch_indices], reg)
      loss_history.append(loss)

      # Update weights matrix
      self.W = self.W - learning_rate * dW
   
      # Print loss history once every 10 iterations
      if verbose and (it % 10 == 0):
        print('Iteration: %d / %d, Loss: %f' % (it, num_iters, loss))

    return loss_history

  def predict(self, X):
    """
    Use the trained weights of this linear classifier to predict labels for samples

    Inputs:
    - X: A numpy array of shape (num_features, num_samples) containing data.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length num_samples, and each element is an integer giving the predicted
      class.
    """
    # Predicted labels
    y_pred = np.zeros(X.shape[1])
   
    # Calculate scores matrix
    S = np.dot(self.W, X)
    
    # Set predicted label for each sample (that is column) as equal to maximum score index value 
    y_pred = S.argmax(axis = 0)
    
    return y_pred
  
  def loss_gradient(self, X_train_batch, y_train_batch, reg):
    """
    Compute the loss function and its gradient. 
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

  def loss_gradient(self, X_train_batch, y_train_batch, reg):
    return svm_loss_gradient(self.W, X_train_batch, y_train_batch, reg)


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss_gradient(self, X_train_batch, y_train_batch, reg):
    return softmax_loss_gradient(self.W, X_train_batch, y_train_batch, reg)

