import numpy as np

def softmax_loss_gradient(W, X, y, reg):
    """
    Softmax loss and gradient function using a vectorized implementation
    
    Inputs:
    - W: A numpy array of shape (num_labels, num_features) containing the weights
    - X: A numpy array of shape (num_features, num_samples) containing data.
    - y: A numpy array of shape (num_samples, ) containing correct labels.
    - reg: (float) regularization strength.  
       
    """
    # Parameters
    num_samples = X.shape[1] 
    num_labels = W.shape[0]
    
    # Scores matrix
    S = np.dot(W, X)

    # Calculate normalized probability matrix in a numerically stable way
    C = np.exp(-np.max(S, axis = 0))
    P = np.exp(S+C) / np.sum(np.exp(S+C), axis = 0)
        
    # Loss = (total average data loss) + (regularization loss)
    loss_data = np.mean(-np.log(P[y, np.arange(num_samples)]))
    loss_reg = reg * np.sum(W[:, :-1] * W[:, :-1])
    loss = loss_data + loss_reg

    # Gradient of (total average data loss plus regularization loss) w.r.t. weights matrix W
    P[y, np.arange(num_samples)] -= 1 # adjust probability matrix
    dW = (np.dot(P, X.T) / num_samples) +  2 * reg * np.hstack([W[:,:-1], np.zeros((num_labels, 1))])
   
    return loss, dW