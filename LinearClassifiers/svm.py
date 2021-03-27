import numpy as np

def svm_loss_gradient(W, X, y, reg):
    """
    SVM loss and gradient function using a vectorized implementation
    
    Inputs:
    - W: A numpy array of shape (num_labels, num_features) containing the weights
    - X: A numpy array of shape (num_features, num_samples) containing data.
    - y: A numpy array of shape (num_samples, ) containing labels
    - reg: (float) regularization strength.  
       
    """
    # Parameters
    num_samples = X.shape[1] 
    num_labels = W.shape[0]
    
    # Scores matrix
    S = np.dot(W, X)

    # Calculate the margins matrix
    delta = 1 # SVM parameter
    M = S - S[y, np.arange(num_samples)] + delta
    # Set margin for correct category in each column equal to 0
    M[y, np.arange(num_samples)] = 0
        
    # Final loss = (total average data loss) + (regularization loss)
    loss_data = np.mean(np.sum(M * (M > 0), axis = 0))
    loss_reg = reg * np.sum(W[:, :-1] * W[:, :-1])
    loss = loss_data + loss_reg

    # Gradient of (total average data loss plus regularization loss) w.r.t. weights matrix W
    M = (M > 0).astype(int)  # update margins matrix to represent presence of non-negative entries  
    # Adjust correct category row in each column to get the modified margins matrix
    M[y, np.arange(num_samples)] = -np.sum(M, axis = 0)
    dW = (np.dot(M, X.T) / num_samples) + 2 * reg * np.hstack([W[:,:-1], np.zeros((num_labels, 1))])
   
    return loss, dW