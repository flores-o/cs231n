import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  """
  scores = X.dot(W)  # shape [N X C]
  scores -= np.max(scores, axis=1)
  p = np.exp(scores) / np.sum(np.exp(scores), axis=1)
  """
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in range(num_train):
    #loss
    scores = X[i].dot(W)
    #shift values for 'scores' for numeric reason; Now the max value will be 0
    scores -= scores.max()
    scores_expsum = np.sum(np.exp(scores))
    cor_ex = np.exp(scores[y[i]])
    loss += -np.log(cor_ex / scores_expsum)

    #grad
    #for correct class
    dW[:, y[i]] += (-1) * (scores_expsum - cor_ex) / scores_expsum * X[i]
    for j in range(num_classes):
      #pass correct class gradient
      if j == y[i]:
        continue
      #for incorrect classes
      dW[:, j] += np.exp(scores[j]) / scores_expsum * X[i]

  
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2 * reg * W

   
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores -=  np.max(scores)
  scores_expsum = np.sum(np.exp(scores), axis = 1)
  cor_exp = np.exp(scores[list(range(num_train)), y])
  # loss += -np.log(cor_ex / scores_expsum)
  loss = cor_exp / scores_expsum
  loss = -np.sum(np.log(loss)) / num_train + reg * np.sum(W * W)

  # grad
  s = np.divide(np.exp(scores), scores_expsum.reshape(num_train, 1))
  s[list(range(num_train)), y] =- (scores_expsum - cor_exp) / scores_expsum
  dW = X.T.dot(s)
  dW /= num_train
  dW += 2 * reg * W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

