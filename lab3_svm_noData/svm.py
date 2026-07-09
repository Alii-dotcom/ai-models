import numpy as np
from numpy.linalg import norm

eps = np.finfo(float).eps


class SVM:
    """ Models a Support Vector machine classifier based on the PEGASOS algorithm. """

    def __init__(self, n_epochs, lambDa, use_bias=True):
        """ Constructor method """

        # weights placeholder
        self._w = None
        self._original_labels = None
        self._n_epochs = n_epochs
        self._lambda = lambDa
        self._use_bias = use_bias

    def map_y_to_minus_one_plus_one(self, y):
        """
        Map binary class labels y to -1 and 1
        """
        ynew = np.array(y)
        self._original_labels = np.unique(ynew)
        assert len(self._original_labels) == 2
        ynew[ynew == self._original_labels[0]] = -1.0
        ynew[ynew == self._original_labels[1]] = 1.0
        return ynew

    def map_y_to_original_values(self, y):
        """
        Map binary class labels, in terms of -1 and 1, to the original label set.
        """
        ynew = np.array(y)
        ynew[ynew == -1.0] = self._original_labels[0]
        ynew[ynew == 1.0] = self._original_labels[1]
        return ynew

    def loss(self, y_true, y_pred):
        """
        The PEGASOS loss term

        Parameters
        ----------
        y_true: np.array
            real labels in {0, 1}. shape=(n_examples,)
        y_pred: np.array
            predicted labels in [0, 1]. shape=(n_examples,)

        Returns
        -------
        float
            the value of the pegasos loss.
        """

        """
        Write HERE the code for computing the Pegasos loss function.
        """
        hinge_loss = 1/y_true.shape[0] * np.sum(np.maximum(0, 1 - y_true * y_pred))
        return self._lambda/2 * norm(self._w) ** 2 + hinge_loss

    def fit_gd(self, X, Y, verbose=False):
        """
        Implements the gradient descent training procedure.

        Parameters
        ----------
        X: np.array
            data. shape=(n_examples, n_features)
        Y: np.array
            labels. shape=(n_examples,)
        verbose: bool
            whether or not to print the value of cost function.
        """

        if self._use_bias:
            X = np.concatenate([X, np.ones((X.shape[0],1),dtype=X.dtype)], axis=-1)

        n_samples, n_features = X.shape
        Y = self.map_y_to_minus_one_plus_one(Y)

        # initialize weights
        self._w = np.zeros(shape=(n_features,), dtype=X.dtype)

        t = 0
        # loop over epochs
        
        for e in range(1, self._n_epochs+1):
            for j in range(n_samples):
                # wt+1 = wt −ηt∇L(wt)
                t+=1
                learning_rate = 1/(t * self._lambda)
                y_pred = np.dot(X[j], self._w)
                if(Y[j] * y_pred < 1): 
                    self._w = (1 - learning_rate * self._lambda) * self._w + learning_rate * Y[j] * X[j]
                else:
                    self._w = (1 - learning_rate * self._lambda) * self._w
        
                

            # predict training data
            cur_prediction = np.dot(X, self._w)

            # compute (and print) cost
            cur_loss = self.loss(y_true=Y, y_pred=cur_prediction)

            if verbose:
                print("Epoch {} Loss {}".format(e, cur_loss))

            

    def predict(self, X):

        if self._use_bias:
            X = np.concatenate([X, np.ones((X.shape[0],1),dtype=X.dtype)], axis=-1)

        """
        Write HERE the criteria used during inference. 
        W * X > 0 -> positive class
        W * X < 0 -> negative class
        """
        return np.where(np.dot(self._w, X.T) > 0.0, 1, 0)

