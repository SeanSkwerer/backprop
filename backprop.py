import numpy as np

class SquaredError:
    # assumes Y is a vector with K elements
    def __init__(self, weights=None) -> None:
        self.weights = weights

    def evaluate(self, f, x, y):
        if self.weights is None:
            return ((f.evaluate(x)-y)**2.0).T
        else:
            return (((f.evaluate(x)-y)**2.0)@self.weights).T

    def partial(self, f, x, y):
        # partial derivative w.r.t. f evaluated at x
        # assume that y shape is features by sample size
        #   so y in \R^{K x N}
        result = f.evaluate(x)-y # shape = (K, N)
        if self.weights is None:
            return  result # output shape is N by K
        else:
            return result@self.weights # output shape is N by 1

class Logistic:
    def __init__(self) -> None:
        return 
    
    def evaluate(self, x):
        return 1./(1.+np.exp(-x))
    
    def partial(self, y):
        # y = self.evaluate(x)
        return np.multiply(y, (1.-y))

class LinearFeedForwardNetwork:
    def __init__(self, input_dim, layer_dims, output_dim, activation):
        self.act = activation
        # uses random normal initialization for all parameters

        # specify network dimensions
        self.input_dim = input_dim
        self.layer_dims = layer_dims
        self.output_dim = output_dim
        self.len = 1 + len(layer_dims)
        self.forward = None
        self.backward = None

        # first layer is input_dim by layer_dims[0]
        # second layer is layer_dims[0] by layer_dims[1]
        # ...
        # final layer is layer_dims[-1] by output_dim
        
        self.slopes = list()
        self.biases = list()
        # initialize the slopes and biases to values sampled from
        # random normal distribution
        # input layer
        self.slopes.append(np.random.randn(input_dim, layer_dims[0]))
        self.biases.append(np.random.randn(1, layer_dims[0]))
        # hidden layers
        for l in range(1, len(self.layer_dims)):
            self.slopes.append(np.random.randn(layer_dims[l-1], layer_dims[l])) # shape = (|l-1|, |l|)
            self.biases.append(np.random.randn(1, layer_dims[l]))
        # output layer
        self.slopes.append(np.random.randn(layer_dims[-1], output_dim))
        self.biases.append(np.random.randn(1, output_dim))
        
    def evaluate(self, x):
        assert(x.shape[1] == self.input_dim)
        self.forward = [x]
        for slopes, biases in zip(self.slopes, self.biases):
            self.forward.append(self.act.evaluate(self.forward[-1]@slopes + biases))
        return self.forward[-1]
            
    def backprop(self, loss, x, y, alpha=None):
        # produce partial derivative w.r.t. biases and slopes in each layer and return
        #   dl/dp_l = (dl/df_L)*(df_L/df_{L-1})*...*(df_{l+1}/df_l)*(df_l/dp_l)

        # if alpha is a numeric value then the parameters will be updated with step alpha
        
        # for df_l/df_{l-1}, since each layer is linear the partial of the function
        # in layer l w.r.t input from layer l+1 is the slopes 
        # of layer l, that is df_l/df_{l+1} = self.slopes[l] = [w_l..]

        # for df_l/dp_l, since each layer is linear, the partial of the function
        # in layer l w.r.t. a slope parameter in layer l is the value from layer l-1 
        # which is multiplied by that slope, that is df_l/dw_lij = df_{l-1}j, 
        # and for the deriviates of the bias terms are  df_l/db = 1.

        # for example, for layer L
        #   the output of layer L is a_L in \R^K and q are fixed weigths in \R^K
        #   loss(f, x, y, q) = sum_{i=1}{n}sum_k(q_k(y_ik - a_Lk)**2)
        #   dl/da_L = sum_{n=1}{N}sum_k(2q_k(y_ik-a_Lk)) where k indexes nodes in layer L, the output layer
        #   dl/da_Lj = sum_{n=1}{N}(2q_j(y_ij-a_Lj)) where j indexes nodes in layer L, the output layer
        #   da_Lj/dw_Lij = a_{L-1}i where j indexes nodes in layer L and i indexes nodes in layer L-1
        #   da_Lj/db_Lj = 1 where j indexes the nodes in layer L
        #   dl/dw_Lij = dl/da_Lj*da_Lj/dw_Lij = sum_{i=1}{n}(2q_j(y_ij-a_Lj))*a_{L-1}i
        #   dl/dw_L has shape |L| by |L-1| where |l| denotes the number of nodes in layer L. For the output layer K = |L|
        #       expressed with array multiplication
        #       dl/dw_L = [dl/da_Lj]{j=0:K,i=0:|L-1|} * [da_Lj/dw_Lij]{j=0:K,i=0:|L-1|}
        #                   (K, |L-1|)             (K, |L-1|)
        #                = [sum_{n=1}{N}(2q_j(y_ij-a_Lj))]{j=0:K,i=0:|L-1|} * [a_{L-1}i]{j=0:K,i=0:|L-1|}
        #               = np.outer(loss.partial(x, y), forward[-1])
        #   da_Lj/da_{L-1}i = w_Lij # weight in node j of layer L for the output of node i in layer L-1
        #   dl/da_{L-1} = dl/da_L * da_L/da_{L-1}
        #               = [sum_{n=1}{N}sum_k(2q_j(y_ik-a_Lj)) * wLij]{j=0:K, i=0:|L-1|}
        #   dl/da_{L-2} = dl/da_L * da_L/da_{L-1} * da_{L-1}/da_{L-2}
        #               =  

        slope_gradients = list()
        bias_gradients = list()
        # calculate the gradient of the loss w.r.t the layer outputs
        # note this will also evaluate the layers to get the forward pass
        loss_gradient = loss.partial(self, x, y)[:,None,:] # shape = (N, 1, K)
        
        # calculate the gradients of each layer recurisively
        for l in range(1, self.len+1):
            layer = self.len-l+1
            batch, layer_dim = self.forward[layer].shape
            previous_layer_dim = self.forward[layer-1].shape[1]
            activation_gradient = self.act.partial(self.forward[layer])
            activation_gradient = np.diag(activation_gradient[0,:])[None,:,:]
            slope_gradients_per_layer_input = np.reshape(np.repeat(self.forward[layer-1], layer_dim), (batch, previous_layer_dim, layer_dim)) # shape = (N, |l-1|, |l|)
            slope_grads = np.mean((slope_gradients_per_layer_input@activation_gradient)*loss_gradient, axis=0)
            slope_gradients.append(slope_grads) # shape = (1, |l-1|, |l|)
            bias_gradients.append(np.mean(loss_gradient, axis=0))
            loss_gradient = loss_gradient@np.transpose(self.slopes[self.len-l]) # (dl/df_1)*(df_1/df_2)*...*(df_l/df_{l+1}) shape=(|l|, N)
        self.backward = {"slope_gradients": slope_gradients, "bias_gradients": bias_gradients}
        if alpha is not None:
            for l in range(self.len):
                self.slopes[l] -= alpha*slope_gradients[self.len-1-l]
                self.biases[l] -= alpha*bias_gradients[self.len-1-l]
        return np.mean(loss.evaluate(self, x, y))

