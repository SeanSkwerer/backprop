# Backpropagation

What is the repo all about? Well it's really just an effort on my part to re-educate myself in the fundamentals of training deep learning models. The backpropogation algorithm is one of the most important methods in machine learning today. 

Why should you care? If you are preparing for machine learning interviews and want to take a deep dive on backpropagation, then this repo might help you! More specifically, as networks and data have become larger, the need for distributed training has become more important. Companies are interested in engineers who can use or even implement distributed training solutions.

In my case, I was recently asked to explain and implement backpropagation in a machine learning engineer interview (I was not specifically interviewing for a distributed training engineer role). Having not used or studied the algorithm for about 17 years at the time, I was not able to answer the interview question very well. I first learned the backpropagation algorithm in 2007 in a course I took about neural networks in my third year of college as part of my undergraduate curriculum for a degree in mathematical decision science. I recently decided to publish that project on github. You can read a summary of the project on GitHub (https://github.com/SeanSkwerer/MyJuniorYear2007NNProject).

Generally, I don't spend time learning things that aren't immediately applicable to my day-to-day work these days. However, I found this exercise rewarding, and I think you might too.

## What is backpropagation?
Backpropagation is algorithm to calculate the partial derivatives of a function of the form
    $f(x; p_1, ..., p_L) = f_L(f_{L-1}(...f_1(x)))$
where the parameter sets $p_l$ of each function $f_l$ are independent.

Applying the chain rule to f, the formula for gradient w.r.t. $p_l$ is
    $df/dp_l = (df_1/df_2)(df_2/df_3)*...*df_l/dp_l$

Typically this will be applied in the context "learning" or "estimating" a function by 
minimizing some loss function $C(f, y, x)$ measuring the quality of the learned function
by comparing its output to observed output evaluated on the corresponding input
for a dataset, $(y_1, x_1), ..., (y_N, x_N) \in (\R^K, \R^M)$.
     
Applying the chain rule to the loss function, the partial derivative w.r.t $p_l$ is
    $dC/dp_l = (dC/df_1)*(df_1/df_2)*...*(df_l/dp_l)$

When $p_l$ is a vector or tensor the notation above means the vector or tensor of partial 
derivatives w.r.t. the constituents of $p_l$ ordered respectively.

the slope parameter for node $i$ in layer $l-1$ to node $j$ in layer $l$ is $w_lij$
the bias parameter for node $j$ in layer $l$ is $b_lj$.

## Feedforward Neural Networks
Feedforward neural networks are organized into layers, where each layer is composed of a set of nodes or "artificial neurons" that are functions. The input layer recieves "raw data". The hidden layers (1) recieve inputs from the previous layer and (2) send outputs to the next layer. The output layer returns the final output of the network. The nodes within a layer do not provide any inputs to eachother. A fully-connected layer is one in which each node in that layer recieves inputs from all nodes in the previous layer if it is hidden, or all input values if it is the input layer.

A node is composed of two parts (1) a transfer function, and (2) an activation function. The purpose of the transfer function is to aggregate the inputs to the node, and the activation function transforms the aggregated value to a scalar. A common transfer function is the linear transfer function, which is a weighted sum of the neuron inputs plus a bias term. A classic activation function is the logistic function. Given inputs $z_1, ..., z_k$, the output of a node with a linear transfer function and logistic activation function would be $o = \alpha(\tau(z))$ where $\alpha(t) = 1/(1+e^{-t})$ and $\tau(z) = w_1z_1 + ... + w_kz_k + b$.

The input to the network is $x_i=(x_{i1},...,x_{iK}) \in \R^K$.

Let's establish notation that we will use for specifying backpropagation for a fully-connected feedforward neural network. Let $l=0,...,L$ denote the layers of the network with layer 0 being identified with the network input $x$. Let $|l|$ denote the number of nodes in layer $l$. The input to node $j$ in layer $l$ is $z^l_j$ and the output of node $j$ in layer $l$ is $o^l_j$. The input to node $j$ in layer $l$ is $z^l_j = (o^{l-1}_1,...,o^{l-1}_{|l-1|})$. The weights and bias for the transfer function of node $j$ in layer $l$ are $W^l_j = (w^l_{j,1}, ..., w^l_{j,|l-1|})$ and $b^l_j$. 

Let's assume we are solving a regression problem and use a squared error loss $C(f, x_i, y_i) = (f(x_i)-y_i)^2$.

One of the important things to remember when implementing backpropogation is that the gradients are calculate per datapoint and then aggergated before the update step. So in each of these formulas the gradients or partial derivatives are going to be evaluated per datapoint in the recursive backward pass, rather than aggregating over the dataset or batch at each recursive step of the backward pass. 

For the output layer $L$:

$
\begin{equation}
\partial C / \partial o^L = 2(f(x_i)-y_i) = 2(o^L-y_i)
\end{equation}
$

$\begin{equation}\partial C / \partial W^L = \partial C / \partial o^L \cdot \partial o^L / \partial t^L \cdot \partial t^L / \partial W^L\end{equation}$

where $ \begin{equation} \partial C / \partial o^L = (\partial C / \partial o^L_1, ...,\partial C / \partial o^L_m) \end{equation} $ is the vector of partial derivatives with one element for each output dimension.

$ \begin{equation}\partial o^L_j / \partial t^L_j = t^L_j(1-t^L_j) \end{equation}$ is the derivative of the output of node $j$ w.r.t the transfer function value. 

$ \begin{equation}\partial t^L_j / \partial W^L_{qj} = o^{|L-1|}_q \end{equation}$ because the weight $w^L_{qj}$ is multiplied by the output from node $q$ in layer $L-1$ in the transfer function for node $j$ in layer $L$.

$\begin{equation} \partial C / \partial b^L = \partial C / \partial o^L \cdot \partial o^L / \partial t^L \cdot \partial t^L / \partial b^L \end{equation}$

For any layer $l$,

$\begin{equation}\partial C / \partial W^l = \partial C / \partial o^L \cdot \partial o^L / \partial t^L \cdot \partial t^L / \partial W^L \cdot ... \cdot \partial o^l / \partial t^l \cdot \partial t^l / \partial W^l \end{equation}$
 
and 

$\begin{equation}\partial C / \partial b^l = \partial C / \partial o^L \cdot \partial o^L / \partial t^L \cdot \partial t^L / \partial W^L \cdot ... \cdot \partial o^l / \partial t^l \cdot \partial t^l / \partial b^l \end{equation}$

where $ \partial t^l / \partial b^l = 1$. 

The matrix formulas for layer $l$ are:

$
\begin{equation}
\partial o^l / \partial t^l =
\begin{pmatrix}
  \partial o^l_1 / \partial t^l_1       & 0   & 0   & \cdots  & 0  \\
  0       & \partial o^l_2 / \partial t^l_2   & 0   & \cdots  & 0  \\
  \vdots  & \vdots  & \vdots  & \ddots  & \vdots \\
  0       & 0   & 0   & \cdots  & \partial o^l_{|l|} / \partial t^l_{|l|}  \\
\end{pmatrix}
= \begin{pmatrix} 
  t^l_1(1-t^l_1)      & 0   & 0   & \cdots  & 0  \\
  0       & t^l_2(1-t^l_2)   & 0   & \cdots  & 0  \\
  \vdots  & \vdots  & \vdots  & \ddots  & \vdots \\
  0       & 0   & 0   & \cdots  & t^l_{|l|}(1-t^l_{|l|})  \\
   \end{pmatrix}
\end{equation} $ 

$\begin{equation}
\partial t^l / \partial W^l = 
\begin{pmatrix}
  \partial t^l_{1,1} / \partial W^l_{1,1}       & \partial t^l_{2,1} / \partial W^l_{2,1}   & \cdots  & \partial t^l_{|l-1|,1} / \partial W^l_{|l-1|,1}  \\
  \partial t^l_{1,2} / \partial W^l_{1,2}       & \partial t^l_{2,2} / \partial W^l_{2,2}   & \cdots  & \partial t^l_{|l-1|,2} / \partial W^l_{|l-1|,2}  \\
  \vdots  & \vdots  & \vdots  & \ddots  & \vdots \\
  \partial t^l_{1,|l|} / \partial W^l_{1,|l|}       & \partial t^l_{2,|l|} / \partial W^l_{2,|l|}   & \cdots  & \partial t^l_{|l-1|,|l|} / \partial W^l_{|l-1|,|l|}  \\
\end{pmatrix}
= \begin{pmatrix}
  o^{l-1}_1       & o^{l-1}_2   & \cdots  & o^{l-1}_{|l-1|}  \\
  o^{l-1}_1       & o^{l-1}_2   & \cdots  & o^{l-1}_{|l-1|} \\
  \vdots  & \vdots  & \vdots  & \ddots  & \vdots \\
  o^{l-1}_1       & o^{l-1}_2   & \cdots  & o^{l-1}_{|l-1|}  \\
\end{pmatrix}
\end{equation}
$

## Implementation of Gradient Descent
Backpropogation can be applied to calculate the gradients of the loss function with respect to the parameters of the network. Given those gradients, updates to the weights can be made iteratively by adding scaled values of the gradients to their respective parameters. Each step of the basic gradient descent algorithm with a fixed learning rate $\rho$ is:

0. sample a batch of observations $(y_{i_1}, x_{i_1}), ..., (y_{i_B}, x_{i_B})$ of size $B$ , $1 <= B <= N$
1. calculate the gradients of the biases and weights for each element in the batch
2. let $\delta^l_i W = [\partial C/ \partial W^l_{qj}(y_i, x_i)]_{q,j}^{|l-1|,l}$ and $\delta^l_i b = [\partial C/ \partial b^l_{j}(y_b, x_b)]_{j}^{l}$ be the matrices and vector for the weight and biases gradients of layer $l$ evaluated at observation $i$. Calculate the batch average gradients for each layer $\delta_l = \frac{1}{B} \sum {\delta^l_i}$.
3. update the weights and biases of layer $l$ with the batch average gradients
$
\begin{equation}
W^l = W^l - \rho \delta_l W
\end{equation}
$
$
\begin{equation}
b^l = b^l - \rho \delta_l b
\end{equation}
$
Steps 0 to 3 are repeated until some termination condition is reached. Typically termination conditions are based on the current magnitude of the gradients, the current loss value, and the maximum number of allowed iterations.

## Distributed Training
Due to the increasingly common use of larger and larger of models and datasets, distributed training has become an important part of many deep learning projects. This repo does not include a distributed training component, however, given the importance of this topic in todays deep learning ecosystem I would encourage readers to learn more. For example, Horovod is an open source distributed deep learning training framekwork (https://github.com/horovod/horovod).

