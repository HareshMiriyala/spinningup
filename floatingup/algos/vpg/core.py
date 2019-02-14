import numpy as np
import tensorflow as tf
import scipy.signal
from gym.spaces import Box, Discrete
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.beta import Beta

EPS = 1e-8
"""
def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError

def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]

"""
# wont need any of the placeholder functions in vpg.py

# def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
#     for h in hidden_sizes[:-1]:
#         x = tf.layers.dense(x, units=h, activation=activation)
#     return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

# rebuild mlp class, but in pytorch
class NeuralNetwork(nn.Module):
    def __init__(self,layers,activation=torch.tanh,output_activation=None,output_squeeze=False):
        super(NeuralNetwork,self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_squeeze = output_squeeze
        # form the layers here by appending linear layers to nn.ModuleList()
        for i,layer in enumerate(layers[1:]): # except the first input layer
            self.layers.append(nn.Linear(layers[i],layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, input):
        x = input
        # build the network
        for layer in self.layers[:-1]: # until the last layer
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x)
        else:
            x = self.output_activation(self.layers[-1](x))
        return x if self.output_squeeze==False else x.squeeze()

def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]

def count_vars(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


"""
Policies
"""

# def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
#     act_dim = action_space.n
#     logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
#     logp_all = tf.nn.log_softmax(logits)
#     pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
#     logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
#     logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
#     return pi, logp, logp_pi

# debuild categorical policy, but in pytorch.
class CategoricalPolicy(nn.Module):
    def __init__(self,in_features,hidden_sizes,activation,
                 output_activation,action_dim):
        super(CategoricalPolicy,self).__init__()
        self.logits = NeuralNetwork(layers=[in_features]+list(hidden_sizes)+[action_dim],
                                    activation=activation)

    def forward(self, x,a=None): # what is a ? why are we finding logp. what is logp ?
        logits = self.logits(x)
        policy = Categorical(logits=logits)
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).squeeze()
        if a is not None:
            logp = policy.log_prob(a).squeeze()
        else:
            logp = None

        return pi,logp,logp_pi


# def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
#     act_dim = a.shape.as_list()[-1]
#     mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
#     log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
#     std = tf.exp(log_std)
#     pi = mu + tf.random_normal(tf.shape(mu)) * std
#     logp = gaussian_likelihood(a, mu, log_std)
#     logp_pi = gaussian_likelihood(pi, mu, log_std)
#     return pi, logp, logp_pi

# rebuild gaussian policy, but in pytorch
class GaussianPolicy(nn.Module):
    def __init__(self,in_features,hidden_sizes,activation,output_activation,action_dim):
        super(GaussianPolicy,self).__init__()
        self.mu = NeuralNetwork(layers=[in_features]+list(hidden_sizes)+[action_dim],
                                activation = activation,
                                output_activation=output_activation)
        self.log_std = nn.Parameter(-0.5*torch.ones(action_dim,dtype=torch.float32))

    def forward(self, input):
        mu = self.mu
        policy = Normal(mu,self.log_std.exp())
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).sum(dim=1) # why do we have the .sum(dim=1) ?? find out
        if a is not None:
            logp = policy.log_prob(a).sum(dim=1)
        else :
            logp = None

        return pi,logp,logp_pi


# define a new policy called the Beta policy.
class BetaPolicy(nn.Module):
    def __init__(self,in_features,hidden_sizes,activation,output_activation,action_dim):
        super(BetaPolicy,self).__init__()
        self.alpha = NeuralNetwork(layers=[in_features]+list(hidden_sizes)+[action_dim],
                                   activation=activation,
                                   output_activation=output_activation)
        self.beta = NeuralNetwork(layers=[in_features]+list(hidden_sizes)+[action_dim],
                                   activation=activation,
                                   output_activation=output_activation)
    def forward(self,x,a=None):
        # according to the paper, add 1 to both alpha and beta
        # TODO : check if this is the right way to add bias to alpha and beta.
        alpha = 1+self.alpha
        beta = 1+self.beta
        policy = Beta(alpha,beta)
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).sum(dim=1)
        if a is not None:
            logp = policy.log_prob(a).sum(dim=1)
        else :
            logp = None

        return pi,logp,logp_pi

"""
Actor-Critics
"""
# def mlp_actor_critic(x, a, hidden_sizes=(64,64), activation=tf.tanh,
#                      output_activation=None, policy=None, action_space=None):
#
#     # default policy builder depends on action space
#     if policy is None and isinstance(action_space, Box):
#         policy = mlp_gaussian_policy
#     elif policy is None and isinstance(action_space, Discrete):
#         policy = mlp_categorical_policy
#
#     with tf.variable_scope('pi'):
#         pi, logp, logp_pi = policy(x, a, hidden_sizes, activation, output_activation, action_space)
#     with tf.variable_scope('v'):
#         v = tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
#     return pi, logp, logp_pi, v


class ActorCritic(nn.Module):
    def __init__(self, in_features, action_space, hidden_sizes=(64, 64), activation=torch.tanh,
                 output_activation=None, policy=None):
        super(ActorCritic, self).__init__()
        if policy is None and isinstance(action_space,Box):
            self.policy = GaussianPolicy(in_features,hidden_sizes,activation,output_activation,action_dim=action_space.shape[0])
        elif policy is None and isinstance(action_space,Discrete):
            self.policy = CategoricalPolicy(in_features,hidden_sizes,activation,output_activation,action_dim=action_space.shape[0])
        elif policy is "Beta":
            #TODO : Experimental stage for beta policy. check if this is working.
            self.policy = BetaPolicy(in_features,hidden_sizes,activation,output_activation,action_dim=action_space.shape[0])

        self.value_function = NeuralNetwork(layers=[in_features]+list(hidden_sizes)+[1],activation=activation,
                                            output_squeeze=True) # squeeze the output to remove 1 dimensions

    def forward(self, x, a=None):
        pi,logp,logp_pi = self.policy(x,a)
        v = self.value_function(x) #automatically calls the forward function

        return pi,logp,logp_pi,v
