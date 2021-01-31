#Adapted from Stanford CS231n Course
import numpy as np
from abc import ABC, abstractmethod
from .helpers import flatten_unflatten


class Layer(ABC):
    def __init__(self, input_size, output_size):
        self.W = np.random.rand(input_size, output_size)
        self.b = np.zeros(output_size)
        self.x = None
        self.db = np.zeros_like(self.b)
        self.dW = np.zeros_like(self.W)

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError('Abstract class!')

    @abstractmethod
    def backward(self, x):
        raise NotImplementedError('Abstract class!')

    def __repr__(self):
        return 'Abstract layer class'


class Dropout(Layer):
    def __init__(self, p=.5):
        '''
            :param p: dropout factor
        '''
        self.mask = None
        self.mode = 'train'
        self.p = p

    def forward(self, x, seed=None):
        '''
            :param x: input to dropout layer
            :param seed: seed (used for testing purposes)
        '''
        if seed is not None:
            np.random.seed(seed)
        # YOUR CODE STARTS
        if self.mode == 'train':
            out = None

            # Create a dropout mask
            mask = (np.random.rand(*x.shape) > self.p)   #mask created with the probability
           
            # Do not forget to save the created mask for dropout in order to use it in backward
            self.mask = mask.copy()
            
            out = x*mask#input multiplied by mask

            return out
        elif self.mode == 'test':
            out = x #in test mode we are not going to use mask
            return out
        # YOUR CODE ENDS
        else:
            raise ValueError('Invalid argument!')

    def backward(self, dprev):

        dx = dprev*self.mask #out = x*mask --> dx = mask*dout

        return dx


class BatchNorm(Layer):
    def __init__(self, D, momentum=.9):
        self.mode = 'train'
        self.normalized = None

        self.x_sub_mean = None
        self.momentum = momentum
        self.D = D
        self.running_mean = np.zeros(D)
        self.running_var = np.zeros(D)
        self.gamma = np.ones(D)
        self.beta = np.zeros(D)
        self.ivar = np.zeros(D)
        self.sqrtvar = np.zeros(D)

    # @flatten_unflatten
    def forward(self, x, gamma=None, beta=None):
        if self.mode == 'train':
            sample_mean = np.mean(x, axis=0)
            sample_var = np.var(x, axis=0)
            if gamma is not None:
                self.gamma = gamma.copy()
            if beta is not None:

                self.beta = beta.copy()

            # Normalise our batch
            self.normalized = ((x - sample_mean) /
                               np.sqrt(sample_var + 1e-5)).copy()
            self.x_sub_mean = x - sample_mean

            # YOUR CODE HERE            
            
            out = self.gamma*self.normalized +self.beta #out=gamma* (X-XMu/sqrt(var(X))) + beta

            # Update our running mean and variance then store.

            running_mean = self.momentum * self.running_mean + (1 - self.momentum) * sample_mean
            running_var = self.momentum * self.running_mean + (1 - self.momentum) * sample_mean


            # YOUR CODE ENDS
            self.running_mean = running_mean.copy()
            self.running_var = running_var.copy()

            self.ivar = 1./np.sqrt(sample_var + 1e-5)
            self.sqrtvar = np.sqrt(sample_var + 1e-5)

            return out
        elif self.mode == 'test':
            out = (x-self.running_mean)/np.sqrt(self.running_var + 1e-5) #just used running mean and var already calculated in training mode
            out = self.gamma*out +self.beta
        else:
            raise Exception(
                "INVALID MODE! Mode should be either test or train")
        return out

    def backward(self, dprev):
        N, D = dprev.shape
        # YOUR CODE HERE
        dx, dgamma, dbeta = None, None, None
        dbeta = np.sum(dprev, axis=0)
        dgamma = np.sum(dprev*self.normalized, axis=0)
        
        
        dnorm = self.gamma*dprev  # normalized*gamma +beta
        dsubmean = dnorm/self.sqrtvar #((x - sample_mean) /np.sqrt(sample_var + 1e-5))---> 1/np.sqrt(sample_var + 1e-5) (self.sqrtvar)parta
        
        dvar = np.sum(self.x_sub_mean * dnorm, axis=0) #((x - sample_mean) /np.sqrt(sample_var + 1e-5)) -->(x-meanx)*dprev  partb
        
        dsqrt = (-0.5 * dvar )/ (self.sqrtvar * self.sqrtvar**2) # 1/sqrtx ' derivative -0.5/(x*sqrtx)
        dsq = 1./N*np.ones((N,D))*dsqrt #derivative of mean 1/N
        dfinal = 2* self.x_sub_mean*dsq  #derivative of square(x-mean)^2 -->2*(x-mean)-->2* submean
        dx1 = (dsubmean + dfinal) #sum derivatives ((x - sample_mean) and np.sqrt(sample_var + 1e-5))
        dmu = -1 * np.sum(dx1, axis=0)#derivative of x-mean -1* dprev

        dx2 = 1./N * np.ones((N,D)) * dmu# derivative of mean ->1/N

        dx = dx1 + dx2
        
        # Calculate the gradients
        return dx, dgamma, dbeta


class MaxPool2d(Layer):
    def __init__(self, pool_height, pool_width, stride):
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.stride = stride
        self.x = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_H = np.int(((H - self.pool_height) / self.stride) + 1)
        out_W = np.int(((W - self.pool_width) / self.stride) + 1)

        self.x = x.copy()

        # Initiliaze the output
        out = np.zeros([N, C, out_H, out_W])

        # Implement MaxPool
        # YOUR CODE HERE
        for i in range(N): #for each sample
            for j in range(C): #for each channel 
                for k in range(out_H): #height of output calculated according to stride and pool height
                    for l in range(out_W):#width of output calculated according to stride and pool width
                        out[i][j][k][l]=np.max(x[i,j, k*self.stride:k*self.stride+self.pool_height, l*self.stride:l*self.stride+self.pool_width])
                        #maximum value of related area is taken as a output.
        return out

    def backward(self, dprev):
        x = self.x
        N, C, H, W = x.shape
        _, _, dprev_H, dprev_W = dprev.shape

        dx = np.zeros_like(self.x)

        # Calculate the gradient (dx)
        # YOUR CODE HERE
        for i in range(N):
            for j in range(C):
                for k in range(dprev_H):
                    for l in range(dprev_W):
                        #pooling area selected
                        area = x[i,j, k*self.stride:k*self.stride+self.pool_height, l*self.stride:l*self.stride+self.pool_width]
                        
                        s = np.argmax(area)#indices of maximum value(inflat)
                        s = np.unravel_index(s,(self.pool_height,self.pool_width))#flatted value becomes matrix shape
                        #x=np.max(X)  -->dx=1*dout(1 is for which have the max value in that region
                        dx[i,j,k*self.stride+s[0],l*self.stride+s[1]] = 1*dprev[i,j,k,l]
        return dx


class Flatten(Layer):
    def __init__(self):
        self.N, self.C, self.H, self.W = 0, 0, 0, 0

    def forward(self, x):
        self.N, self.C, self.H, self.W = x.shape
        out = x.reshape(self.N, -1)
        return out

    def backward(self, dprev):
        return dprev.reshape(self.N, self.C, self.H, self.W)

