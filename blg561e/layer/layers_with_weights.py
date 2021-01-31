#Adapted from Stanford CS231n Course

from .layer import Layer
from copy import copy
from abc import abstractmethod
import numpy as np


class LayerWithWeights(Layer):
    '''
        Abstract class for layer with weights(CNN, Affine etc...)
    '''

    def __init__(self, input_size, output_size, seed=None):
        if seed is not None:
            np.random.seed(seed)
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

class Conv2d(LayerWithWeights):
    def __init__(self, in_size, out_size, kernel_size, stride, padding):
        self.in_size = in_size
        self.out_size = out_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.x = None
        self.W = np.random.rand(out_size, in_size, kernel_size, kernel_size)
        self.b = np.random.rand(out_size)
        self.db = np.random.rand(out_size, in_size, kernel_size, kernel_size)
        self.dW = np.random.rand(out_size)

    def forward(self, x):
        N, C, H, W = x.shape
        F, C, FH, FW = self.W.shape
        self.x = copy(x)
        # pad X according to the padding setting
        padded_x = np.pad(self.x, ((0, 0), (0, 0), (self.padding, self.padding),
                                   (self.padding, self.padding)), 'constant')

        # Calculate output's H and W according to your lecture notes
        out_H = np.int(((H + 2*self.padding - FH) / self.stride) + 1)
        out_W = np.int(((W + 2*self.padding - FW) / self.stride) + 1)

        # Initiliaze the output
        out = np.zeros([N, F, out_H, out_W])

        # TO DO: Do cross-correlation by using for loops
        for i in range(N):# for every sample
            for j in range(F): # we will produce number of filter output 
                for k in range(out_H): #Height of output calculated according to filter shape,input shape,paddings and stride
                    for l in range(out_W):#Width of output calculated according to filter shape,input shape,paddings and stride
                        out[i][j][k][l]=np.sum(padded_x[i,:, k*self.stride:k*self.stride+FH, l*self.stride:l*self.stride+FW]*self.W[j,:])+self.b[j]
        # filter is applied to input according to strides.

        return out

    def backward(self, dprev):
        dx, dw, db = None, None, None
        padded_x = np.pad(self.x, ((0, 0), (0, 0), (self.padding, self.padding),
                                   (self.padding, self.padding)), 'constant')
        N, C, H, W = self.x.shape
        F, C, FH, FW = self.W.shape
        _, _, out_H, out_W = dprev.shape

        dx_temp = np.zeros_like(padded_x).astype(np.float32)
        dw = np.zeros_like(self.W).astype(np.float32)
        db = np.zeros_like(self.b).astype(np.float32)
        """
        db = None
        dw = None
        dx = None
        """
        # Your implementation here
        for i in range(N):
            for j in range(F):
                db[j]+=np.sum(dprev[i,j]) #out = X*W + b db=1*dprev
                for k in range(out_H):
                    for l in range(out_W):
                        #dx = W*dprev (but to related area)
                        dx_temp[i,:, k*self.stride:k*self.stride+FH, l*self.stride:l*self.stride+FW] += dprev[i,j,k,l]*self.W[j,:]
                        #dw = X*dprev (Related area of X)
                        dw[j]+=padded_x[i,:, k*self.stride:k*self.stride+FH, l*self.stride:l*self.stride+FW]*dprev[i,j,k,l]
        
        dx = dx_temp[:,:,self.padding:self.padding+H,self.padding:self.padding+W]
        #padded x resized to the normal X shape
        
        self.db = db.copy()
        self.dW = dw.copy()
        return dx, dw, db

