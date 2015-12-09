from shared import *

def NonRandomUniform(N):
    return np.linspace(0,1,N)

def negative_ones(N):
    return -np.ones(N)


def RandomGaussian(N):
    return np.random.normal(0, 1, N)

def RandomUniform(N):
    return np.random.random(N)

def RandomUniformVel(N):
    return RandomUniform(N)-0.5
