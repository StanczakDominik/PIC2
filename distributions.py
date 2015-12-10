from shared import *

def NonRandomUniform(N):
    return np.linspace(0,L,N)

def negative_ones(N):
    return -np.ones(N)*L


def RandomGaussian(N):
    return np.random.normal(0, 1, N)*L

def RandomUniform(N):
    return np.random.random(N)*L

def RandomUniformVel(N):
    return (RandomUniform(N)-0.5)*L
