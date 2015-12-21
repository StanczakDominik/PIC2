import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numba
from numba import jit
import time

L=1
NX = 256
grid, dx = np.linspace(0,L,NX, retstep=True, endpoint=False)

N=3200
SOR_omega = 2/(1+np.pi/NX)
SOR_L2_target = 5e-5

T = 0.01
NT = 801
timegrid, dt = np.linspace(0,T,NT, retstep=True)

field_scale=5000

charge_grid = np.zeros_like(grid)
charge_history = np.zeros((NT, NX))
electric_field_grid = np.zeros_like(grid)
electric_field_history = np.zeros((NT, NX))
potential_history = np.zeros((NT, NX))
iterations_history = np.zeros(NT)
l2_diff_history = np.zeros((NT, 20000))

timestamp = time.strftime("%y-%m-%d_%H-%M-%S")
