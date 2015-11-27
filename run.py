import numpy as np
import matplotlib.pyplot as plt
L=1
NX = 32
grid, dx = np.linspace(0,L,NX, retstep=True)

T = 10
NT=101
timegrid, dt = np.linspace(0,T,NT, retstep=True)



class ParticleSpecies:
    def __init__(self, N, PositionDistribution, VelocityDistribution, Mass, Charge):
        self.iteration=0
        self.N=N
        self.r = PositionDistribution(N)
        self.CleanupPositions()

        self.v = VelocityDistribution(N)
        self.m = np.ones(N)*Mass
        self.q = np.ones(N)*Charge

        self.r_history = np.zeros((NT+1,N))
        self.v_history = np.zeros((NT+1,N))
        self.r_history[0] = self.r
        self.v_history[0] = self.v
    def Acceleration(self, ElectricField):
        return ElectricField(self.r)*self.q/self.m

    def SimpleStep(self, dt, ElectricField):
        return self.r + self.v*dt, self.v+self.Acceleration(ElectricField)*dt

    def CleanupPositions(self):
        self.r = self.r % L

    def Update(self, dt, ElectricField, Step):
        self.r, self.v = Step(dt, ElectricField)
        self.CleanupPositions()
        self.iteration+=1
        self.r_history[self.iteration]=self.r
        self.v_history[self.iteration]=self.v
    def TextDiagnostics(self):
        print (self.r)
        print (self.v)
        print (self.m)
        print (self.q)

    def PositionPlotDiagnostics(self):
        plt.hist(self.r, bins=100)
        plt.grid()
        plt.xlabel('x')
        plt.show()

    def PhasePlotDiagnostics(self):
        plt.plot(self.r_history[:self.iteration+1],self.v_history[:self.iteration+1], '-')
        plt.xlim(0,L)
        plt.grid()
        plt.xlabel('x')
        plt.ylabel(r'v_x')
        plt.show()

def RandomGaussian(N):
    return np.random.normal(0, 1, N)

def NoElectricField(r):
    return np.zeros_like(r)

def UniformElectricField(r):
    """
    Takes particle positions and returns their E acceleration
    Simple model assumes acceleration by a constant factor
    """
    return np.ones_like(r)

def RampElectricField(r):
    return -(r-L/2)

def SinElectricField(r):
    return -np.sin((r-L/2))

N=200
# Electrons = ParticleSpecies(N, RandomGaussian, RandomGaussian, 1, -1)
Positrons = ParticleSpecies(N, RandomGaussian, RandomGaussian, 1, 1)
Species = [Positrons]
for t in timegrid:
    for species in Species:
        # species.TextDiagnostics()
        # print("Update")
        species.Update(dt, SinElectricField, species.SimpleStep)
        # species.TextDiagnostics()
for species in Species:
    species.PhasePlotDiagnostics()
